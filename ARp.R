rm(list = ls())
repenv <- Sys.getenv("SLURM_ARRAY_TASK_ID")
i <- as.numeric(repenv)

library(Rcpp, lib.loc = 'packages')
library(RcppArmadillo, lib.loc = 'packages')
library(RcppEigen, lib.loc = 'packages')
library(rstan, lib.loc = 'packages')
source('RFuns.R')
sourceCpp('ARP.cpp')

set.seed(i)

T <- 200
initialBatch <- 50
updateSize <- 25
lags <- 3
mu <- rnorm(1)
H <- 5

stationary <- FALSE
sd <- 1
while(!stationary){
  phi <- rnorm(lags, 0, sd)
  roots <- polyroot(c(1, -phi))
  if(all(Mod(roots) > 1)){
    stationary <- TRUE
  } else {
    sd <- sd * 0.95
  }
}

sigmaSq <- 1/rgamma(1, 5, 5)
x <- rnorm(T+100+H)
for(t in (lags+1):(T+100+H)){
  x[t] <- mu + sum(phi * (x[t-(1:lags)] -mu)) + rnorm(1, 0, sqrt(sigmaSq))
}

qplot(1:T, x[1:T + 100], geom = 'line')

dim <- 2 + lags

priorMean <- rep(0, dim)
priorVar <- diag(10, dim)
priorVarInv <- solve(priorVar)
priorLinv <- solve(t(chol(priorVar)))

batches <- (T - initialBatch) / updateSize
data <- c(100, seq(100 + initialBatch, T+100, length.out = batches+1))

forecast <- NULL
support <- seq(min(x)-sd(x), max(x)+sd(x), length.out = 1000)

# MCMC for exact forecast densities
for(t in data[2]:data[batches+1]){
  
  # Fit MCMC
  MCMCfit <- ARpMCMCallMH(data = x[(data[1]+1 - lags):t],
                          reps = 15000,
                          draw = rep(0, dim),
                          hyper = list(mean = priorMean, varInv = priorVarInv),
                          lags = lags)$draws[10001:15000,]

  # Set up forecast density
  densMCMC <- rep(0, 1000)
  # Forecast
  for(i in 1:1000){
      sigSq <- exp(MCMCfit[5*i, 1])
      mu <- MCMCfit[5*i, 2]
      phi <- MCMCfit[5*i, 3:(2 + lags)]
      mean <- mu + phi %*% (x[(data[t+1]+s+1-(1:lags))] - mu)
      densMCMC <- densMCMC + dnorm(support, mean, sqrt(sigSq)) / 1000
    }
    lower <- max(which(support < x[t+1]))
    upper <- min(which(support > x[t+1]))
    
    if(lower == -Inf){
      lsMCMC <- log(densMCMC[upper])
    } else if(upper == Inf) {
      lSMCMC <- log(densMCMC[lower])
    } else {
      lsMCMC <- log(linearInterpolate(support[lower], support[upper], densMCMC[lower], densMCMC[upper], x[t+1]))
    }
    
    forecast <- rbind(forecast,
                      data.frame(t = t + 1,
                                 ls = lsMCMC,
                                 inference = 'MCMC',
                                 K = 1:K,
                                 id = i))
}


## Apply VB approximations with K = 1, 2, or 3 component mixture approximations
for(K in 1:3){
  
  # Initial Lambda
  lambda <- c(rep(c(-1, rep(0, lags+1)), K) + rnorm(dim*K, 0, 0.1),
              rnorm(dim*K, -1, 0.1),
              rep(1, K))
              
  VBfit <- matrix(0, length(lambda), batches)
  UVBfit <- VBfit
  
  for(t in 1:batches){
  
    # VB Fits
    #If K = 1, use the reparam gradients
    if(K == 1){
      VBfit[,1] <- fitVB(data = x[(data[1]+1 - lags):data[2]],
                         lambda = lambda,
                         model = gradARP,
                         S = 25,
                         dimTheta = dim,
                         mean = priorMean,
                         Linv = priorLinv,
                         lags = lags)$lambda
      
    } else {
      # Otherwise use score gradients
      VBfit[,1] <- fitVBScore(data = x[(data[1]+1 - lags):data[2]],
                              lambda = lambda,
                              model = singlePriorMixApprox,
                              dimTheta = dim,
                              mix = K,
                              S = 25,
                              mean = priorMean,
                              varInv = priorVarInv,
                              lags = lags)$lambda
    }
    
    # UVB fits, At time 1, VB = UVB
    if(t == 1){
      UVBfit[,1] <- VBfit[,1]
    } else {
      # Otherwise apply UVB
      if(K == 1){
        # Reparam gradients
        updateMean <- UVBfit[1:dim, t-1]
        updateU <- matrix(UVBfit[(dim+1):(dim*(dim+1)), t-1], ncol = dim)
        updateLinv <- t(solve(updateU))
        
        UVBfit[,t] <- fitVB(data = x[(data[t]+1-lags):data[t+1]],
                            lambda = UVBfit[,t-1],
                            model = gradARP,
                            S = 25,
                            dimTheta = dim,
                            mean = updateMean,
                            Linv = updateLinv,
                            lags = lags)$lambda
        
      } else {
        # Score gradients
        updateMean <- matrix(UVBfit[1:(dim*K), t-1], ncol = K)
        updateVarInv <- array(0, dim = c(dim, dim, K))
        dets <- rep(0, K)
        for(k in 1:K){
          updateVarInv[,,k] <- diag(exp(UVBfit[dim*K + (dim-1)*k + 1:dim])^(-2))
          dets[k] <- 1 / prod(exp(UVBfit[dim*K + (dim-1)*k + 1:dim]))
        }
        updateZ <- UVBfit[dim*K*2 + 1:dim] 
        updateWeight <- exp(updateZ) / sum(exp(updateZ))
        
        UVBfit[,t] <- fitVBScore(data = x[(data[t]+1-lags):data[t+1]],
                                 lambda = UVBfit[,t-1],
                                 model = mixPriorMixApprox,
                                 dimTheta = dim,
                                 mix = K,
                                 S = 25,
                                 mean = updateMean,
                                 SigInv = updateVarInv,
                                 dets = dets,
                                 weights = updateWeight,
                                 lags = lags)$lambda
        
      }
      
      
   
    
  }
  
    # Propogate particles forward for one step forecasts
    for(s in 0:(updateSize-1)){

      # Initial Particles
      if(s == 0){
        # Easy sampling when K == 1
        if(K == 1){
          meanVB <- VBfit[1:dim, t]
          uVB <- matrix(VBfit[(dim+1):(dim*(dim+1)), t], ncol = dim)
          drawVB <- mvtnorm::rmvnorm(1000, meanVB, t(uVB) %*% uVB)
          qVB <- mvtnorm::dmvnorm(drawVB, meanVB, t(uVB) %*% uVB)
          pVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], drawVB, priorMean, priorVarInv, lags)
          wVB <- pVB / qVB
          wVBNorm <- wVB / sum(wVB)
          
          meanUVB <- UVBfit[1:dim, t]
          uUVB <- matrix(UVBfit[(dim+1):(dim*(dim+1)), t], ncol = dim)
          drawUVB <- mvtnorm::rmvnorm(1000, meanUVB, t(uUVB) %*% uUVB)
          qUVB <- mvtnorm::dmvnorm(drawUVB, meanUVB, t(uUVB) %*% uUVB)
          pUVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], drawUVB, priorMean, priorVarInv, lags)
          wUVB <- pUVB / qUVB
          wUVBNorm <- wUVB / sum(wUVB)
        } else {
          # Mixture sampling is a bit more difficult
          meanVB <- matrix(VBfit[1:(dim*K), t-1], ncol = K)
          varVB <-  array(0, dim = c(dim, dim, K))
          for(k in 1:K){
            varVB[,,k] <- diag(exp(VBfit[dim*K + (dim-1)*k + 1:dim])^(2))
          }
          ZVB <- VBfit[dim*K*2 + 1:dim] 
          piVB <- exp(ZVB) / sum(exp(ZVB))
          
          drawVB <- matrix(0, 1000, dim)
          qVB <- rep(0, 1000)
          for(i in 1:1000){
            group <- sample(1:K, 1, prob = piVB)
            drawVB[i, ] <- mvtnorm::rmvnorm(1, meanVB[,group], varVB[,,group])
            
            for(k in 1:K){
              qVB[i] <- qVB[i] + piVB[k] * mvtnorm::dmvnorm(drawVB[i,], meanVB[,k], varVB[,,k])
            }
          }
          pVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], drawVB, priorMean, priorVarInv, lags)
          wVB <- pVB / qVB
          wVBNorm <- wVB / sum(wVB)
          
          # UVB
          meanUVB <- matrix(UVBfit[1:(dim*K), t-1], ncol = K)
          varUVB <-  array(0, dim = c(dim, dim, K))
          for(k in 1:K){
            varUVB[,,k] <- diag(exp(UVBfit[dim*K + (dim-1)*k + 1:dim])^(2))
          }
          ZUVB <- UVBfit[dim*K*2 + 1:dim] 
          piUVB <- exp(ZUVB) / sum(exp(ZUVB))
          
          drawUVB <- matrix(0, 1000, dim)
          qUVB <- rep(0, 1000)
          for(i in 1:1000){
            group <- sample(1:K, 1, prob = piUVB)
            drawUVB[i, ] <- mvtnorm::rmvnorm(1, meanUVB[,group], varUVB[,,group])
            
            for(k in 1:K){
              qUVB[i] <- qUVB[i] + piUVB[k] * mvtnorm::dmvnorm(drawUVB[i,], meanUVB[,k], varUVB[,,k])
            }
          }
          pUVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], drawUVB, priorMean, priorVarInv, lags)
          wUVB <- pUVB / qUVB
          wUVBNorm <- wUVB / sum(wUVB)
        }
       
      } else {
        # Update Particles is always simple
        pVB <- ARLikelihood(x[data[t+1] + (s-lags):s], drawVB, lags)
        wVB <- wVB * pVB
        wVBNorm <- wVB / sum(wVB)
      
        pUVB <- ARLikelihood(x[data[t+1] + (s-lags):s], drawVB, lags)
        wUVB <- wUVB * pUVB
        wUVBNorm <- wUVB / sum(wUVB)
      }
    
      # Set up forecast densities
      densVB <- densUVB <- rep(0, 1000)
      # Create forecast densities by averaging over the 1000 draws
      for(i in 1:1000){
        sigSq <- exp(drawVB[i, 1])
        mu <- drawVB[i, 2]
        phi <- drawVB[i, 3:(2 + lags)]
        mean <- mu + phi %*% (x[(data[t+1]+s+1-(1:lags))] - mu)
        densVB <- densVB + dnorm(support, mean, sqrt(sigSq)) * wVBNorm[i]
      
        sigSq <- exp(drawUVB[i, 1])
        mu <- drawUVB[i, 2]
        phi <- drawUVB[i, 3:(2 + lags)]
        mean <- mu + phi %*% (x[(data[t+1]+s+1-(1:lags))] - mu)
        densUVB <- densUVB + dnorm(support, mean, sqrt(sigSq)) * wUVBNorm[i]
      }
      lower <- max(which(support < x[data[t+1]+s+1]))
      upper <- min(which(support > x[data[t+1]+s+1]))
    
      if(lower == -Inf){
        lsVB <- log(densVB[upper])
        lsUVB <- log(densUVB[upper])
      } else if(upper == Inf) {
        lsVB <- log(densVB[lower])
        lsUVB <- log(densUVB[lower])
      } else {
        lsVB <- log(linearInterpolate(support[lower], support[upper], densVB[lower], densVB[upper], x[data[t+1]+s+1]))
        lsUVB <- log(linearInterpolate(support[lower], support[upper], densUVB[lower], densUVB[upper], x[data[t+1]+s+1]))
      }
    
      forecast <- rbind(forecast,
                        data.frame(t = data[t+1] + s + 1,
                                   ls = c(lsVB, lsUVB),
                                   inference = c('VB', 'UVB'),
                                   K = K,
                                   id = i))
    }
  }
}
  
