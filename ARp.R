rm(list = ls())
repenv <- Sys.getenv("SLURM_ARRAY_TASK_ID")
rep <- as.numeric(repenv)

library(Rcpp, lib.loc = 'packages')
library(RcppArmadillo, lib.loc = 'packages')
library(RcppEigen, lib.loc = 'packages')
library(rstan, lib.loc = 'packages')
source('RFuns.R')
sourceCpp('ARP.cpp')

forecast <- list()
counter <- 1

T <- 300
initialBatch <- 100
updateSize <- 25
batches <- (T - initialBatch) / updateSize + 1
data <- c(100, seq(100 + initialBatch, T+100, length.out = batches))

lags <- 3
MCsamples <- 2000

dim <- 2 + lags
priorMean <- rep(0, dim)
priorVar <- diag(10, dim)
priorVarInv <- solve(priorVar)
priorLinv <- solve(t(chol(priorVar)))

set.seed(rep)


mu <- rnorm(1)

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
x <- rnorm(lags, 0, sqrt(sigmaSq))
for(t in (lags+1):(T+100+1)){
  x <- c(x, mu + sum(phi * (x[t-(1:lags)] -mu)) + rnorm(1, 0, sqrt(sigmaSq)))
}

#qplot(1:T, x[1:T + 100], geom = 'line')
support <- seq(min(x)-sd(x), max(x)+sd(x), length.out = 1000)

# MCMC for exact forecast densities
for(t in 1:batches){
  
  # Fit MCMC
  MCMCfit <- ARpMCMCallMH(data = x[(data[1]+1 - lags):data[t+1]],
                          reps = 25000,
                          draw = rep(0, dim),
                          hyper = list(mean = priorMean, varInv = priorVarInv),
                          lags = lags)$draws[floor(seq(10001, 15000, length.out = MCsamples)), ]

  # Set up forecast density
  densMCMC <- rep(0, 1000)
  # Forecast
  for(i in 1:MCsamples){
      sigSq <- exp(MCMCfit[i, 1])
      mu <- MCMCfit[i, 2]
      phi <- MCMCfit[i, 3:(2 + lags)]
      mean <- mu + phi %*% (x[data[t+1]+1-(1:lags)] - mu)
      densMCMC <- densMCMC + dnorm(support, mean, sqrt(sigSq)) / MCsamples
    }
    lower <- max(which(support < x[data[t+1]+1]))
    upper <- min(which(support > x[data[t+1]+1]))
    
    if(lower == -Inf){
      lsMCMC <- log(densMCMC[upper])
    } else if(upper == Inf) {
      lSMCMC <- log(densMCMC[lower])
    } else {
      lsMCMC <- log(linearInterpolate(support[lower], support[upper], densMCMC[lower], densMCMC[upper], x[data[t+1]+1]))
    }
    
    forecast[[counter]] <- data.frame(t = data[t + 1],
                                 ls = lsMCMC,
                                 inference = 'MCMC',
                                 K = 1:3,
                                 runTime = NA,
                                 ESS = NA,
                                 ELBO = NA,
                                 id = rep)
    counter <- counter + 1
}

## Apply VB approximations with K = 1, 2, or 3 component mixture approximations
for(K in 1:3){
  for(Update in 0:1){
    # Start Time
    ISTotal <- 0
    VBTotal <- 0
    # Initial Lambda
    lambda <- c(rep(c(-1, rep(0, lags+1)), K) + rnorm(dim*K, 0, 0.1),
                  rnorm(dim*K, -1, 0.1),
                  rep(1, K))
      
      
    VBfit <- matrix(0, length(lambda), batches)
    for(t in 1:batches){
      startVB <- Sys.time()
      # VB Fits
      if(t == 1 | !Update){
        VB <- fitVBScore(data = x[(data[1]+1 - lags):data[t+1]],
                         lambda = lambda,
                         model = AR3VB,
                         dimTheta = 5,
                         mix = K,
                         S = 25,
                         mean = matrix(priorMean, ncol = 1),
                         SigInv = array(priorVarInv, dim = c(5, 5, 1)),
                         weights = 1,
                         dets = sqrt(det(priorVarInv)))
          
          VBfit[,t] <- VB$lambda
          ELBO <- VB$LB[VB$iter - 1]
        
      } else {
        # UVB
        updateMean <- fitMean
        updateVarInv <- array(0, dim = c(dim, dim, K))
        dets <- rep(0, K)
        for(k in 1:K){
          sd <- fitSd[,k]
          updateVarInv[,,k] <- diag(1/sd^2)
          dets[k] <- 1 / prod(sd)
        }
        updateWeight <- fitWeight
          
        VB<- fitVBScore(data = x[(data[t]+1-lags):data[t+1]],
                        lambda = VBfit[,t-1],
                        model = AR3VB,
                        dimTheta = dim,
                        mix = K,
                        S = 25,
                        mean = updateMean,
                        SigInv = updateVarInv,
                        dets = dets,
                        weights = updateWeight)
        VBfit[,t] <- VB$lambda
        ELBO <- VB$LB[VB$iter - 1]
      }
        #If K = 1, use the reparam gradients
       
      endVB <- Sys.time() - startVB
      VBTotal <- VBTotal + as.numeric(endVB)
        
      # Draw from VB
      fitMean <- matrix(VBfit[1:(dim*K), t], ncol = K)
      fitSd <- matrix(exp(VBfit[dim*K + 1:(dim*K), t]), ncol = K)
      fitZ <- VBfit[2*dim*K + 1:K, t]
      fitWeight <- exp(fitZ) / sum(exp(fitZ))
      draw <- matrix(0, MCsamples, dim)
      for(i in 1:MCsamples){
        group <- sample(1:K, 1, prob = fitWeight)
        draw[i, ] <- mvtnorm::rmvnorm(1, fitMean[,group], diag(fitSd[,group]^2))
      }
      
      # Importance Sample Weights
      startIS <- Sys.time()
            
      qVB <- rep(0, MCsamples)
      for(k in 1:K){
        qVB <- qVB + fitWeight[k] *  mvtnorm::dmvnorm(draw, fitMean[,k], diag(fitSd[,group]^2))
      }
      pVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], draw, priorMean, priorVarInv, lags)
      wVB <- pVB / qVB
      weightIS <- wVB / sum(wVB)
      endIS <- Sys.time() - startIS
      ISTotal <- ISTotal + as.numeric(endIS)
            
      # Set up forecast densities
      densVB <- rep(0, 1000)
      densIS <- rep(0, 1000)
      # Create forecast densities by averaging over the 1000 draws
      for(i in 1:MCsamples){
        sigSq <- exp(draw[i, 1])
        mu <- draw[i, 2]
        phi <- draw[i, 3:(2 + lags)]
        mean <- mu + phi %*% (x[(data[t+1]+1-(1:lags))] - mu)
        densVB <- densVB + dnorm(support, mean, sqrt(sigSq)) / MCsamples
        densIS <- densIS + dnorm(support, mean, sqrt(sigSq)) * weightIS[i]
      }
      lower <- max(which(support < x[data[t+1]+1]))
      upper <- min(which(support > x[data[t+1]+1]))
        
      if(lower == -Inf){
        lsVB <- log(densVB[upper])
        lsIS <- log(densIS[upper])
      } else if(upper == Inf) {
        lsVB <- log(densVB[lower])
        lsIS <- log(densIS[lower])
      } else {
        lsVB <- log(linearInterpolate(support[lower], support[upper], densVB[lower], densVB[upper], x[data[t+1]+1]))
        lsIS <- log(linearInterpolate(support[lower], support[upper], densIS[lower], densIS[upper], x[data[t+1]+1]))
      }
        
          
      forecast[[counter]] <- data.frame(t = data[t+1],
                                        ls = c(lsVB, lsIS),
                                        inference = paste0(ifelse(Update, 'U', ''), c('VB', 'VB-IS')),
                                        K = K,
                                        runTime = c(VBTotal, VBTotal + ISTotal),
                                        ESS = c(MCsamples, 1 / sum(weightIS^2)),
                                        ELBO = ELBO,
                                        id = rep)
      counter <- counter + 1
    }
  }
}


forecast <- do.call(rbind.data.frame, forecast)

write.csv(forecast, paste0('results/AR3_', rep, '.csv'), row.names = FALSE)

