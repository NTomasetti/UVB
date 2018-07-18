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
MCsamples <- 50

dim <- 2 + lags
priorMean <- rep(0, dim)
priorVar <- diag(10, dim)
priorVarInv <- solve(priorVar)
priorLinv <- solve(t(chol(priorVar)))

for(rep in 1:10){
  

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

## Apply IS-UVB approximations with K = 1, 2, or 3 component mixture approximations
for(K in 1:3){
  # Start Time
  VBTotal <- 0
  # Initial Lambda
  lambda <- c(rep(c(-1, rep(0, lags+1)), K) + rnorm(dim*K, 0, 0.1),
              rnorm(dim*K, -1, 0.1),
              rep(1, K))
  
  VBfit <- matrix(0, length(lambda), batches)
  for(t in 1:batches){
    print(paste(rep, K, t))
    start <- Sys.time()
    # VB Fits
    if(t == 1){
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
      
      # Calcualte density under the old distribution
      qVB <- rep(0, MCsamples)
      for(k in 1:K){
        qVB <- qVB + fitWeight[k] * mvtnorm::dmvnorm(draw, fitMean[,k], diag(fitSd[,k]^2))
      }
      # Calculate the log-joint distribution through the UVB prior recursion
      if(K == 1){
        pVB <- ARjointDens(x[(data[t]+1 - lags):(data[t+1])], draw, fitMean, diag(c(1 / fitSd^2)), lags)
      } else {
        varInv <- array(0, dim = c(2 + lags, 2 + lags, K))
        dets <- rep(0, K)
        for(k in 1:K){
          varInv[,,k] <- diag(c(1 / fitSd[,k]^2))
          dets[k] <- prod(1 / fitSd[,k])
        }
        pVB <- ARjointDensMixPrior(x[(data[t]+1 - lags):(data[t+1])], draw, fitMean, varInv, dets, fitWeight, lags)
      }
      # Run VB Update
      VB <- ISUVB(lambda = VBfit[,t-1],
                  qScore = AR3Score,
                  samples = draw,
                  dSamples = qVB,
                  logjoint = c(log(pVB)),
                  maxIter = 2000,
                  mix = K)
      
      VBfit[,t] <- VB$lambda
      ELBO <- VB$LB[VB$iter - 1]
    }
    
    endVB <- Sys.time() - start
    if(attr(endVB, 'units') == 'mins'){
      endVB <- endVB * 60
    }
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
    
    # Set up forecast densities
    densVB <- rep(0, 1000)
    # Create forecast densities by averaging over the 1000 draws
    for(i in 1:MCsamples){
      sigSq <- exp(draw[i, 1])
      mu <- draw[i, 2]
      phi <- draw[i, 3:(2 + lags)]
      mean <- mu + phi %*% (x[(data[t+1]+1-(1:lags))] - mu)
      densVB <- densVB + dnorm(support, mean, sqrt(sigSq)) / MCsamples
    }
    lower <- max(which(support < x[data[t+1]+1]))
    upper <- min(which(support > x[data[t+1]+1]))
    
    if(lower == -Inf){
      lsVB <- log(densVB[upper])
    } else if(upper == Inf) {
      lsVB <- log(densVB[lower])
    } else {
      lsVB <- log(linearInterpolate(support[lower], support[upper], densVB[lower], densVB[upper], x[data[t+1]+1]))
    }
    
    
    forecast[[counter]] <- data.frame(t = data[t+1],
                                      ls = lsVB,
                                      inference = 'IS-UVB',
                                      K = K,
                                      runTime = VBTotal,
                                      ESS = NA,
                                      ELBO = ELBO,
                                      id = rep)
    counter <- counter + 1
    
  }
}

}
forecast <- do.call(rbind.data.frame, forecast)
forecast$N <- 50
forecast <- rbind(fc2, forecast)

forecast %>%
  group_by(t, K) %>%
  summarise(ls = mean(ls),
            runTime = mean(runTime)) %>%
  gather(var, result, ls, runTime) %>%
  ggplot() + geom_line(aes(t, result, colour = factor(K))) + facet_wrap(~var, scales = 'free', ncol = 1)



write.csv(forecast, paste0('results/AR3_', rep, '_IS.csv'), row.names = FALSE)

