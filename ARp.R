rm(list = ls())
#repenv <- Sys.getenv("SLURM_ARRAY_TASK_ID")
#rep <- as.numeric(repenv)

library(Rcpp)#, lib.loc = 'packages')
library(RcppArmadillo)#, lib.loc = 'packages')
library(RcppEigen)#, lib.loc = 'packages')
library(rstan)#, lib.loc = 'packages')
source('RFuns.R')
sourceCpp('ARP.cpp')

forecast <- NULL

min <- 1
max <- 3

T <- 200
initialBatch <- 50
updateSize <- 25
batches <- (T - initialBatch) / updateSize + 1
data <- c(100, seq(100 + initialBatch, T+100, length.out = batches))

lags <- 3
MCsamples <- 500

dim <- 2 + lags
priorMean <- rep(0, dim)
priorVar <- diag(10, dim)
priorVarInv <- solve(priorVar)
priorLinv <- solve(t(chol(priorVar)))


for(rep in min:max){

if(rep == min){
  startTime <- Sys.time()
} else if(rep == min + 1){
  timePerIter <- Sys.time() - startTime
  class(timePerIter) <- 'numeric'
  if(attr(timePerIter, 'units') == 'mins'){
    attr(timePerIter, 'units') = 'secs'
    timePerIter <- timePerIter * 60
  } else if(attr(timePerIter, 'units') == 'hours'){
    attr(timePerIter, 'units') = 'hours'
    timePerIter <- timePerIter * 3600
  }
  print(paste0('Time Per Iter: ', round(timePerIter / 60, 2), ' minutes. Estimated Finishing Time: ', Sys.time() + (max - min) * timePerIter))
} else {
  timePerIter <- (Sys.time() - startTime) / (rep - min)
  class(timePerIter) <- 'numeric'
  if(attr(timePerIter, 'units') == 'mins'){
    attr(timePerIter, 'units') = 'secs'
    timePerIter <- timePerIter * 60
  } else if(attr(timePerIter, 'units') == 'hours'){
    attr(timePerIter, 'units') = 'hours'
    timePerIter <- timePerIter * 3600
  }
  print(paste0('Iteration: ', rep, ' Estimated Finishing Time: ', Sys.time() + (max + 1 - rep) * timePerIter))
}
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
for(t in data[2]:data[batches+1]){
  
  # Fit MCMC
  MCMCfit <- ARpMCMCallMH(data = x[(data[1]+1 - lags):t],
                          reps = 15000,
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
      mean <- mu + phi %*% (x[t+1-(1:lags)] - mu)
      densMCMC <- densMCMC + dnorm(support, mean, sqrt(sigSq)) / MCsamples
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
                                 K = 1:3,
                                 runTime = NA,
                                 ESS = NA,
                                 ELBO = NA,
                                 id = rep))
}

## Apply VB approximations with K = 1, 2, or 3 component mixture approximations
for(K in 1:3){
  for(Update in 0:1){
    for(IS in 0:1){
      
      # Start Time
      startTimeVB <- Sys.time()
      
      # Initial Lambda
      if(K == 1){
        lambda <- c(rnorm(1, -1, 0.1), rnorm(lags+1, 0, 0.1), chol(diag(1, dim)))
      } else {
        lambda <- c(rep(c(-1, rep(0, lags+1)), K) + rnorm(dim*K, 0, 0.1),
                    rnorm(dim*K, -1, 0.1),
                    rep(1, K))
        
      }
      
      VBfit <- matrix(0, length(lambda), batches)

      for(t in 1:batches){
        if(!Update){
          # VB Fits
          #If K = 1, use the reparam gradients
          if(K == 1){
            VB <- fitVB(data = x[(data[1]+1 - lags):data[t+1]],
                               lambda = lambda,
                               model = gradARP,
                               S = 25,
                               dimTheta = dim,
                               mean = priorMean,
                               Linv = priorLinv,
                               lags = lags)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[min(5000, VB$iter)]
            
          } else {
            # Otherwise use score gradients
            VB <- fitVBScore(data = x[(data[1]+1 - lags):data[t+1]],
                                    lambda = lambda,
                                    model = singlePriorMixApprox,
                                    dimTheta = dim,
                                    mix = K,
                                    S = 25,
                                    mean = priorMean,
                                    varInv = priorVarInv,
                                    lags = lags)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[min(5000, VB$iter)]
            
          }
        } else {
          # UVB fits, At time 1, VB = UVB
          if(t == 1){
            if(K == 1){
              VB <- fitVB(data = x[(data[1]+1 - lags):data[t+1]],
                                 lambda = lambda,
                                 model = gradARP,
                                 S = 25,
                                 dimTheta = dim,
                                 mean = priorMean,
                                 Linv = priorLinv,
                                 lags = lags)
              
              VBfit[,t] <- VB$lambda
              ELBO <- VB$LB[min(5000, VB$iter)]
              
              
            } else {
              # Otherwise use score gradients
              VB <- fitVBScore(data = x[(data[1]+1 - lags):data[t+1]],
                                      lambda = lambda,
                                      model = singlePriorMixApprox,
                                      dimTheta = dim,
                                      mix = K,
                                      S = 25,
                                      mean = priorMean,
                                      varInv = priorVarInv,
                                      lags = lags)
              
              
              VBfit[,t] <- VB$lambda
              ELBO <- VB$LB[min(5000, VB$iter)]
            }
          } else {
            # Otherwise apply UVB
            if(K == 1){
              # Reparam gradients
              updateMean <- VBfit[1:dim, t-1]
              updateU <- matrix(VBfit[(dim+1):(dim*(dim+1)), t-1], ncol = dim)
              updateLinv <- t(solve(updateU))
              
              VB <- fitVB(data = x[(data[t]+1-lags):data[t+1]],
                          lambda = VBfit[,t-1],
                          model = gradARP,
                          S = 25,
                          dimTheta = dim,
                          mean = updateMean,
                          Linv = updateLinv,
                          lags = lags)
              
              
              VBfit[,t] <- VB$lambda
              ELBO <- VB$LB[min(5000, VB$iter)]
  
            } else {
              # Score gradients
              updateMean <- matrix(VBfit[1:(dim*K), t-1], ncol = K)
              updateVarInv <- array(0, dim = c(dim, dim, K))
              dets <- rep(0, K)
              for(k in 1:K){
                sd <- exp(VBfit[dim*K + dim*(k-1) + 1:dim, t-1])
                updateVarInv[,,k] <- diag(1/sd^2)
                dets[k] <- 1 / prod(sd)
              }
              updateZ <- UVBfit[dim*K*2 + 1:K, t-1] 
              updateWeight <- exp(updateZ) / sum(exp(updateZ))
              
              VB<- fitVBScore(data = x[(data[t]+1-lags):data[t+1]],
                                       lambda = VBfit[,t-1],
                                       model = mixPriorMixApprox,
                                       dimTheta = dim,
                                       mix = K,
                                       S = 25,
                                       mean = updateMean,
                                       SigInv = updateVarInv,
                                       dets = dets,
                                       weights = updateWeight,
                                       lags = lags)
              
              VBfit[,t] <- VB$lambda
              ELBO <- VB$LB[min(5000, VB$iter)]
              
            }
            
        }
        
       
        
       
          
          
          
        }
        
        
        # Propogate particles forward for one step forecasts
        for(s in 0:(updateSize-1)){
          if(t == batches & s > 0){
            break
          }
          # Initial Particles
          if(s == 0){
            # Easy sampling when K == 1
            if(K == 1){
              mean <- VBfit[1:dim, t]
              u <- matrix(VBfit[(dim+1):(dim*(dim+1)), t], ncol = dim)
              draw <- mvtnorm::rmvnorm(MCsamples, mean, t(u) %*% u)
       
              if(IS){
                qVB <- mvtnorm::dmvnorm(draw, mean, t(u) %*% u)
                pVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], draw, priorMean, priorVarInv, lags)
                wVB <- pVB / qVB
                weight <- wVB / sum(wVB)
              } else {
                weight <- rep(1 / MCsamples, MCsamples)
              }
              
              
            } else {
              # Mixture sampling is a bit more difficult
              mean <- matrix(VBfit[1:(dim*K), t], ncol = K)
              var <-  array(0, dim = c(dim, dim, K))
              for(k in 1:K){
                var[,,k] <- diag(exp(VBfit[dim*K + dim*(k-1) + 1:dim, t])^2)
              }
              Z <- VBfit[dim*K*2 + 1:K, t] 
              pi <- exp(Z) / sum(exp(Z))
              
              draw <- matrix(0, MCsamples, dim)
              for(i in 1:MCsamples){
                group <- sample(1:K, 1, prob = pi)
                draw[i, ] <- mvtnorm::rmvnorm(1, mean[,group], var[,,group])
              }
              
              if(IS){
                qVB <- rep(0, MCsamples)
                for(i in 1:MCsamples){
                  for(k in 1:K){
                    qVB[i] <- qVB[i] + pi[k] * mvtnorm::dmvnorm(draw[i,], mean[,k], var[,,k])
                  }
                }
                pVB <- ARjointDens(x[(data[1]+1 - lags):(data[t+1])], draw, priorMean, priorVarInv, lags)
                wVB <- pVB / qVB
                weight <- wVB / sum(wVB)
              } else {
                weight <- rep(1 / MCsamples, MCsamples)
              }
            }
            
          } else if(IS){
            # Update Particles is always simple
            pVB <- ARLikelihood(x[data[t+1] + (s-lags):s], draw, lags)
            wVB <- wVB * pVB
            weight <- wVB / sum(wVB)
          }
          
          # Set up forecast densities
          densVB <- rep(0, 1000)
          # Create forecast densities by averaging over the 1000 draws
          for(i in 1:MCsamples){
            sigSq <- exp(draw[i, 1])
            mu <- draw[i, 2]
            phi <- draw[i, 3:(2 + lags)]
            mean <- mu + phi %*% (x[(data[t+1]+s+1-(1:lags))] - mu)
            densVB <- densVB + dnorm(support, mean, sqrt(sigSq)) * weight[i]
          }
          lower <- max(which(support < x[data[t+1]+s+1]))
          upper <- min(which(support > x[data[t+1]+s+1]))
          
          if(lower == -Inf){
            lsVB <- log(densVB[upper])
          } else if(upper == Inf) {
            lsVB <- log(densVB[lower])
          } else {
            lsVB <- log(linearInterpolate(support[lower], support[upper], densVB[lower], densVB[upper], x[data[t+1]+s+1]))
          }
          
          runTime <- Sys.time() - startTimeVB
          class(runTime) <- 'numeric'
          if(attr(runTime, 'units') == 'mins'){
            attr(runTime, 'units') = 'secs'
            runTime <- runTime * 60
          } else if(attr(runTime, 'units') == 'hours'){
            attr(runTime, 'units') = 'hours'
            runTime <- runTime * 3600
          }
          
          forecast <- rbind(forecast,
                            data.frame(t = data[t+1] + s + 1,
                                       ls = lsVB,
                                       inference = paste0(ifelse(Update, 'U', ''), 'VB', ifelse(IS, '-IS', '')),
                                       K = K,
                                       runTime = runTime,
                                       ESS = 1 / sum(weight^2),
                                       ELBO = ELBO,
                                       id = rep))
        }
      }
    }
  }
}
}

write.csv(forecast, paste0('results/AR3_', min, '_', max, '.csv'), row.names = FALSE)


forecast %>% 
  filter(t <= 300) %>%
  spread(inference, ls) %>%
  mutate(`VB-IS` = `VB-IS` - MCMC,
         `UVB-IS` = `UVB-IS` - MCMC,
         VB = VB - MCMC,
         UVB = UVB - MCMC,
         t = t - 100) %>%
  select(t, K, `VB-IS`, `UVB-IS`, VB, UVB) %>%
  gather(inference, diff, -t, -K) %>%
  mutate(inference = factor(inference, levels = c('VB', 'VB-IS', 'UVB', 'UVB-IS'))) %>%
  group_by(inference, t, K) %>%
  filter(!is.na(diff)) %>%
  summarise(med = median(diff)) %>%
  ggplot() + geom_line(aes(t, med)) +
  geom_hline(aes(yintercept = 0), colour = 'red') +
  facet_grid(K ~ inference) + 
  theme_bw() + 
  labs(x = 't', y = 'Median Difference in Logscores (Approximate Inference - Exact)')

filter(t <= 300) %>%
  spread(inference, ls) %>%
  mutate(`VB-IS` = `VB-IS` - MCMC,
         `UVB-IS` = `UVB-IS` - MCMC,
         VB = VB - MCMC,
         UVB = UVB - MCMC,
         t = t - 100) %>%
  select(t, K, `VB-IS`, `UVB-IS`, VB, UVB) %>%
  gather(inference, diff, -t, -K) %>%
  mutate(inference = factor(inference, levels = c('VB', 'VB-IS', 'UVB', 'UVB-IS'))) %>%
  group_by(inference, t, K) %>%
  filter(!is.na(diff)) %>%
  summarise(med = median(diff)) %>%
  ggplot() + geom_line(aes(t, med)) +
  geom_hline(aes(yintercept = 0), colour = 'red') +
  facet_grid(K ~ inference) + 
  theme_bw()
  
forecast %>%
  filter(inference != 'MCMC' & K == 1) %>%
  ggplot() + geom_line(aes(t, runTime)) + facet_wrap(~inference)
  
