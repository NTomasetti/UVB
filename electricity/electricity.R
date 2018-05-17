library(tidyverse)
library(lubridate)
library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
source('RFuns.R')
sourceCpp('electricity/electricityVB.cpp')
N <- 20

y <- readRDS('electricity/elecY.RDS')[,1:N]
x <- readRDS('electricity/elecX.RDS')

Tseq <- c(48 * 7, 48 * 7 * 5)
for(t in 1:(62-35)){
  Tseq <- c(Tseq, 48 * (35 + t))
}

K <- c(1, 3)
order <- c(1, 2, 3, 48, 96, 144, 336)
dim <- 3 + length(order)
samples <- 250
switch <- 0
k <- 2
batchSize <- 2

results <- list()
counter <- 1

lambda <- NULL
## Repeat for Each Dynamic model: Log variance, mean, sarima parameters, temperature coefficient
## Generate different random numbers for the mean of each model
for(ki in 1:K[k]){
  lambda <- c(lambda, c(rnorm(dim, 0, 0.1), diag(0.1, dim))) 
}
electricityMod <- elecStd
priorMean <- matrix(0, dim, K[k] + 2*switch)
priorLinv <- array(0, dim = c(dim, dim, K[k] + 2*switch))

if(switch){
  lambda <- c(rep(c(0, 0, 0.1, 0.1, 0.1), 2), lambda) ## Constant Model Log Variance and Mean and p01/p10 terms added
  electricityMod <- elecSwitch
  priorMean[,1] <- 0
  priorLinv[1:2, 1:2, 1] <- solve(chol(diag(10, 2)))
  priorPS <- rep(0.1, N)
}
for(i in 1:K[k]){
  priorMean[,i + 2 * switch] <- 0
  priorLinv[,, i + 2 * switch] <- solve(chol(diag(10, dim)))
}
priorProbK <- matrix(1/K[k], nrow = K[k], ncol = N)

fitMat <- matrix(0, length(lambda), length(Tseq))
for(t in 2:(length(Tseq)-1)){
  print(paste0('t: ', t, ', switch : ', switch, ', k: ', k, ', time: ', Sys.time()))
  
  # Fit model, either via standard VB (t = 2 --> First fit) or updated VB
  # Switching and non switching need different inputs so are split with if(switch)
  if(t == 2){
    if(switch){
      fit <- fitVB(data = y[1:Tseq[2],],
                   lambda = lambda,
                   model = electricityMod,
                   dimTheta = 4 + dim * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.2,
                   priorMean = priorMean,
                   priorLinv = priorLinv,
                   probK = priorProbK, 
                   pS1 = priorPS,
                   order = order,
                   Tn = Tseq[2] - Tseq[1],
                   x = x[1:Tseq[2]],
                   uniformRho = TRUE,
                   batch = batchSize)
    } else {
      fit2 <- fitVB(data = y[1:Tseq[2],],
                   lambda = lambda,
                   model = electricityMod,
                   dimTheta = dim * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.2,
                   priorMean = priorMean,
                   priorLinv = priorLinv,
                   probK = priorProbK, 
                   order = order,
                   Tn = Tseq[2] - Tseq[1],
                   x = x[1:Tseq[2]],
                   batch = batchSize)
    }
  } else {
    if(switch){
      fit <- fitVB(data = y[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                   lambda = c(fit$lambda),
                   model = electricityMod,
                   dimTheta = 4 + dim * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.2,
                   priorMean = updateMean,
                   priorLinv = updateLinv,
                   probK = updateProbK, 
                   pS1 = updateProbS1,
                   order = order,
                   Tn = Tseq[t] - Tseq[t-1],
                   x = x[(Tseq[t-1] - max(order) + 1):Tseq[t]],
                   uniformRho = FALSE,
                   batch = batchSize)
    } else {
      fit <- fitVB(data = y[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                   lambda = c(fit$lambda),
                   model = electricityMod,
                   dimTheta = dim * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.2,
                   priorMean = updateMean,
                   priorLinv = updateLinv,
                   probK = updateProbK, 
                   order = order,
                   Tn = Tseq[t] - Tseq[t-1],
                   x = x[(Tseq[t-1] - max(order) + 1):Tseq[t]],
                   batch = batchSize)
    }
  }
  fitMat[,t] <- fit$lambda

  # Calculate updated priors for theta parameters
  updateMean <- matrix(0, dim, K[k] + 2 * switch)
  updateLinv <- updateL <- array(0, dim = c(dim, dim, K[k] + 2 * switch))
  if(switch){
    updateMean[1:2,1] <- fit$lambda[1:2]
    updateL[1:2, 1:2, 1] <- matrix(c(fit$lambda[3:4], 0, fit$lambda[5]), 2)
    updateLinv[1:2, 1:2, 1] <- solve(updateL[1:2, 1:2, 1])
    updateMean[1:2, 2] <- fit$lambda[6:7]
    updateL[1:2, 1:2, 2] <- matrix(c(fit$lambda[8:9], 0, fit$lambda[10]), 2)
    updateLinv[1:2, 1:2, 2] <- solve(updateL[1:2, 1:2, 2])
  }
  for(ki in 1:K[k]){
    updateMean[,2*switch + ki] <- fit$lambda[10*switch + (ki-1) * dim * (dim + 1) + 1:dim]
    updateL[,,2*switch + ki] <- matrix( fit$lambda[10*switch + (ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
    updateLinv[,,2*switch + ki] <- solve(updateL[,,2*switch + ki])
  }
  
  # Sample theta values
  tFull <- array(0, dim = c(dim, K[k], samples))
  tCons <- matrix(0, samples, 4)
  fcVar <- array(0, dim = c(48, K[k], samples))
  for(i in 1:samples){
    
    if(switch){
      thetaC <- c(updateMean[1:2, 1] + updateL[1:2, 1:2, 1] %*% rnorm(2))
      rho <- 1 / (1 + exp(- c(updateMean[1:2, 2] + updateL[1:2, 1:2, 2] %*% rnorm(2))))
    }
    tCons[i, ] <- c(thetaC, rho)
    
    theta <- matrix(0, dim, K[k])
    for(ki in 1:K[k]){
      theta[, ki] <- updateMean[,2 * switch + ki] + updateL[,,2 * switch + ki] %*% rnorm(dim)
      
      root <- polyroot(c(1, -theta[4:6, ki]))
      if(any(Mod(root) < 1)){
        if(i > 1){
          theta[,ki] <- tFull[,ki,i-1]
        } else {
          stat <- FALSE
          try <- 0
          while(!stat){
            theta[, ki] <- updateMean[,2 * switch + ki] + updateL[,,2 * switch + ki] %*% rnorm(dim)
            root <- polyroot(c(1, -theta[4:6, ki]))
            try <- try + 1
            if(all(Mod(root) > 1)){
              stat <- TRUE
            } else if(try > 10){
              theta[4:6, ki] <- 0
              stat <- TRUE
            }
          }
        }
      }
      
      autocov <- ltsa::tacvfARMA(theta[4:6, ki], sigma2 = exp(theta[1, ki]), maxLag = 48)
      fcVar[, ki, i] <- ltsa::PredictionVariance(autocov, 48)
    }
    tFull[,,i] <- theta
  }
  
  # For each household:
  # 1) Calculate p(k = j) / p(S_T = 1)
  # 2) Create forecast densities per k and combine with above probabilities
  # 3) Calculate updated priors for p(k) and p(s) to be pushed into the next update
  # 4) Evaluate forecast densities
  # Steps 1) and 2) are handled in c++
  support <- seq(min(y[Tseq[t-1]:Tseq[t],]) - sd(y[Tseq[t-1]:Tseq[t],]), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
  updateProbK <- matrix(0, K[k], N)
  updateProbS1 <- rep(0, N)
  for(j in 1:N){
    density <- matrix(0, 5000, 48)
    for(i in 1:samples){
      if(switch){
        fc <- forecastHF(y[1:Tseq[t+1], j], x[1:Tseq[t+1]], tCons[i, 1:2], tCons[i, 3:4], tFull[,,i], fcVar[,,i], order, priorPS[j], priorProbK[,j], support, Tseq[t] - Tseq[1])
        density <- density + fc$density / samples
        updateProbK[,j] <- updateProbK[,j] + fc$pk / samples
        updateProbS1[j] <- updateProbS1[j] + fc$ps / samples
      } else {
        fc <- forecastStandard(y[1:Tseq[t+1], j], x[1:Tseq[t+1]], tFull[,,i], order, priorProbK[,j], fcVar[,,i], support, Tseq[t] - Tseq[1])
        density <- density + fc$density / samples
        updateProbK[,j] <- updateProbK[,j] + fc$pk / samples
      }
    }
    for(h in 1:48){
      lower <- max(which(support < y[Tseq[t] + h, j]))
      upper <- min(which(support > y[Tseq[t] + h, j]))
      dens <- linearInterpolate(support[lower], support[upper], density[lower, h], density[upper, h], y[Tseq[t] + h, j])
      
      map <- support[which.max(density[,h])]
      
      results[[counter]] <- data.frame(ls = log(dens), 
                                       map = map,
                                       actual = y[Tseq[t] + h, j],
                                       id = j,
                                       h = h,
                                       t = Tseq[t],
                                       k = K[k],
                                       switch = switch)
      counter <- counter + 1
    }
  }
}
    

results <- bind_rows(results)
write.csv(results, paste0('elec_k',K[k], 's', switch, '.csv'), row.names = FALSE)

results %>%
  group_by(t) %>%
  summarise(pred = sum(map),
            actual = sum(actual)) %>%
  mutate(error = abs(actual - pred) / actual) %>%
  ggplot() + geom_line(aes(t, error))

results %>%
  group_by(id, t) %>%
  summarise(ls = mean(ls)) %>%
  ggplot() + geom_boxplot(aes(factor(t), ls)) + 
  theme(legend.position = 'none')

results <- rbind(read_csv('electricity/elec_k1s0.csv'),
                 read_csv('electricity/elec_k3s0.csv'),
                 read_csv('electricity/elec_k1s1.csv'),
                 read_csv('electricity/elec_k3s1.csv'))

results %>%
  group_by(k, switch, t, h) %>%
  summarise(pred = sum(map),
            actual = sum(actual)) %>%
  ungroup() %>%
  mutate(error = abs(actual - pred) / actual,
         t = t / 48 - 7,
         group = paste(k, switch)) %>%
  group_by(group, t) %>%
  summarise(mape = mean(error)) %>%
  ggplot() + geom_line(aes(t, mape, colour = group))

results %>%
  group_by(t, k, switch) %>%
  mutate(ls = mean(ls, na.rm = TRUE),
         group = paste(k, switch)) %>%
  ggplot() + geom_line(aes(t, ls, colour = group))


results %>%
  group_by(k, switch, t, h) %>%
  summarise(pred = sum(map), actual = sum(actual), ls = mean(ls, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(error = abs(actual - pred) / actual) %>%
  group_by(k, switch) %>%
  summarise(mape = mean(error),
            ls = mean(ls))
  

results %>%
  group_by(k, switch, t, h) %>%
  summarise(pred = sum(map),
            actual = sum(actual)) %>%
  ungroup() %>%
  gather(var, value, pred, actual) %>%
  ggplot() + geom_line(aes(t + h, value, colour = var)) + facet_wrap(k ~ switch)
