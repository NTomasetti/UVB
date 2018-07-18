library(tidyverse)
library(lubridate)
library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
source('RFuns.R')
source('electricity/elecAux.R')
sourceCpp('electricity/electricityVB.cpp')
N <- 200

y <- readRDS('electricity/elecY.RDS')[,1:N]
y <- log(y + 0.01)
x <- readRDS('electricity/elecX.RDS')[1:nrow(y),]

Tseq <- c(48 * 31, 48 * 92)
for(t in 93:(93 + 365)){
  Tseq <- c(Tseq, 48 * t)
}
TseqTrain <- c(48 * 31)
for(t1 in 32:92){
  TseqTrain <- c(TseqTrain, 48 * t1)
}

K <- c(1, 3)
order <- c(1, 2, 3, 48, 96, 144, 336)
dim <- 1 + ncol(x) + length(order)
samples <- 100
switch <- 1
k <- 2
var <- 1
batchSize <- 20

results <- list()
counter <- 1

pkFull <- array(0, dim = c(N, K[k], length(Tseq)))
psFull <- array(0, dim = c(N, K[k], length(Tseq)))

lambda <- NULL
## Add K[k] many sets of switching params
if(switch){
  lambda <- rep(c(0, 0, 0.1, 0, 0.1), 2*K[k])
}
## Repeat for Each Dynamic model: Log variance, mean, sarima parameters, temperature coefficient
## Generate different random numbers for the mean of each model
for(ki in 1:K[k]){
  lambda <- c(lambda, c(rnorm(dim, 0, 0.1), diag(0.1, dim))) 
}

priorPS <- matrix(0.1, K[k], N)
if(switch){
  priorMean <- array(0, dim = c(dim, K[k], 2))
  priorMean[2,,1] <- log(0.01)
  priorLinv <- array(0, dim = c(dim, dim, 2*K[k]))
  for(ki in 1:K[k]){
    priorLinv[1:2, 1:2, ki] <- solve(chol(diag(1, 2)))
    priorLinv[3:4, 1:2, ki] <- solve(chol(diag(1, 2)))
    priorLinv[,,K[k] + ki] <- solve(chol(diag(1, dim)))
  }
} else {
  priorMean <- array(0, dim = c(dim, K[k], 1))
  priorLinv <- array(0, dim = c(dim, dim, K[k]))
  for(ki in 1:K[k]){
    priorLinv[,, ki] <- solve(chol(diag(1, dim)))
  }
}

priorProbK <- matrix(1/K[k], nrow = K[k], ncol = N)

pkFull[,,1] <- t(priorProbK)
psFull[,,1] <- t(priorPS)


fitMat <- matrix(0, length(lambda), length(Tseq))
for(t in 3:(length(Tseq)-1)){
  print(paste0('t: ', t, ', switch : ', switch, ', k: ', k, ', time: ', Sys.time()))
  
  # Fit model, either via standard VB (t = 2 --> First fit) or updated VB
  # Switching and non switching need different inputs so are split with if(switch)
  if(t == 2){
    if(var){
      for(t1 in 2:(length(TseqTrain) - 1)){
        print(paste(t1, Sys.time()))
        if(t1 == 2){
          fit <- fitVB(data = y[1:TseqTrain[2],],
                       lambda = lambda,
                       model = elecModel,
                       dimTheta = (4*switch + dim) * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.05 * N,
                       priorMean = priorMean,
                       priorLinv = priorLinv,
                       probK = priorProbK, 
                       ps1 = priorPS,
                       order = order,
                       Tn = TseqTrain[2] - TseqTrain[1],
                       x = x[1:TseqTrain[2],],
                       uniformRho = TRUE,
                       var = FALSE,
                       switching = switch,
                       batch = batchSize)
        } else {
          updatePriors <- updatePrior(fit$lambda, FALSE, switch, K[k], dim)
          theta <- sampleTheta(updatePriors, samples, FALSE, switch, K[k], dim, 11:13)
          
          updateProbK <- matrix(0, K[k], N)
          updateProbS1 <- matrix(0, K[k], N)
          for(j in 1:N){
            for(i in 1:samples){
              if(switch){
                for(ki in 1:K[k]){
                  probs <- probSKHF(y = y[1:TseqTrain[t1-1], j], 
                                    x = x[1:TseqTrain[t1-1], ],
                                    thetaD = matrix(theta$tDynFull[,,ki,i], ncol = 1),
                                    thetaC = theta$tConsFull[,ki,i],
                                    order = order,
                                    pS1 = priorPS[ki, j],
                                    prior = priorProbK[ki, j],
                                    T = TseqTrain[t1-1] - TseqTrain[1],
                                    var = FALSE)
                  updateProbK[ki,j] <- updateProbK[ki,j] + probs[1] / samples
                  updateProbS1[ki, j] <- updateProbS1[j] + probs[2] / samples
                }
              } else {
                for(ki in 1:K[k]){
                  probs <- probK(y = y[1:TseqTrain[t1-1], j], 
                                 x = x[1:TseqTrain[t1-1], ],
                                 theta = matrix(theta$tDynFull[,,ki,i], ncol = 1),
                                 order = order,
                                 prior = priorProbK[ki, j],
                                 T = TseqTrain[t1-1] - TseqTrain[1],
                                 var = FALSE)
                  updateProbK[ki,j] <- updateProbK[ki,j] + probs / samples
                }
              }
            }
          }
          updateProbK <- apply(updateProbK, 2, function(x){
            y = x - max(x);
            exp(y) / sum(exp(y))
          })
      
          
          fit <- fitVB(data = y[1:TseqTrain[t1],],
                       lambda = c(fit$lambda),
                       model = elecModel,
                       dimTheta = (4*switch + dim) * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.05 * N,
                       priorMean = updatePriors$updateMean,
                       priorLinv = updatePriors$updateLinv,
                       probK = updateProbK, 
                       ps1 = updateProbS1,
                       order = order,
                       Tn = TseqTrain[t1] - TseqTrain[t1-1],
                       x = x[1:TseqTrain[t1],],
                       uniformRho = FALSE,
                       var = FALSE,
                       switching = switch,
                       batch = batchSize)
       
          
        }
      }
      saveRDS(fit, 'varSetup.RDS')
     
      lambdaVar <- NULL
      if(switch){
        lambdaVar <- fit$lambda[1: (10 * K[k])]
      }
      paramPerMod <- dim * (dim + 1) 
      for(ki in 1:K[k]){
        # Attach Means, repeat over 48 halfhours
        lambdaVar <- c(lambdaVar, rep(fit$lambda[10 * K[k] * switch + (ki - 1) * paramPerMod + 1:dim], rep(48, dim)))
        U <- matrix(fit$lambda[10 * K[k] * switch + (ki - 1) * paramPerMod + dim + 1:(dim^2)], dim)
        Sigma <- t(U) %*% U
        logsd <- log(sqrt(diag(Sigma)))
        # Attach log(sd)
        lambdaVar <- c(lambdaVar, rep(logsd, rep(48, dim)))
      }
      priorMean <- array(0, dim = c(dim, 48, switch + K[k]))
      priorLinv <- array(0, dim = c(48, 48, (dim + switch) * K[k]))
      if(switch){
        priorMean[2, , 1] <- log(0.01)
        for(ki in 1:K[k]){
          priorLinv[1:2, 1:2, ki] <- priorLinv[3:4, 1:2, ki] <- solve(chol(diag(10, 2)))
        }
      }
      initialVar <- 1
      updateVar <- 0.25
      varSeq <- cumsum(c(initialVar, rep(updateVar, 47)))
      Sigma <- matrix(0, 48, 48)
      for(i in 1:48){
        for(j in 1:48){
          Sigma[i, j] <- min(varSeq[i], varSeq[j])
        }
      }
      SigmaLinv <- solve(t(chol(Sigma)))
      for(ki in 1:K[k]){
        for(i in 1:dim){
          priorLinv[,,K[k] * switch + (ki - 1) * dim + i] <- SigmaLinv
        }
      }
      fit <- fitVB(data = y[1:Tseq[2],],
                   lambda = lambdaVar,
                   model = elecModel,
                   dimTheta = (4*switch + dim * 48) * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.05 * N,
                   priorMean = priorMean,
                   priorLinv = priorLinv,
                   probK = priorProbK, 
                   ps1 = priorPS,
                   order = order,
                   Tn = Tseq[2] - Tseq[1],
                   x = x[1:Tseq[2],],
                   uniformRho = TRUE,
                   var = TRUE,
                   switching = switch,
                   batch = batchSize)
      saveRDS(fit, 'varInitialFit.RDS')
      fitMat <- matrix(0, length(lambdaVar), length(Tseq))
      
    } else {
      fit <- fitVB(data = y[1:Tseq[2],],
                   lambda = lambda,
                   model = elecModel,
                   dimTheta = (4*switch + dim) * K[k],
                   S = 50,
                   maxIter = 2000,
                   threshold = 0.05 * N,
                   priorMean = priorMean,
                   priorLinv = priorLinv,
                   probK = priorProbK, 
                   ps1 = priorPS,
                   order = order,
                   Tn = Tseq[2] - Tseq[1],
                   x = x[1:Tseq[2],],
                   uniformRho = TRUE,
                   var = FALSE,
                   switching = switch,
                   batch = batchSize)
    }

  } else {
    fit <- fitVB(data = y[1:Tseq[t],],
                 lambda = c(fit$lambda),
                 model = elecModel,
                 dimTheta = (4*switch + ifelse(var, 48, 1) * dim) * K[k],
                 S = 50,
                 maxIter = 2000,
                 threshold = 0.05 * N,
                 priorMean = updatePriors$updateMean,
                 priorLinv = updatePriors$updateLinv,
                 probK = updateProbK, 
                 ps1 = updateProbS1,
                 order = order,
                 Tn = Tseq[t] - Tseq[t-1],
                 x = x[1:Tseq[t],],
                 uniformRho = FALSE,
                 var = var,
                 switching = switch,
                 batch = batchSize)
  }
  fitMat[,t] <- fit$lambda
  
  
  

  updatePriors <- updatePrior(fit$lambda, var, switch, K[k], dim)
  theta <- sampleTheta(updatePriors, samples, var, switch, K[k], dim, 11:13)
  
  # For each household:
  # 1) Calculate p(k = j) / p(S_T = 1)
  # 2) Create forecast densities per k and combine with above probabilities
  # 3) Calculate updated priors for p(k) and p(s) to be pushed into the next update
  # 4) Evaluate forecast densities
  # Steps 1) and 2) are handled in c++
  support <- seq(log(0.009), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
 
  updateProbK <- matrix(0, K[k], N)
  updateProbS1 <- matrix(0, K[k], N)
  for(j in 1:N){
    density <- matrix(0, 5000, 48)
    for(i in 1:samples){
      if(switch){
        newPK <- rep(-Inf, K[k])
        newPS <- rep(0, K[k])
        for(ki in 1:K[k]){
          if(pkFull[j, ki, t-1] > 0){
            probs <- probSKHF(y = y[1:Tseq[t], j],
                              x = x[1:Tseq[t],],
                              thetaD = theta$tDynFull[,,ki, i],
                              thetaC = theta$tConsFull[,ki, i],
                              order = order,
                              pS1 = psFull[j, ki, t-1],
                              prior = pkFull[j, ki, t-1],
                              T = Tseq[t] - Tseq[t-1],
                              var = var)
            newPK[ki] <- probs[1]
            newPS[ki] <- probs[2]
          }
        }
        newPK <- newPK - max(newPK)
        newPK <- exp(newPK) / sum(exp(newPK))
        
        fcDensity <- forecastHF(y = y[1:Tseq[t+1], j],
                         x = x[1:Tseq[t+1],],
                         thetaC = theta$tConsFull[,,i],
                         thetaD = array(theta$tDynFull[,,,i], dim = c(dim(theta$tDynFull)[1:3])),
                         fcVar = array(theta$fcVar[,,,i], dim = c(dim(theta$fcVar)[1:3])),
                         order = order,
                         pS1prior = newPS,
                         pkNorm = newPK,
                         support = support,
                         var = var)
        density <- density + fcDensity / samples
        updateProbK[,j] <- updateProbK[,j] + newPK / samples
        updateProbS1[j] <- updateProbS1[j] + newPS / samples
      } else {
        newDataLL <- rep(0, K[k])
        for(ki in 1:K[k]){
          if(pkFull[j, ki, t-1] > 0){
            newDataLL[ki] <- sum(arLikelihood(y[1:Tseq[t], j], x[1:Tseq[t],], matrix(theta$tDynFull[,,ki,i], nrow = dim),
                                          order, Tseq[t] - Tseq[t-1], TRUE, var))
          }
        }
        newDataLL <- newDataLL + log(pkFull[j,,t-1])
        newDataLL[is.na(newDataLL)] <- -Inf
        newDataLL <- newDataLL - max(newDataLL)
        newPK <-  exp(newDataLL) / sum(exp(newDataLL))
       
        updateProbK[,j] <- updateProbK[,j] + newPK / samples
      
        fcDensity <- forecastStandard(y = y[1:Tseq[t+1], j],
                               x = x[1:Tseq[t+1],],
                               theta = array(theta$tDynFull[,,,i], dim = c(dim(theta$tDynFull)[1:3])),
                               order = order,
                               pkNorm = newPK,
                               fcVar = array(theta$fcVar[,,,i], dim = c(dim(theta$fcVar)[1:3])),
                               support = support,
                               var = var)
        density <- density + fcDensity / samples
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
                                       var = var,
                                       switch = switch)
      counter <- counter + 1
    }
  }
  if(any(round(colSums(updateProbK), 5) != 1)){
    break
    print('check pk')
  }

  pkFull[,,t] <- t(updateProbK)
  psFull[,,t] <- t(updateProbS1)
  
  #print(qplot(rowSums(pkFull[,,t])))
  saveRDS(list(fitMat, pkFull, updateProbS1, results, t), 'elecFitSw.RDS')
}


results <- readRDS('elecFit.RDS')

results <- bind_rows(results)
write.csv(results, paste0('elec_k',K[k], 's', switch, '.csv'), row.names = FALSE)


varInit <- readRDS('fit_VAR_K3_3.RDS')
varFit <- updatePrior(varInit$lambda, TRUE, FALSE, 3, dim)
varMean <- varFit$updateMean
varMean[1,,] <- exp(varMean[1,,] + 0.5 * varFit$updateSd[1,,]^2)

varMean <- data.frame(mean = c(varMean), 
                      group = rep(1:K[k], rep(48 * 17, 3)),
                      halfhour = rep(1:48, rep(17, 48)),
                      var = c('sigma^{2}', 'intercept', 'temp', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'publicHol',
                                  'phi[1]', 'phi[2]', 'phi[3]', 'phi[48]', 'phi[96]', 'phi[144]', 'phi[336]'))

varMean %>%
  spread(var, mean) %>%
  mutate(day1 = intercept + day1,
         day2 = intercept + day2,
         day3 = intercept + day3,
         day4 = intercept + day4,
         day5 = intercept + day5,
         day6 = intercept + day6) %>%
  rename(Sunday = intercept, Monday = day1, Tuesday = day2, Wedesday = day3, Thursday = day4, Friday = day5, Saturday = day6) %>%
  gather(var, mean, -group, -halfhour) %>%
  mutate(var = factor(var, levels = c('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'publicHol', 'temp',
                                      'sigma^{2}', 'phi[1]', 'phi[2]', 'phi[3]', 'phi[48]', 'phi[96]', 'phi[144]', 'phi[336]'))) -> varFit


ggplot(varFit) + geom_line(aes(halfhour, mean, colour = factor(group))) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed, ncol = 5) + 
  theme_bw()


results %>%
  filter(h %in% c(1, 2, 6, 12, 24, 48)) %>%
  mutate(mapExp = exp(map) - 0.01,
         mapActual = exp(actual) - 0.01,
         week = floor(t / 336)) %>%
  group_by(t, h) %>%
  summarise(map = sum(mapExp),
            actual = sum(mapActual),
            week = head(week, 1)) %>%
  mutate(ape = abs(map - actual) / actual) %>%
  ungroup() %>%
  group_by(h, week) %>%
  summarise(mape = mean(ape),
            t = min(t)) %>%
  ggplot() + geom_line(aes(t, mape)) + facet_wrap(~h)

results %>%
  filter(h %in% c(1, 2, 6, 12, 24, 48)) %>%
  group_by(t, h) %>%
  summarise(ls = mean(ls, na.rm = TRUE)) %>%
  ggplot() + geom_line(aes(t, ls)) + facet_wrap(~h)
