library(tidyverse)
library(lubridate)
library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
source('RFuns.R')
source('electricity/elecAux.R')
sourceCpp('electricity/electricityVB.cpp')
N <- 25

y <- readRDS('electricity/ireland/Y.RDS')[,1:N]
y <- log(y + 0.01)
x <- readRDS('electricity/ireland/X.RDS')[1:nrow(y),]

Tseq <- c(48 * 31, 48 * 92)
for(t in 93:(93 + 365)){
  Tseq <- c(Tseq, 48 * t)
}
TseqTrain <- c(48 * 31)
for(t1 in 32:92){
  TseqTrain <- c(TseqTrain, 48 * t1)
}

K <- 2
mix <- 2
order <- c(1, 2, 3, 48, 96, 144, 336)
dim <- 1 + ncol(x) + length(order)
samples <- 100
switch <- 0
var <- 0
batchSize <- 0

results <- list()

pkFull <- array(0, dim = c(N, K, length(Tseq)))
psFull <- array(0, dim = c(N, K, length(Tseq)))

lambda <- NULL
for(m in 1:mix){
  lambdaTemp <- NULL
  ## Add K many sets of switching params
  if(switch){
    lambdaTemp <- rep(c(0, 0, 0, 0.1, 0, 0.1, 0, 0, 0.1), 2*K)
  }
  ## Repeat for Each Dynamic model: Log variance, mean, sarima parameters, temperature coefficient
  ## Generate different random numbers for the mean of each model
  for(ki in 1:K){
    lambdaTemp <- c(lambdaTemp, c(rnorm(dim, 0, 0.05), diag(0.1, dim))) 
  }
  lambda <- cbind(lambda, lambdaTemp)
}
weightsZ <- rep(0, mix)

priorPS <- matrix(0.1, K, N)
if(switch){
  priorMean <- array(0, dim = c(dim, K, 2))
  priorMean[2,,1] <- log(0.01)
  priorLinv <- array(0, dim = c(dim, dim, 2*K))
  for(ki in 1:K){
    priorLinv[1:2, 1:2, ki] <- solve(chol(diag(10, 2)))
    priorLinv[3:4, 1:2, ki] <- solve(chol(diag(10, 2)))
    priorLinv[,,K + ki] <- solve(chol(diag(10, dim)))
  }
} else {
  priorMean <- array(0, dim = c(dim, K, 1))
  priorLinv <- array(0, dim = c(dim, dim, K))
  for(ki in 1:K){
    priorLinv[,, ki] <- solve(chol(diag(1, dim)))
  }
}

priorProbK <- matrix(1/K, nrow = K, ncol = N)

pkFull[,,1] <- t(priorProbK)
psFull[,,1] <- t(priorPS)


fitMat <- matrix(0, length(lambda), length(Tseq))
for(t in 2:(length(Tseq)-1)){
  print(paste0('t: ', t, ', switch : ', switch, ', k: ', k, ', time: ', Sys.time()))
  
  # Fit model, either via standard VB (t = 2 --> First fit) or updated VB
  # Switching and non switching need different inputs so are split with if(switch)
  if(t == 2){
    if(var){
      for(t1 in 2:(length(TseqTrain) - 1)){
        print(paste(t1, Sys.time()))
        if(t1 == 2){
          fit <- fitVBMix(data = y[1:TseqTrain[2],],
                          lambda = lambda,
                          model = elecModel,
                          dimTheta = (4*switch + dim) * K,
                          S = 15,
                          mix = mix,
                          weightsZ = weightsZ,
                          maxIter = 2000,
                          threshold = 0.05 * N,
                          priorMean = priorMean,
                          priorLinv = priorLinv,
                          priorWeights = 1,
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
          updatePriors <- list()
          for(m in 1:mix){
            updatePriors[[m]] <- updatePrior(fit$lambda[,m], FALSE, switch, K, dim)
          }
          theta <- sampleTheta(updatePriors, samples, FALSE, switch, K, dim, 12:14, exp(fit$weights) / sum(exp(fit$weights)))
          
          updateProbK <- matrix(0, K, N)
          updateProbS1 <- matrix(0, K, N)
          for(j in 1:N){
            for(i in 1:samples){
              if(switch){
                for(ki in 1:K){
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
                for(ki in 1:K){
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
          
          updateMean <- array(0, dim = c(dim, K, mix))
          updateLinv <- array(0, dim = c(dim, dim, K * mix))
          for(m in 1:mix){
            updateMean[,,m] <- updatePriors[[m]]$updateMean
            for(k in 1:K){
              updateLinv[,,(m-1) *K + k] <- updatePriors[[m]]$updateLinv[,,k]
            }
          }
          
          
          fit <- fitVBMix(data = y[1:TseqTrain[t1],],
                          lambda = fit$lambda,
                          model = elecModel,
                          dimTheta = (4*switch + dim) * K,
                          S = 15,
                          maxIter = 2000,
                          threshold = 0.05 * N,
                          mix = mix,
                          weightsZ = fit$weights,
                          priorMean = updateMean,
                          priorLinv = updateLinv,
                          priorWeights = exp(fit$weights) / sum(exp(fit$weights)),
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
        lambdaVar <- fit$lambda[1: (10 * K), ]
      }
      paramPerMod <- dim * (dim + 1) 
      for(m in 1:mix){
        lvTemp <- NULL
        for(ki in 1:K){
          # Attach Means, repeat over 48 halfhours
          lv <- rep(fit$lambda[10 * K * switch + (ki - 1) * paramPerMod + 1:dim, m], rep(48, dim))
          U <- matrix(fit$lambda[10 * K * switch + (ki - 1) * paramPerMod + dim + 1:(dim^2), m], dim)
          Sigma <- t(U) %*% U
          logsd <- log(sqrt(diag(Sigma)))
          # Attach log(sd)
          lv <- c(lv, rep(logsd, rep(48, dim)))
          lvTemp <- cbind(lvTemp, lb)
        }
        lambdaVar <- rbind(lambdaVar, lbTemp)
      }
     
      priorMean <- array(0, dim = c(dim, 48, switch + K))
      priorLinv <- array(0, dim = c(48, 48, (dim + switch) * K))
      if(switch){
        priorMean[2, , 1] <- log(0.01)
        for(ki in 1:K){
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
      for(ki in 1:K){
        for(i in 1:dim){
          priorLinv[,,K * switch + (ki - 1) * dim + i] <- SigmaLinv
        }
      }
      fit <- fitVBMix(data = y[1:Tseq[2],],
                      lambda = lambdaVar,
                      model = elecModel,
                      dimTheta = (4*switch + dim * 48) * K,
                      S = 25,
                      maxIter = 2000,
                      mix = mix,
                      weightsZ = fit$weights,
                      threshold = 0.05 * N,
                      priorMean = priorMean,
                      priorLinv = priorLinv,
                      priorWeights = 1,
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
                   dimTheta = (5*switch + dim) * K,
                   S = 10,
                   mix = mix,
                   weightsZ = rep(1, mix),
                   maxIter = 2000,
                   threshold = 0.05 * N,
                   priorMean = priorMean,
                   priorLinv = priorLinv,
                   priorWeights = 1,
                   probK = priorProbK, 
                   ps1 = priorPS,
                   order = order,
                   Tn = Tseq[2] - Tseq[1],
                   x = x[1:Tseq[2],],
                   uniformRho = TRUE,
                   var = FALSE,
                   switching = switch)
      # batch = batchSize)
    }
    
  } else {
    
    updateMean <- array(0, dim = c(dim, K, mix))
    updateLinv <- array(0, dim = c(dim, dim, K * mix))
    for(m in 1:mix){
      updateMean[,,m] <- updatePriors[[m]]$updateMean
      for(k in 1:K){
        updateLinv[,,(m-1) *K + k] <- updatePriors[[m]]$updateLinv[,,k]
      }
    }
    
    fit <- fitVB(data = y[1:Tseq[t],],
                 lambda = c(fit$lambda),
                 model = elecModel,
                 dimTheta = (4*switch + ifelse(var, 48, 1) * dim) * K,
                 S = 50,
                 maxIter = 2000,
                 threshold = 0.05 * N,
                 priorMean = updateMean,
                 priorLinv = updateLinv,
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
  
  for(m in 1:mix){
    updatePriors[[m]] <- updatePrior(fit$lambda[,m], FALSE, switch, K, dim)
  }
  theta <- sampleTheta(updatePriors, samples, var, switch, K, dim, 12:14, exp(fit$weights) / sum(exp(fit$weights)))
  
  # For each household:
  # 1) Calculate p(k = j) / p(S_T = 1)
  # 2) Create forecast densities per k and combine with above probabilities
  # 3) Calculate updated priors for p(k) and p(s) to be pushed into the next update
  # 4) Evaluate forecast densities
  # Steps 1) and 2) are handled in c++
  support <- seq(log(0.009), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
  
  updateProbK <- matrix(0, K, N)
  updateProbS1 <- matrix(0, K, N)
  df <- data.frame()
  for(j in 1:N){
    density <- matrix(0, 5000, 48)
    for(i in 1:samples){
      if(switch){
        newPK <- rep(-Inf, K)
        newPS <- rep(0, K)
        for(ki in 1:K){
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
        newDataLL <- rep(0, K)
        for(ki in 1:K){
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
      
      df <- rbind(df,
                  data.frame(ls = log(dens), 
                             map = map,
                             actual = y[Tseq[t] + h, j],
                             id = j,
                             h = h,
                             t = Tseq[t],
                             k = K,
                             var = var,
                             mix = mix,
                             switch = switch))
      results[[t-1]] <- df
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

support <- list(seq(1.2, 1.6, length.out = 1000),
                seq(-0.5, 0.5, length.out = 1000),
                seq(-0.1, 0.15, length.out = 1000),
                seq(-0.1, 0.05, length.out = 1000),
                seq(-2, 2, length.out = 1000),
                seq(-0.5, 2, length.out = 1000),
                seq(-0.5, 2, length.out = 1000),
                seq(-1, 0.5, length.out = 1000),
                seq(-1, 0.5, length.out = 1000),
                seq(-2, 2, length.out = 1000),
                seq(-3, 3, length.out = 1000),
                seq(0, 0.3, length.out = 1000),
                seq(-0.15, 0.25, length.out = 1000),
                seq(-0.1, 0.35, length.out = 1000),
                seq(-0.2, 0.35, length.out = 1000),
                seq(0, 0.35, length.out = 1000),
                seq(0, 0.35, length.out = 1000),
                seq(-0.05, 0.4, length.out = 1000))


density <- data.frame()
ncomp <- dim * (dim + 1)
for(m in 1:mix){
  for(k in 1:K){
    fitList <- list(mean = fit$lambda[(k-1) * ncomp + 1:dim, m],
                    U = fit$lambda[(k-1) * ncomp + (dim+1):ncomp, m])
    
    dens <- vbDensity(fit = fitList,
                      transform = c('exp', rep('identity', dim - 1)),
                      names = c('sigma^{2}', 'intercept', 'temp', 'humidity', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'publicHol',
                                'phi[1]', 'phi[2]', 'phi[3]', 'phi[48]', 'phi[96]', 'phi[144]', 'phi[336]'),
                      support = support)
 
    dens$w <- exp(fit$weights[m]) / sum(exp(fit$weights))
    dens$mix <- m
    dens$K <- k
    density <- rbind(density, dens)
  }
}

density %>%
  add_column(i = rep(1:1000, dim * K * mix)) %>%
  group_by(var, i, K) %>%
  summarise(density = sum(density * w),
            support = min(support)) %>%
  ggplot() + geom_line(aes(support, density, colour = factor(K))) + facet_wrap(~var, scales = 'free')






results <- readRDS('elecFit.RDS')

results <- bind_rows(results)
write.csv(results, paste0('elec_k',K, 's', switch, '.csv'), row.names = FALSE)


varInit <- readRDS('fit_VAR_K3_3.RDS')
varFit <- updatePrior(varInit$lambda, TRUE, FALSE, 3, dim)
varMean <- varFit$updateMean
varMean[1,,] <- exp(varMean[1,,] + 0.5 * varFit$updateSd[1,,]^2)

varMean <- data.frame(mean = c(varMean), 
                      group = rep(1:K, rep(48 * 17, 3)),
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
