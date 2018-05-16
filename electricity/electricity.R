library(tidyverse)
library(lubridate)
library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
source('RFuns.R')
sourceCpp('electricity/electricityVB.cpp')
N <- 20

full <- readr::read_csv('electricity/elecdataBal.csv')
set.seed(3)



full %>%
  group_by(ID) %>%
  filter(date(time) >= '2013-06-01' & date(time) <= '2013-08-01' & tariff == 'Std') %>%
  filter(all(!is.na(energy))) %>%
  summarise(n = n()) %>%
  filter(n == 2976) %>%
  .$ID %>%
  sample(N) -> IDvec

full %>%
  filter(ID %in% IDvec & date(time) >= '2013-06-01' & date(time) <= '2013-08-01') %>%
  select(time, ID, energy) %>%
  spread(ID, energy) -> data

data %>%
  select(-time) %>%
  as.matrix() -> y

full %>%
  filter(ID == IDvec[1] & 
           date(time) >= '2013-06-01' &
           date(time) <= '2013-08-01') %>%
  .$temp -> x

saveRDS(y, 'electricity/elecY.RDS')
saveRDS(x, 'electricity/elecX.RDS')

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

results <- list()
counter <- 1

for(k in 1:2){
  for(switch in 0:1){
      
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
    
    for(t in 2:(length(Tseq)-1)){
      print(paste0('t: ', t, ', switch : ', switch, ', k: ', k, ', time: ', Sys.time()))

      if(t == 2){
        if(switch){
          fit <- fitVB(data = y[1:Tseq[2],],
                       lambda = lambda,
                       model = electricityMod,
                       dimTheta = 4 + dim * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.1,
                       priorMean = priorMean,
                       priorLinv = priorLinv,
                       probK = priorProbK, 
                       pS1 = priorPS,
                       order = order,
                       Tn = Tseq[2] - Tseq[1],
                       x = x[1:Tseq[2]],
                       uniformRho = TRUE)
        } else {
          fit <- fitVB(data = y[1:Tseq[2],],
                       lambda = lambda,
                       model = electricityMod,
                       dimTheta = dim * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.1,
                       priorMean = priorMean,
                       priorLinv = priorLinv,
                       probK = priorProbK, 
                       order = order,
                       Tn = Tseq[2] - Tseq[1],
                       x = x[1:Tseq[2]])
        }
      } else {
        if(switch){
          fit <- fitVB(data = y[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                       lambda = c(fit$lambda),
                       model = electricityMod,
                       dimTheta = 4 + dim * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.1,
                       priorMean = updateMean,
                       priorLinv = updateLinv,
                       probK = updateProbK, 
                       pS1 = updateProbS1,
                       order = order,
                       Tn = Tseq[t] - Tseq[t-1],
                       x = x[(Tseq[t-1] - max(order) + 1):Tseq[t]],
                       uniformRho = FALSE)
        } else {
          fit <- fitVB(data = y[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                       lambda = c(fit$lambda),
                       model = electricityMod,
                       dimTheta = dim * K[k],
                       S = 50,
                       maxIter = 2000,
                       threshold = 0.1,5,
                       priorMean = updateMean,
                       priorLinv = updateLinv,
                       probK = updateProbK, 
                       order = order,
                       Tn = Tseq[t] - Tseq[t-1],
                       x = x[(Tseq[t-1] - max(order) + 1):Tseq[t]])
        }
      }
     
      density <- array(0, dim = c(5000, 48, N))
      support <- seq(min(y[Tseq[t-1]:Tseq[t],]) - sd(y[Tseq[t-1]:Tseq[t],]), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
      pk <- ps <- array(0, dim = c(K[k], N, samples))
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
      
      for(i in 1:samples){
        
        if(switch){
          thetaC <- c(updateMean[1:2, 1] + updateL[1:2, 1:2, 1] %*% rnorm(2))
          rho <- 1 / (1 + exp(- c(updateMean[1:2, 2] + updateL[1:2, 1:2, 2] %*% rnorm(2))))
        }
      
        theta <- matrix(0, dim, K[k])
        fcVar <- matrix(0, 48, K[k])
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
          fcVar[, ki] <- ltsa::PredictionVariance(autocov, 48)
        }
      
        for(j in 1:N){
          
          if(K[k] == 1){
            pk[,,] <- 1
            pkj <- 1
            if(switch){
              ps[1, j, i] <- probSKHF(y[1:Tseq[t], j], x[1:Tseq[t]], theta[,1], thetaC, order, priorPS[j], rho, priorProbK[1, j], Tseq[t] - Tseq[1])[2]
            }
          } else {
            for(ki in 1:K[k]){
              if(switch){
                prob <- probSKHF(y[1:Tseq[t], j], x[1:Tseq[t]], theta[,ki], thetaC, order, priorPS[j], rho, priorProbK[ki, j], Tseq[t] - Tseq[1])
                pk[ki, j, i] <- prob[1]
                ps[ki, j, i] <- prob[2]
              } else {
                pk[ki, j, i] <- probK(y[1:Tseq[t], j], x[1:Tseq[t]], theta[,ki], order ,priorProbK[ki, j], Tseq[t] - Tseq[1])
              }
            }
            pk[,j,i] <- pk[,j,i] - max(pk[,j,i])
            pkj <- exp(pk[, j, i]) / sum(exp(pk[, j, i]))
          }
          
         
          if(switch){
            ki <- sample(1:K[k], 1, prob = pkj)
            density[,,j] <- density[,,j] + forecastHF(y[(Tseq[t] - max(order) + 1):Tseq[t+1],j], x[(Tseq[t] - max(order) + 1):Tseq[t+1]],
                                                        thetaC, theta[,ki], fcVar[, ki], order, ps[ki, j, i], rho, support, 48) / samples
          } else {
            
            lagYPartial <- y[Tseq[t] - (0:2), j]
            for(h in 1:48){
              lagYFull <- c(lagYPartial, y[Tseq[t] + h - order[4:length(order)], j])
              lagX <- x[Tseq[t] + h - order]
              
              for(ki in 1:K[k]){
                mean <- theta[2, ki] + theta[4:nrow(theta), ki] %*% (lagYFull - theta[2, ki] - theta[3, ki] * lagX)
                density[,h,j] <- density[,h,j] + pkj[ki] * dnorm(support, mean, sqrt(fcVar[h, ki])) / samples
              }
              lagYPartial <- c(mean, lagYPartial[2:3])
            }
          }
         
        }
      }
      updateProbK <- apply(pk, 2:3, function(x) exp(x) / sum(exp(x)))
    
      if(K[k] == 1){
        updateProbK <- matrix(1, nrow = 1, ncol = N)
        updateProbS1 <- apply(ps, 1:2, mean)
        
      } else {
        updatePS <- calcProb(updateProbK, ps)
        updateProbS1 <- colMeans(updatePS)
        updateProbK <- apply(updateProbK, 1:2, mean)
      }
      
      for(j in 1:N){
        for(h in 1:48){
          lower <- max(which(support < y[Tseq[t] + h, j]))
          upper <- min(which(support > y[Tseq[t] + h, j]))
          dens <- linearInterpolate(support[lower], support[upper], density[lower, h, j], density[upper, h, j], y[Tseq[t] + h, j])
        
          map <- support[which.max(density[,h, j])]
        
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
                 read_csv('electricity/elec_k2s0.csv'),
                 read_csv('electricity/elec_k5s0.csv'),
                 read_csv('electricity/elec_k1s1.csv'),
                 read_csv('electricity/elec_k2s1.csv'),
                 read_csv('electricity/elec_k5s1.csv'))

results %>%
  filter(switch == 1 | (switch == 0 & k == 1)) %>%
  group_by(k, switch, t, h) %>%
  summarise(pred = sum(map),
            actual = sum(actual)) %>%
  mutate(error = abs(actual - pred) / actual) %>%
  ungroup() %>%
  group_by(k, switch, h) %>%
  summarise(mape = mean(error)) %>%
  ggplot() + geom_line(aes(h, mape)) + facet_wrap(k ~ switch)

results %>%
  filter(switch == 1 | (switch == 0 & k == 1)) %>%
  group_by(h, k, switch) %>%
  mutate(ls = mean(ls, na.rm = TRUE),
         group = paste(k, switch)) %>%
  ggplot() + geom_line(aes(h, ls, colour = group))
  
