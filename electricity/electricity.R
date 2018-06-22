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
y <- log(y + 0.01)
x <- readRDS('electricity/elecX.RDS')[1:nrow(y),]

Tseq <- c(48 * 7 * 2, 48 * 7 * 5)
for(t in 1:(62-35)){
  Tseq <- c(Tseq, 48 * (35 + t))
}

K <- c(1, 2)
order <- c(1, 2, 3, 48, 96, 144, 336)
dim <- 1 + ncol(x) + length(order)
samples <- 250
switch <- 1
k <- 2
batchSize <- 5

results <- list()
counter <- 1

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

priorPS <- rep(0.1, N)
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
  priorMean <- matrix(0, dim, K[k])
  priorLinv <- array(0, dim = c(dim, dim, K[k]))
  for(ki in 1:K[k]){
    priorLinv[,, ki] <- solve(chol(diag(1, dim)))
  }
}

priorProbK <- matrix(1/K[k], nrow = K[k], ncol = N)

fitMat <- matrix(0, length(lambda), length(Tseq))
for(t in 2:(length(Tseq)-1)){
  print(paste0('t: ', t, ', switch : ', switch, ', k: ', k, ', time: ', Sys.time()))
  
  # Fit model, either via standard VB (t = 2 --> First fit) or updated VB
  # Switching and non switching need different inputs so are split with if(switch)
  if(t == 2){
    fit <- fitVB(data = y[1:Tseq[2],],
                 lambda = lambda,
                 model = elecModel,
                 dimTheta = (4*switch + dim) * K[k],
                 S = 50,
                 maxIter = 2000,
                 threshold = 0.25,
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
  } else {
    fit <- fitVB(data = y[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                 lambda = c(fit$lambda),
                 model = elecModel,
                 dimTheta = (4*switch + dim) * K[k],
                 S = 50,
                 maxIter = 2000,
                 threshold = 0.2,
                 priorMean = updateMean,
                 priorLinv = updateLinv,
                 probK = updateProbK, 
                 ps1 = updateProbS1,
                 order = order,
                 Tn = Tseq[t] - Tseq[t-1],
                 x = x[(Tseq[t-1] - max(order) + 1):Tseq[t],],
                 uniformRho = FALSE,
                 var = FALSE,
                 switching = switch,
                 batch = batchSize)
  }
  fitMat[,t] <- fit$lambda

  # Calculate updated priors for theta parameters
  updateMean <- priorMean
  updateLinv <- updateL <- priorLinv
  if(switch){
    for(ki in 1:K[k]){
      updateMean[1:2, ki, 1] <- fit$lambda[(ki-1)*5 + 1:2]
      updateMean[3:4, ki, 1] <- fit$lambda[5*K[k] + (ki-1)*5 + 1:2]
      updateL[1:2, 1:2, ki] <- matrix(c(fit$lambda[(ki-1)*5 + 3:4], 0, fit$lambda[(ki-1)*5 + 5]), 2)
      updateL[3:4, 1:2, ki] <- matrix(c(fit$lambda[K[k] * 5 + (ki-1)*5 + 3:4], 0, fit$lambda[K[k] * 5 + (ki-1)*5 + 5]), 2) 
      updateLinv[1:2, 1:2, ki] <- solve(updateL[1:2, 1:2, ki])
      updateLinv[3:4, 1:2, ki] <- solve(updateL[3:4, 1:2, ki])
      updateMean[,ki, 2] <- fit$lambda[10*K[k] + (ki-1) * dim * (dim + 1) + 1:dim]
      updateL[,,K[k] + ki] <- matrix( fit$lambda[10*K[k] + (ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
      updateLinv[,,K[k] + ki] <- solve(updateL[,,K[k] + ki])
    }
  } else {
    for(ki in 1:K[k]){
      updateMean[,ki] <- fit$lambda[(ki-1) * dim * (dim + 1) + 1:dim]
      updateL[,, ki] <- matrix( fit$lambda[(ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
      updateLinv[,, ki] <- solve(updateL[,,ki])
    }
  }

  # Sample theta values
  tDynFull <- array(0, dim = c(dim, 1, K[k], samples))
  tConsFull <- array(0, dim = c(4, K[k], samples))
  fcVar <- array(0, dim = c(48, 1, K[k], samples))
  for(i in 1:samples){
    
    thetaCons <- matrix(0, 4, K[k])
    if(switch){
      for(ki in 1:K[k]){
        thetaCons[1:2,ki] <- updateMean[1:2, ki, 1] + updateL[1:2, 1:2, ki] %*% rnorm(2)
        thetaCons[3:4, ki] <- 1 / (1 + exp(-(updateMean[3:4, ki, 1] + updateL[3:4, 1:2, ki] %*% rnorm(2))))
      }
    }
    tConsFull[,, i] <- thetaCons
    
    thetaDyn <- matrix(0, dim, K[k])
    for(ki in 1:K[k]){
      if(switch){
        thetaDyn[, ki] <- updateMean[,ki, 2] + updateL[,,K[k] + ki] %*% rnorm(dim)
      } else {
        thetaDyn[, ki] <- updateMean[,ki] + updateL[,,ki] %*% rnorm(dim)
      }
      
      
      root <- polyroot(c(1, -thetaDyn[11:13, ki]))
      if(any(Mod(root) < 1)){
        if(i > 1){
          thetaDyn[,ki] <- tDynFull[,1,ki,i-1]
        } else {
          stat <- FALSE
          try <- 0
          while(!stat){
            if(switch){
              thetaDyn[, ki] <- updateMean[,ki, 2] + updateL[,,K[k] + ki] %*% rnorm(dim)
            } else {
              thetaDyn[, ki] <- updateMean[,ki] + updateL[,,ki] %*% rnorm(dim)
            }
            
            root <- polyroot(c(1, -thetaDyn[11:13, ki]))
            try <- try + 1
            if(all(Mod(root) > 1)){
              stat <- TRUE
            } else if(try > 10){
              thetaDyn[11:13, ki] <- 0
              stat <- TRUE
            }
          }
        }
      }
      
      autocov <- ltsa::tacvfARMA(thetaDyn[11:13, ki], sigma2 = exp(thetaDyn[1, ki]), maxLag = 48)
      fcVar[,1, ki, i] <- ltsa::PredictionVariance(autocov, 48)
      tDynFull[,1,ki,i] <- thetaDyn[,ki]
    }
  }
  
  # For each household:
  # 1) Calculate p(k = j) / p(S_T = 1)
  # 2) Create forecast densities per k and combine with above probabilities
  # 3) Calculate updated priors for p(k) and p(s) to be pushed into the next update
  # 4) Evaluate forecast densities
  # Steps 1) and 2) are handled in c++
  support <- seq(log(0.009), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
  updateProbK <- matrix(0, K[k], N)
  updateProbS1 <- rep(0, N)
  for(j in 1:N){
    density <- matrix(0, 5000, 48)
    for(i in 1:samples){
      if(switch){
        fc <- forecastHF(y = y[1:Tseq[t+1], j],
                         x = x[1:Tseq[t+1],],
                         thetaC = tConsFull[,,i],
                         thetaD = extractArray4th(tDynFull, i),
                         fcVar = extractArray4th(fcVar, i),
                         order = order,
                         pS1prior = priorPS[j],
                         priorK = priorProbK[,j],
                         support = support,
                         Tn = Tseq[t] - Tseq[1],
                         var = FALSE)
        density <- density + fc$density / samples
        updateProbK[,j] <- updateProbK[,j] + fc$pk / samples
        updateProbS1[j] <- updateProbS1[j] + fc$ps / samples
      } else {
        fc <- forecastStandard(y = y[1:Tseq[t+1], j],
                               x = x[1:Tseq[t+1],],
                               theta = extractArray4th(tDynFull, i),
                               order = order,
                               priorK = priorProbK[,j],
                               fcVar = extractArray4th(fcVar, i),
                               support = support,
                               Tn = Tseq[t] - Tseq[1],
                               var = FALSE)
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
  group_by(h, switch) %>%
  summarise(pred = sum(exp(map)),
            actual = sum(exp(actual))) %>%
  mutate(error = abs(actual - pred) / actual) %>%
  ggplot() + geom_line(aes(h, error, colour = factor(switch)))

results %>%
  group_by(h, switch) %>%
  summarise(ls = mean(ls, na.rm = TRUE)) %>%
  ggplot() + geom_line(aes(h, ls, colour = factor(switch)))

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


# Run VB on a a standard model to get VAR starting points
fit <- fitStd
lambdaVar <- NULL
for(ki in 1:K[k]){
  mu <- fit$lambda[1:dim +(ki-1)* 306]
  L <- matrix(fit$lambda[(ki-1)*306 + dim + 1:dim^2], byrow = T, nrow =dim)
  logsd <- log(diag(L %*% t(L)))
  lambdaVar <- c(lambdaVar, rep(mu, rep(48, dim)), rep(logsd, rep(48, dim)))
}

priorMean = matrix(0, 48*dim, K[k])
priorLinv = array(0, dim = c(48, 48, dim*K[k]))

varStart <- 1
varUpdate <- 0.01
varSeq <- cumsum(c(varStart, rep(varUpdate, 47)))
Sigma <- matrix(0, 48, 48)
for(i in 1:48){
  for(j in 1:48){
    Sigma[i, j] <- min(varSeq[i], varSeq[j])
  }
}
for(ki in 1:K[k]){
  for(i in 1:dim){
    priorLinv[,,(ki-1)*dim + i] <- t(chol(solve(Sigma)))[1:10, 1:10]
  }
}

var <- fitVB(data = y[1:Tseq[2],],
             lambda = lambdaVar,
             model = elecStdVAR,
             dimTheta = 48 * dim * K[k],
             S = 50,
             maxIter = 2000,
             threshold = 0.2,
             priorMean = priorMean,
             priorLinv = priorLinv,
             probK = priorProbK, 
             order = order,
             Tn = Tseq[2] - Tseq[1],
             x = x[1:Tseq[2],],
             batch = batchSize)

### Forecast for the VAR

updateMean <- updateSd <- array(dim = c(dim, 48, K[k]))
updateLinv <- array(0, dim = c(48, 48, dim * K[k]))

for(ki in 1:K[k]){
  updateMean[,,ki] <- matrix(var$lambda[48*dim*2*(ki-1) + 1:(48*dim)], ncol = 48, byrow = TRUE)
  updateSd[,,ki] <- matrix(var$lambda[48*dim*(2*(ki-1)+1) + 1:(48*dim)], ncol = 48, byrow = TRUE)
  for(i in 1:dim){
    updateLinv[,,(ki-1)*dim + i] <- diag(updateSd[i,,ki])
  }
}

# Sample theta values
tFull <- array(0, dim = c(dim, 48, K[k], samples))
fcVar <- array(0, dim = c(48, 48, K[k], samples))
theta <- array(0, dim = c(dim, 48, samples))
for(i in 1:samples){
  
  for(ki in 1:K[k]){
    theta <- updateMean[,,ki]  + updateSd[,,ki] *  matrix(rnorm(480), ncol = 48)
    
    for(j in 1:48){
      root <- polyroot(c(1, -theta[11:13, j]))
      if(any(Mod(root) < 1)){
        if(i > 1){
          theta[,j] <- tFull[, j, ki, i-1]
        } else {
          stat <- FALSE
          try <- 0
          while(!stat){
            theta[, j] <-updateMean[,,ki]  + updateSd[,,ki] *  matrix(rnorm(480), ncol = 48)
            root <- polyroot(c(1, -theta[11:13, j]))
            try <- try + 1
            if(all(Mod(root) > 1)){
              stat <- TRUE
            } else if(try > 10){
              theta[4:6, j] <- 0
              stat <- TRUE
            }
          }
        }
      }
      autocov <- ltsa::tacvfARMA(theta[11:13, j], sigma2 = exp(theta[1, j]), maxLag = 48)
      fcVar[, j, ki, i] <- ltsa::PredictionVariance(autocov, 48)
    }
    tFull[,, ki, i] <- theta
  } 
}

# For each household:
# 1) Calculate p(k = j) / p(S_T = 1)
# 2) Create forecast densities per k and combine with above probabilities
# 3) Calculate updated priors for p(k) and p(s) to be pushed into the next update
# 4) Evaluate forecast densities
# Steps 1) and 2) are handled in c++
resOld <- results
resOld$var = FALSE
results <- list()
counter <- 1
support <- seq(log(0.009), max(y[Tseq[t-1]:Tseq[t],]) + sd(y[Tseq[t-1]:Tseq[t],]), length.out = 5000)
probK <- matrix(0, K[k], N)
for(j in 1:N){
  density <- matrix(0, 5000, 48)
  for(i in 1:samples){
    fc <- forecastStandardVAR(y[1:Tseq[t+1], j], x[1:Tseq[t+1], ], tFull[,,,i], fcVar[,,,i], order, priorPS[j], priorProbK[,j], support, Tseq[t] - Tseq[1])
    density <- density + fc$density / samples
    probK[,j] <- probK[,j] + fc$pk / samples
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
                                     switch = switch,
                                     var = TRUE)
    counter <- counter + 1
  }
}
results <- bind_rows(results)

results <- rbind(reuslts, resOld)

results %>%
  filter(switch == 0) %>%
  group_by(h, var) %>%
  summarise(ls = mean(ls)) %>%
  ggplot() + geom_line(aes(h, ls, colour = var))


results %>%
  filter(switch == 0) %>%
  group_by(h, var) %>%
  summarise(map = sum(exp(map)),
            actual = sum(exp(actual))) %>%
  mutate(ape = abs(map - actual) / actual) %>%
  ungroup() %>%
  ggplot() + geom_line(aes(h, ape, colour = var))
  
updateMean[1,,] <- exp(updateMean[1,,] + 0.5 * updateSd[1,,]^2) 

varMean <- data.frame(mean = c(updateMean), 
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
  gather(var, mean, -group, -halfhour) -> varFit

# Get regular model estimates back
standardMean <- NULL
for(ki in 1:K[k]){
  mu <- fit$lambda[(ki-1)*dim*(dim+1) + 1:dim]
  SigSd <- fit$lambda[(ki-1)*dim*(dim+1) + dim + 1]
  standardMean <- c(standardMean, exp(mu[1] + 0.5 * SigSd^2), mu[2:dim])
}
  
standard <- data.frame(mean = standardMean, 
                       var = c('sigma^{2}', 'intercept', 'temp', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'publicHol',
                                   'phi[1]', 'phi[2]', 'phi[3]', 'phi[48]', 'phi[96]', 'phi[144]', 'phi[336]'),
                       group = rep(1:3, rep(17, 3)))
standard %>%
  spread(var, mean) %>%
  mutate(day1 = intercept + day1,
         day2 = intercept + day2,
         day3 = intercept + day3,
         day4 = intercept + day4,
         day5 = intercept + day5,
         day6 = intercept + day6) %>%
  rename(Sunday = intercept, Monday = day1, Tuesday = day2, Wedesday = day3, Thursday = day4, Friday = day5, Saturday = day6) %>%
  gather(var, mean, -group) -> standard

ggplot(varFit) + geom_line(aes(halfhour, mean, colour = factor(group))) + 
  geom_hline(data = standard, aes(yintercept = standardMean, colour =  factor(group))) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed, ncol = 5) + 
  theme_bw()


ps <- array(0, dim = c(Tseq[2] - Tseq[1], N, samples))
for(i in 1:samples){
  for(j in 1:N){
    ps[,j, i] <- probSKHF(y[1:Tseq[2], j], x[1:Tseq[2]], tFull[,,i], tCons[i,1:2], order, 0.5, tCons[i, 3:4], 1, Tseq[2] - Tseq[1])[-1]
  }
}
c(apply(ps, 1:2, mean)) -> ps1

y[(Tseq[1] + 1):Tseq[2], ] %>%
  as.data.frame() %>%
  cbind(t = (Tseq[1] + 1):Tseq[2]) %>%
  gather(id, energy, -t) %>%
  cbind(ps1 = ps1) %>%
  ggplot() + geom_line(aes(t, energy, colour = ps1, group = id)) + facet_wrap(~id, scales = 'free') + 
  scale_colour_gradient(low = 'darkred', high = 'darkblue')
