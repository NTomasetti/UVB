library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
library(tidyverse)
#sourceCpp('SV.cpp')

fitVB <- function(data, lambda, model, S = 25, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01, 
                  dimTheta = 6, zEpsilon = TRUE, ...){
  if(!is.matrix(lambda)){
    lambda <- matrix(lambda, ncol = 1)
  }
  dimLambda <- length(lambda)
  sobol <- sobol_points(100+S, dimTheta)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- V <- numeric(dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold){
    if(iter > maxIter){
      break
    }
    eval <- numeric(S)
    grad <- matrix(0, dimLambda, S)
    q <- numeric(S)
    unif <- shuffle(sobol)
    unif[unif < 0.001] = 0.001
    unif[unif > 0.999] = 0.999
    if(S == 1){
      epsilon <- unif[101,]
      if(zEpsilon){
        epsilon <- qnorm(epsilon)
      }
      logpj <- model(data, lambda, epsilon, ...)
      eval <- logpj$val
      grad <- logpj$grad
      q <- sum(dnorm(epsilon, log=TRUE))
      gradient <- grad
      gradientSq <- grad^2
      LB[iter] <- eval - q
    } else {
      for(s in 1:S){
        epsilon <- unif[s + 100,]
        if(zEpsilon){
          epsilon <- qnorm(epsilon)
        }
        logpj <- model(data, lambda, epsilon, ...)   
        eval[s] <- logpj$val
        grad[,s] <- logpj$grad
        q <- sum(dnorm(qnorm(unif[s+100,]), log = TRUE))
      }
      eval[eval == -Inf] = NA
      gradient <- rowMeans(grad, na.rm = TRUE)
      gradientSq <- rowMeans(grad^2, na.rm = TRUE)
      
      LB[iter] <- mean(eval - q, na.rm=TRUE) 
    }
    M <- beta1 * M + (1 - beta1) * gradient
    V <- beta2 * V + (1 - beta2) * gradientSq
    Mstar <- M / (1 - beta1^iter)
    Vstar <- V / (1 - beta2^iter)
    update <- alpha * Mstar / (sqrt(Vstar) + e)
    if(any(is.na(update))){
      print('Break')
      break
    }
    lambda <- lambda + update
    if(iter %% 5 == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-4)])
      diff <- abs(meanLB - oldMeanLB)
    } 
    if(iter %% 100 == 0){
      print(paste0('Iteration: ', iter, ' ELBO: ', LB[iter]))
    }
    iter <- iter + 1
  }
  return(list(lambda=lambda, LB = LB[1:min(iter-1, maxIter)], iter = min(maxIter, iter-1)))
}

fitVBScore <- function(data, lambda, model, dimTheta = 4, mix = 4, S = 50, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01, ...){
  dimLambda <- nrow(lambda)
  sobol <- sobol_points(100+6*S, dimTheta)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- rep(0, dimLambda)
  V <- rep(0, dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold | iter < 100){
    if(iter > maxIter){
      break
    }
    grad <- matrix(0, dimLambda, S)
    eval <- numeric(S)
    z <- lambda[dimLambda + (1-mix):0]
    pi <- exp(z) / sum(exp(z))
    unif <- shuffle(sobol)
    s <- 0
    try <- 0
    epsilon <- qnorm(unif[101:(100+6*S), ])
    epsilon[epsilon < -3] = -3
    epsilon[epsilon > 3] = 3
    while(s < S){
      k <- sample(1:mix, 1, prob=pi)
      Qmean <- lambda[(k-1)*dimTheta + 1:dimTheta]
      Qsd <- exp(lambda[mix*dimTheta + (k-1)*dimTheta + 1:dimTheta])
      if(dimTheta > 1){
        theta <- c(Qmean + Qsd * epsilon[try+1,])
      } else {
        theta <- Qmean + Qsd * epsilon[try + 1]
      }
      derivs <- model(data, lambda, theta, mix = mix, ...)
      if(all(is.finite(derivs$grad)) & all(!is.na(derivs$grad)) & is.finite(derivs$val) & !is.na(derivs$val)){
        s <- s + 1
        eval[s] <- derivs$val
        grad[,s] <- derivs$grad
        if(s == S){
          gradient <- rowMeans(grad, na.rm = TRUE)
          gradientSq <- rowMeans(grad^2, na.rm = TRUE)
          LB[iter] <- mean(eval, na.rm = TRUE)
          break
        }
      }
      try <- try + 1
      if(try > 5*S){
        if(s > 1){
          gradient <- rowMeans(grad[,1:s], na.rm = TRUE)
          gradientSq <- rowMeans(grad[,1:s]^2, na.rm = TRUE)
          LB[iter] <- mean(eval[1:s], na.rm = TRUE)
        } else {
          LB[iter] <- LB[iter-1] - 1
        }
        break
      }
    }
    
    M <- beta1 * M + (1 - beta1) * gradient
    V <- beta2 * V + (1 - beta2) * gradientSq
    Mst <- M / (1 - beta1^iter)
    Vst <- V / (1 - beta2^iter)
    if(any(is.na(alpha * Mst / sqrt(Vst + e)))){
      print('Break')
      break
    }
    lambda <- lambda + alpha * Mst / sqrt(Vst + e)
    if(iter %% 5 == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-4)])
      diff <- abs(meanLB - oldMeanLB)
    } 
    if(iter %% 100 == 0){
      print(paste0('iter: ', iter, ' ELBO: ', LB[iter]))
    }
    iter <- iter + 1
  }
  print(paste0('iter: ', min(iter-1, maxIter), ' ELBO: ', LB[min(iter-1, maxIter)]))
  return(list(lambda=lambda, LB = LB, iter = iter))
}

vbDensity <- function(fit, transform, names, supports = NULL){
  n = length(transform)
  if(is.null(supports)){
    supports = as.list(rep(NA, n))
  }
  mu = fit$mean
  if(length(fit$U) == n^2){
    u = matrix(fit$U, n)
    sigma = sqrt(diag(t(u) %*% u))
  } else {
    sigma = fit$U
  }
  dens = data.frame()
  for(i in 1:n){
    if(transform[i] == 'exp'){
      if(is.na(supports[[i]][1])){
        mean = exp(mu[i] + 0.5 * sigma[i]^2)
        stdev = sqrt((exp(sigma[i]^2) - 1)*exp(2*mu[i]+sigma[i]^2))
        support = seq(max(1e-08, mean - 5*stdev), mean+5*stdev, length.out=1000)
      } else {
        support = supports[[i]]
      }
      density = dlnorm(support, mu[i], sigma[i])
    } else if (transform[i] == 'sigmoid') {
      if(is.na(supports[[i]][1])){
        sample = 1 / (1 + exp(-rnorm(1000, mu[i], sigma[i])))
        mean = mean(sample)
        stdev = sd(sample)
        support = seq(max(0.001, mean-5*stdev), min(0.999, mean+5*stdev), length.out=1000)
      } else {
        support = supports[[i]]
      }
      density = dnorm(log(support / (1-support)), mu[i], sigma[i]) / (support - support^2)
    } else if(transform[i] == 'identity') {
      if(is.na(supports[[i]][1])){
        support = seq(mu[i] - 5*sigma[i], mu[i] + 5*sigma[i], length.out=1000)
      } else {
        support = supports[[i]]
      }
      density = dnorm(support, mu[i], sigma[i])
    } else if(transform[i] == 'stretchedSigmoid'){
      if(is.na(supports[[i]][1])){
        sample = 2 / (1 + exp(-rnorm(1000, mu[i], sigma[i]))) - 1
        mean = mean(sample)
        stdev = sd(sample)
        support = seq(max(-0.999, mean-5*stdev), min(0.999, mean+5*stdev), length.out=1000)
      } else {
        support = supports[[i]]
      }
      density = dnorm(-log(2/(support+1)-1), mu[i], sigma[i]) * 2 / (2*(support+1) - (support+1)^2)
    }
    df = data.frame(support = support, density = density, var = names[i])
    dens = rbind(dens, df)
  }
  dens
}

mixQuantiles <- function(q, mean, sd, weights){
  lower <- min(mean - 5 * sd)
  upper <- max(mean + 5 * sd)
  support <- seq(lower, upper, length.out = 2000)
  density <- numeric(2000)
  mix <- length(mean)
  for(k in 1:mix){
    density <- density + weights[k] * dnorm(support, mean[k], sd[k])
  }
  CDF <- cumsum(density) / sum(density)
  quantiles <- numeric(length(q))
  for(k in 1:length(q)){
    x0 <- CDF[max(which(CDF < q[k]))]
    x1 <- CDF[min(which(CDF > q[k]))]
    y0 <- support[max(which(CDF < q[k]))]
    y1 <- support[min(which(CDF > q[k]))]
    quantiles[k] <- y0 + (q[k] - x0) * (y1 - y0) / (x1 - x0)
  }
  quantiles
}

# AR(P) Models

ARP{

T <- 500
H <- 10
lags <- 4
mu <- 1
phi <- c(0.5, -0.2, -0.1, 0.25)
sigmaSq <- 1
x <- rnorm(T+100+H)
for(t in (lags+1):(T+100+H)){
  x[t] <- mu + sum(phi * (x[t-(lags:1)] -mu)) + rnorm(1, 0, sqrt(sigmaSq))
}

qplot(1:T, x[1:T + 100], geom = 'line')


K <- 2
dim <- 2 + lags

mean <- rep(0, dim)
Linv <- solve(t(chol(diag(10, dim))))
lambda <- matrix(c(rep(0, dim*K) + rnorm(dim*K, 0, 0.1), rep(-1, dim*K), rep(0, K)), ncol = 1)

batches <- 50
data <- seq(1000, T+100, length.out = batches+1)
fit <- matrix(0, nrow(lambda), batches)

fit[,1] <- fitVBScore(x[(data[1]+1):data[2]], lambda, ARpdiff, S = 10, dimTheta = dim, mix = K, mean = mean, Linv = Linv)$lambda
for(t in 2:batches){
  mean <- matrix(fit[1:(dim*K), t-1], ncol = K)
  siginv <- array(0, dim = c(dim, dim, K))
  dets <- rep(0, K)
  for(k in 1:K){
    sd <- exp(fit[dim*K + (k-1)*dim + 1:dim, t-1])
    siginv[,,k] <- diag(1 / sd^2)
    dets[k] <- 1 / prod(sd)
  }
  piZ <- fit[2*dim*K + 1:K, t-1]
  weights <- exp(piZ) / sum(exp(piZ))
  lambda <- matrix(fit[,t-1], ncol = 1)
  fit[,t] <-fitVBScore(x[(data[t]+1-lags):data[t+1]], lambda, ARpdiffMix, S = 10, dimTheta = dim, mix = K, mean = mean, SigInv = siginv, dets = dets, weights = weights)$lambda
}

margins <- NULL
supports <- list(seq(0.5, 4, length.out = 1000),
                 seq(-2, 4, length.out = 1000),
                 seq(-1, 1.5, length.out = 1000),
                 seq(-1, 1.5, length.out = 1000),
                 seq(-1, 1.5, length.out = 1000),
                 seq(-1, 1.5, length.out = 1000))
names <- c('sigma^{2}', 'mu', paste0('phi[', 1:lags, ']'))
transform <- c('exp','identity', rep('identity', lags))
for(t in ceiling(seq(2, batches, length.out = 10))){
  piZ <- fit[2*dim*K + 1:K, t]
  weights <- exp(piZ) / sum(exp(piZ))
 
  mean <- matrix(fit[1:(dim*K), t], ncol = K)
  u <- matrix(exp(fit[dim*K + 1:(dim*K), t]), ncol = K)
  densMix <- NULL
  for(k in 1:K){
    dens <- vbDensity(list(mean = mean[,k], U = u[,k]), transform, names, supports)
    dens$t <- t
    dens$group <- k
    dens$pi <- weights[k]
    densMix <- rbind(densMix, dens)
  }
  densMix %>%
    group_by(support, var, t) %>%
    summarise(density = sum(density * pi)) -> densMix
    
  margins <- rbind(margins, densMix)
}

ggplot(margins) + 
  geom_line(aes(support, density, colour = factor(t))) +
  scale_colour_brewer(palette = 'Reds') + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
  theme(legend.position = 'none')

forecasts <- NULL
support <- seq(min(x)-sd(x), max(x)+sd(x), length.out = 1000)
for(t in 1:batches){
  dataSub <- x[data[t+1] + (-lags:H)]
  piZ <- fit[2*dim*K + 1:K, t]
  weights <- exp(piZ) / sum(exp(piZ))
  mean <- matrix(fit[1:(dim*K), t], ncol = K)
  u <- matrix(exp(fit[dim*K + 1:(dim*K), t]), ncol = K)
  density <- matrix(0, 1000, H)
  for(i in 1:1000){
    k <- sample(1:K, 1, prob = weights)
    draw <- mvtnorm::rmvnorm(1, mean[,k], diag(u[,k]^2))
    sigSq <- exp(draw[1])
    phi <- draw[2:(lags+1)]
    lagX <- x[lags:1]
    lagVar <- rep(0, lags)
    for(h in 1:H){
      mean <- lagX %*% phi
      var <- sigSq + lagVar %*% phi^2
      density[,h] <- density[,h] + dnorm(support, mean, sqrt(var)) / 1000
      lagX <- c(mean, lagX[1:(lags-1)])
      lagVar <- c(var, lagVar[1:(lags-1)])
    }
  }
  for(h in 1:H){
    lower <- max(which(support < dataSub[lags+h]))
    upper <- min(which(support > dataSub[lags+h]))
    densityInterpolate <- (density[lower, h] * (support[upper] - dataSub[lags+h]) + density[uppper, h] * (dataSub[lags+h] - support[lower])) / (support[upper] - support[lower])
    forecasts <- rbind(forecasts,
                       data.frame(t = data[t+1],
                                  h = h,
                                  ls = log(densityInterpolate)))
  }
}
ggplot(forecasts) + geom_line(aes(t, ls)) + facet_wrap(~h, scales = 'free')


}

# Various State SPace Model Setups, fixed parameters, single or mixture approximations

DLM-Fixed{
  alpha <- -2
  beta <- 0.95
  sigmaSq <- 0.1
  K <- 3
  
  T <- 200
  z <- rnorm(1, alpha, sqrt(0.1 / (1 - 0.95^2)))
  for(t in 2:T){
    z <- c(z, rnorm(1, alpha + beta * (z[t-1] - alpha), sqrt(sigmaSq)))
  }
  x <- rnorm(T, z, sqrt(sigmaSq))
  
  qplot(1:T, z)
  qplot(1:T, x)
  lambda <- matrix(c(rnorm(K, alpha, 0.1), rnorm(K, 0, 0.1), rep(0, K)), ncol = 1)
  fit <- matrix(0, 3*K, T)
  
  fit[,1] <- fitVBScore(x[1], lambda, LG1mixdiffF, dimTheta = 1, mix = K, S = 25)$lambda  
  
  for(t in 2:T){
    piZ <- fit[2*K + 1:K, t-1]
    weights <- exp(piZ) / sum(exp(piZ))
    zStats <- rep(0, 2*K)
    for(k in 1:K){
      zStats[2*k -1] <- fit[k, t-1]
      zStats[2*k] <- exp(fit[K + k, t-1])
    }
    fit[,t] <- fitVBScore(x[t], lambda, LGtmixdiffF, dimTheta = 1, mix = K, S = 25, zStats = zStats, weights = weights)$lambda
    if(t %% 10 == 0){
      print(t)
    }
  }
  states <- NULL
 
  for(t in 1:T){
    piZ <- fit[2*K + 1:K, t]
    weights <- exp(piZ) / sum(exp(piZ))
    meanZ <- fit[1:K, t]
    sdZ <- exp(fit[K + 1:K, t])
    
    mean <- sum(weights * meanZ)
    interval <- mixQuantiles(c(0.025, 0.975), meanZ, sdZ, weights)
    
    states <- rbind(states,
                    data.frame(t = t,
                               mean = mean,
                               lower = interval[1],
                               upper = interval[2]))
  }
  ggplot(states) + geom_line(aes(t, mean), colour = 'red') + 
    geom_ribbon(aes(t, ymin = lower, ymax = upper), fill = 'red', alpha = 0.5) + 
    geom_line(aes(t, z))
  
  
  
}

SVM-Single{

  T = 100
  alpha = -2
  beta = 0.95
  sigmaSq = 0.1
  
  z <- rnorm(1, alpha, sqrt(sigmaSq / (1 - beta^2)))
  x <- exp(z/2) * rnorm(1)
  
  for(t in 2:T){
    zMean <- alpha + beta * (z[t-1] - alpha)
    z <- c(z, rnorm(1, zMean, sqrt(sigmaSq)))
    x <- c(x, exp(z[t]/2) * rnorm(1))
  }
  
qplot(1:T, x, geom = 'line')
qplot(1:T, z, geom = 'line')
qplot(x, z)

Mean <- c(-2, 0, 0)
Linv <- solve(t(chol(diag(10, 3))))
lambda <- matrix(c(rep(0, 3), diag(0.1, 3), 0, 1), ncol = 1)

fit <- matrix(0, 14, T)
fit[,1] <- fitVB(x[1], lambda, SV1diff, S = 5, dimTheta = 4, mean = Mean, Linv = Linv)$lambda
for(t in 2:T){
  mean = fit[1:3,t-1]
  L = matrix(fit[4:12, t-1], 3, byrow = TRUE)
  zStats = fit[13:14, t-1]
  fit[, t] = fitVB(x[t], lambda, SVtdiff, dimTheta = 4, S = 3, mean = mean, Linv = solve(L), zStats = zStats)$lambda
}

states <- NULL
margins <- NULL
supports <- list(seq(0.01, 1.5, length.out = 1000),
                 seq(-1, 1, length.out = 1000),
                 seq(-1, 1, length.out = 1000))
names <- c('sigma^{2}', 'alpha', 'beta')
transform <- c('exp','identity', 'identity')
for(t in 1:T){
  states <- rbind(states,
                  data.frame(t = t,
                             mean = fit[13, t],
                             lower = fit[13,t] - 1.96 * abs(fit[14, t]),
                             upper = fit[13, t] + 1.96 * abs(fit[14, t])))
  if(t %% floor(T / 9) == 0){
    mu <- fit[1:3, t-1]
    u <- fit[4:12, t-1]
    f <- list(mean = mu, U = u)
    dens <- vbDensity(f, transform, names, supports)
    dens$t <- t
    margins <- rbind(margins, dens)
  }
}
ggplot(states) + geom_line(aes(t, mean), colour = 'red') + 
  geom_ribbon(aes(t, ymin = lower, ymax = upper), fill = 'red', alpha = 0.5) + 
  geom_line(aes(t, z))

true <- data.frame(var = names,
                   true = c(sigmaSq, alpha, beta))

ggplot(margins) + 
  geom_line(aes(support, density, colour = factor(t))) +
  scale_colour_brewer(palette = 'Reds') + 
  geom_vline(data = true, aes(xintercept = true)) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
  theme(legend.position = 'none')
}

DLM-Single{

sigmaSqX <- 0.25
x <- rnorm(T, z, sqrt(sigmaSqX))

Mean <- c(0, 0, 0, 0)
Linv <- solve(t(chol(diag(10, 4))))
lambda <- matrix(c(rep(0, 4), diag(0.1, 4), 0, 1), ncol = 1)

fit <- matrix(0, 22, T)
fit[,1] <- fitVB(x[1], lambda, LG1diff, dimTheta = 5, mean = Mean, Linv = Linv, S = 5)$lambda
for(t in 2:T){
  mean = fit[1:4,t-1]
  L = matrix(fit[5:20, t-1], 4, byrow = TRUE)
  zStats = fit[21:22, t-1]
  fit[, t] = fitVB(x[t], lambda, LGtdiff, dimTheta = 5, mean = mean, Linv = solve(L), zStats = zStats, S = 5)$lambda
}

states <- NULL
margins <- NULL
supports <- list(seq(0.01, 1.5, length.out = 1000),
                 seq(0.01, 1.5, length.out = 1000),
                 seq(-1, 1, length.out = 1000),
                 seq(-1, 1, length.out = 1000))
names <- c('sigma^{2}[x]', 'sigma^{2}[z]', 'alpha', 'beta')
transform <- c('exp', 'exp', 'identity', 'identity')
for(t in 1:T){
  states <- rbind(states,
                  data.frame(t = t,
                             mean = fit[21, t],
                             lower = fit[21,t] - 1.96 * abs(fit[22, t]),
                             upper = fit[21, t] + 1.96 * abs(fit[22, t])))
  if(t %% floor(T/20) == 0){
    mu <- fit[1:4, t-1]
    u <- fit[5:20, t-1]
    f <- list(mean = mu, U = u)
    dens <- vbDensity(f, transform, names, supports)
    dens$t <- t
    margins <- rbind(margins, dens)
  }
}
ggplot(states) + geom_line(aes(t, mean), colour = 'red') + 
  geom_ribbon(aes(t, ymin = lower, ymax = upper), fill = 'red', alpha = 0.5) + 
  geom_line(aes(t, z))

true <- data.frame(var = names,
                   true = c(sigmaSqX, sigmaSq, alpha, beta))

ggplot(margins) + 
  geom_line(aes(support, density, colour = factor(t))) +
  scale_colour_brewer(palette = 'Reds') + 
  geom_vline(data = true, aes(xintercept = true)) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
  theme(legend.position = 'none')
}

SVM-Mix{
K <- 3
Mean <- c(-2, -2, 0)
Linv <- solve(t(chol(diag(10, 3))))
lambda <- matrix(c(rep(c(-2, -2, 0.5, -3), K) + rnorm(4 * K, 0, 0.1), rnorm(4 * K, -2, 0.2), rep(0, K)), ncol = 1)

fitMix <- matrix(0, 9 * K, T)
fitMix[,1] <- fitVBScore(x[1], lambda, SV1mixdiff, S = 10, dimTheta = 4, mix = K, mean = Mean, Linv = Linv)$lambda
for(t in 2:T){
  mean <- matrix(0, 3, K)
  SigInv <- array(0, dim = c(3, 3, K))
  zStats <- rep(0, K * 2)
  dets <- rep(0, K)
  for(k in 1:K){
    mean[,k] <- fitMix[(k-1)*4 + 1:3, t-1]
    sd <- exp(fitMix[4*K + (k-1)*4 + 1:3, t-1])
    SigInv[,,k] <- diag(1/sd^2)
    zStats[k*2 - 1] <- fitMix[k*4, t-1]
    zStats[k*2] <- exp(fitMix[K*4 + k*4, t-1])
    dets[k] <- 1 / prod(sd)
  }
  piZ <- fitMix[8*K + 1:K, t-1]
  weights <- exp(piZ) / sum(exp(piZ))
  lambda <- matrix(c(rep(c(-2, -2, 0.5, -3), K) + rnorm(4 * K, 0, 0.1), rnorm(4 * K, -2, 0.2), rep(0, K)), ncol = 1)
  fitMix[, t] = fitVBScore(x[t], lambda, SVtmixdiff, dimTheta = 4, mix = K, S = 10, mean = mean, 
                           SigInv = SigInv, dets = dets, weights = weights, zStats = zStats)$lambda
  if(t %% 10 == 0){
    print(t)
  }
}

states <- NULL
margins <- NULL
supports <- list(seq(0.001, 0.2, length.out = 1000),
                 seq(-5, 0, length.out = 1000),
                 seq(-0.2, 1.5, length.out = 1000))
names <- c('sigma^{2}', 'alpha', 'beta')
transform <- c('exp','identity', 'identity')
for(t in 1:T){
  piZ <- fitMix[8*K + 1:K, t]
  weights <- exp(piZ) / sum(exp(piZ))
  meanZ <- fitMix[seq(4, 4*K, 4), t]
  sdZ <- exp(fitMix[seq(4*K + 4, 8*K, 4), t])
  
  mean <- sum(weights * meanZ)
  interval <- mixQuantiles(c(0.025, 0.975), meanZ, sdZ, weights)
  
  
  states <- rbind(states,
                  data.frame(t = t,
                             mean = mean,
                             lower = interval[1],
                             upper = interval[2]))
  if(t %% floor(T / 9) == 0){
    mu <- matrix(fitMix[(1:(4*K))[-seq(4, 4*K, 4)], t], ncol = K)
    u <- matrix(0, 3, K)
    densMix <- NULL
    for(k in 1:K){
      u[,k] <- exp(fitMix[4*K + (k-1) + 1:3, t])
      dens <- vbDensity(list(mean = mu[,k], U = u[,k]), transform, names, supports)
      dens$t <- t
      dens$group <- k
      dens$pi <- weights[k]
      densMix <- rbind(densMix, dens)
    }
    densMix %>%
      group_by(support, var, t) %>%
      summarise(density = sum(density * pi)) -> densMix
    
    margins <- rbind(margins, densMix)
  }
}
ggplot(states) + geom_line(aes(t, mean), colour = 'red') + 
  geom_ribbon(aes(t, ymin = lower, ymax = upper), fill = 'red', alpha = 0.5) + 
  geom_line(aes(t, z))

true <- data.frame(var = names,
                   true = c(sigmaSq, alpha, beta))

ggplot(margins) + 
  geom_line(aes(support, density, colour = factor(t))) +
  scale_colour_brewer(palette = 'Reds') + 
  geom_vline(data = true, aes(xintercept = true)) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
  theme(legend.position = 'none')
}

DLM-Mix{
sigmaSqX <- 0.25
x <- rnorm(T, z, sqrt(sigmaSqX))
K <- 3

Mean <- c(0, 0, 0, 0)
Linv <- solve(t(chol(diag(10, 4))))
lambda <- matrix(c(rep(c(-1, -2, -2, 0.5, -3), K) + rnorm(5 * K, 0, 0.1), rnorm(5 * K, -2, 0.2), rep(0, K)), ncol = 1)

fitMix <- matrix(0, 11 * K, T)
fitMix[,1] <- fitVBScore(x[1], lambda, LG1mixdiff, S = 10, dimTheta = 5, mix = K, mean = Mean, Linv = Linv)$lambda
for(t in 2:T){
  mean <- matrix(0, 4, K)
  SigInv <- array(0, dim = c(4, 4, K))
  zStats <- rep(0, K * 2)
  dets <- rep(0, K)
  for(k in 1:K){
    mean[,k] <- fitMix[(k-1)*5 + 1:4, t-1]
    sd <- exp(fitMix[5*K + (k-1)*5 + 1:4, t-1])
    SigInv[,,k] <- diag(1/sd^2)
    zStats[k*2 - 1] <- fitMix[k*5, t-1]
    zStats[k*2] <- exp(fitMix[K*5 + k*5, t-1])
    dets[k] <- 1 / prod(sd)
  }
  piZ <- fitMix[10*K + 1:K, t-1]
  weights <- exp(piZ) / sum(exp(piZ))
  lambda <- matrix(c(rep(c(-2, -2, -2, 0.5, -3), K) + rnorm(5 * K, 0, 0.1), rnorm(5 * K, -2, 0.2), rep(0, K)), ncol = 1)
  fitMix[, t] = fitVBScore(x[t], lambda, LGtmixdiff, dimTheta = 5, mix = K, S = 10, mean = mean, 
                           SigInv = SigInv, dets = dets, weights = weights, zStats = zStats)$lambda
  if(t %% 10 == 0){
    print(t)
  }
}


states <- NULL
margins <- NULL
supports <- list(seq(0.01, 1.5, length.out = 1000),
                 seq(0.01, 1.5, length.out = 1000),
                 seq(-7, 0, length.out = 1000),
                 seq(-1, 1, length.out = 1000))
names <- c('sigma[x]^{2}', 'sigma[z]^{2}', 'alpha', 'beta')
transform <- c('exp', 'exp', 'identity', 'identity')
for(t in 1:T){
  piZ <- fitMix[10*K + 1:K, t]
  weights <- exp(piZ) / sum(exp(piZ))
  meanZ <- fitMix[seq(5, 5*K, 5), t]
  sdZ <- exp(fitMix[seq(5*K + 5, 10*K, 5), t])
  
  mean <- sum(weights * meanZ)
  interval <- mixQuantiles(c(0.025, 0.975), meanZ, sdZ, weights)
  
  
  states <- rbind(states,
                  data.frame(t = t,
                             mean = mean,
                             lower = interval[1],
                             upper = interval[2]))
  if(t %% floor(T / 9) == 0){
    mu <- matrix(fitMix[(1:(5*K))[-seq(5, 5*K, 5)], t], ncol = K)
    u <- matrix(0, 4, K)
    densMix <- NULL
    for(k in 1:K){
      u[,k] <- exp(fitMix[5*K + (k-1) + 1:4, t])
      dens <- vbDensity(list(mean = mu[,k], U = u[,k]), transform, names, supports)
      dens$t <- t
      dens$group <- k
      dens$pi <- weights[k]
      densMix <- rbind(densMix, dens)
    }
    densMix %>%
      group_by(support, var, t) %>%
      summarise(density = sum(density * pi)) -> densMix
    
    margins <- rbind(margins, densMix)
  }
}
ggplot(states) + geom_line(aes(t, mean), colour = 'red') + 
  geom_ribbon(aes(t, ymin = lower, ymax = upper), fill = 'red', alpha = 0.5) + 
  geom_line(aes(t, z[1:T]))

true <- data.frame(var = names,
                   true = c(sigmaSqX, sigmaSq, alpha, beta))

ggplot(margins) + 
  geom_line(aes(support, density, colour = factor(t))) +
  scale_colour_brewer(palette = 'Reds') + 
  geom_vline(data = true, aes(xintercept = true)) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
  theme(legend.position = 'none')
}

### Mixture of Normals - Clustering / Classification

fitVBScoreMN <- function(data, lambda, prior, K, S = 50, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01){
  dimLambda <- nrow(lambda)
  N <- ncol(data)
  T <- nrow(data)
  sobol <- sobol_points(100+6*S, 2*K)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- rep(0, dimLambda)
  V <- rep(0, dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold | iter < 100){
    if(iter > maxIter){
      break
    }
    grad <- matrix(0, dimLambda, S)
    h <- matrix(0, dimLambda, S)
    eval <- numeric(S)
    unif <- shuffle(sobol)
    epsilon <- qnorm(unif[101:(100+6*S), ])
    epsilon[epsilon < -3] = -3
    epsilon[epsilon > 3] = 3
    theta <- lambda[1:(2*K)] + exp(lambda[2*K + 1:(2*K)]) * epsilon
    s <- 0
    try <- 0
    while(s < S){
      pi <- matrix(0, N, K)
      for(i in 1:N){
        pi[i,] <- MCMCpack::rdirichlet(1, exp(lambda[4*K + 1:K]))
      }
      
      derivs <- mixtureNormal(data, lambda, theta[try + 1,], pi, prior, K)
      if(all(is.finite(derivs$grad)) & all(!is.na(derivs$grad)) & is.finite(derivs$val) & !is.na(derivs$val)){
        s <- s + 1
        eval[s] <- derivs$val
        grad[,s] <- derivs$grad
        h[,s] <- derivs$grad / derivs$val
        if(s == S){
          a <- vapply(1:dimLambda, function(x) cov(grad[x, ], h[x,]) / var(h[x,]), 1)
          a[is.na(a)] <- 0
          gradient <- rowMeans(grad - a * h, na.rm = TRUE) 
          gradientSq <- rowMeans((grad - a * h)^2, na.rm = TRUE)
          LB[iter] <- mean(eval, na.rm = TRUE)
          break
        }
      }
      try <- try + 1
      if(try > 5*S){
        if(s > 1){
          a <- vapply(1:dimLambda, function(x) cov(grad[x, 1:S], h[x,1:s]) / var(h[x,1:s]), 1)
          a[is.na(a)] <- 0
          gradient <- rowMeans(grad[,1:s] - a * h[,1:s], na.rm = TRUE)
          gradientSq <- rowMeans((grad[,1:s] - a * h[,1:s])^2, na.rm = TRUE)
          LB[iter] <- mean(eval[1:s], na.rm = TRUE)
        } else {
          LB[iter] <- LB[iter-1] - 1
        }
        break
      }
    }
    
    M <- beta1 * M + (1 - beta1) * gradient
    V <- beta2 * V + (1 - beta2) * gradientSq
    Mst <- M / (1 - beta1^iter)
    Vst <- V / (1 - beta2^iter)
    if(any(is.na(alpha * Mst / sqrt(Vst + e)))){
      print('Break')
      return(matrix(c(lambda, Mst, Vst), ncol = 3))
    }
    lambda <- lambda + alpha * Mst / sqrt(Vst + e)
    if(iter %% 5 == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-4)])
      diff <- abs(meanLB - oldMeanLB)
    } 
    if(iter %% 100 == 0){
      print(paste0('iter: ', iter, ' ELBO: ', LB[iter]))
    }
    iter <- iter + 1
  }
  print(paste0('iter: ', min(iter-1, maxIter), ' ELBO: ', LB[min(iter-1, maxIter)]))
  return(list(lambda=lambda, LB = LB, iter = iter))
}

TwoGroups{
N <- 100
T <- 100
K <- 2
mu <- rnorm(K, 0, 0.25)
sigmaSq <- runif(2, 1, 2)
group <- sample(1:K, N, replace = TRUE)
y <- matrix(0, T, N)
stats <- data.frame()

for(i in 1:N){
  y[,i] <- rnorm(T, mu[group[i]], sqrt(sigmaSq[group[i]]))
  stats <- rbind(stats, data.frame(ID = i, group = group[i], mean = mean(y[,i]), var = var(y[,i])))
}
ggplot(stats) + geom_point(aes(mean, var, colour = group))

y %>%
  as.data.frame() %>%
  mutate(t = 1:T) %>%
  gather(var, val, -t) %>%
  mutate(var = rep(1:N, rep(T, N)),
         group = rep(group, rep(T, N))) %>%
  ggplot() + geom_line(aes(t, val, colour = factor(var))) + facet_wrap(~group) + 
  theme(legend.position = 'none')

y %>%
  as.data.frame() %>%
  mutate(t = 1:T) %>%
  gather(var, val, -t) %>%
  mutate(var = rep(1:N, rep(T, N)),
         group = rep(group, rep(T, N))) %>%
  group_by(group) %>%
  summarise(var = var(val), mean = mean(val)) %>%
  select(-group) %>%
  unlist() -> true

lambda <- matrix(c(rnorm(K), rnorm(K, -0.5, 0.2), c(diag(1, 4)), rnorm(2*N, 0, 0.25)), ncol = 1)
sourceCpp('mixtureNormal.cpp')

# All at once
fit <- fitVB(y, lambda, mixtureNormalRP, dimTheta = 4 + N, knownVar = FALSE, S = 5)
qplot(1:(fit$iter-1), fit$LB[1:(fit$iter-1)], geom = 'line')

dens <- vbDensity(list(mean = fit$lambda[1:4], U = fit$lambda[5:20]),
                       names = c('sigma[1]^{2}', 'sigma[2]^{2}', 'mu[1]', 'mu[2]'), 
                       transform = c('exp', 'exp', 'identity', 'identity'))
dens$type <- rep(c('variance', 'mean'), rep(2000, 2))
dens$group <- rep(1:2, rep(1000, 2))
ggplot(dens) + geom_line(aes(support, density, colour = var)) +
  geom_vline(data = data.frame(true = true, type = c('variance', 'variance', 'mean', 'mean'), group = c(1, 2, 1, 2)),
             aes(xintercept = true)) + 
  facet_wrap(group~type, scales = 'free')

kFit <- c(0, N)
for(i in 1:N){
  kFit[i] <- 1 + (fit$lambda[20 + i] < 0)
}
caret::confusionMatrix(factor(kFit), factor(group))$table
max(sum(kFit == group), sum(kFit != group)) / N

# Updating
batches <- 20
data <- seq(0, T, length.out = batches+1)
fit <- matrix(0, nrow(lambda), batches)
fit[,1] <- fitVB(y[(data[1]+1):data[2], ], lambda, mixtureNormalRP, dimTheta = 4 + N, S = 5, knownVar = FALSE)$lambda

for(t in 2:batches){
  mean <- fit[1:4,t-1]
  Linv <- solve(t(matrix(fit[5:20, t-1], 4)))
  piPrior <- matrix(0, N, 2)
  for(i in 1:N){
    piPrior[i,] <- c(fit[20 + i, t-1], fit[20 + N + i, t-1]^2)
  }
  fit[,t] <- fitVB(y[(data[t]+1):data[t+1], ], fit[,t-1], mixtureNormalRPU, K = K, dimTheta = 4 + N, S = 3, knownVar = FALSE,
                   mean = mean, Linv = Linv, pi = piPrior)$lambda
  print(t)
}

margins <- NULL
supports <- list(seq(0, 2, length.out = 1000),
                 seq(0, 2, length.out = 1000),
                 seq(-1, 1, length.out = 1000),
                 seq(-1, 1, length.out = 1000))
names = c('sigma[1]^{2}', 'sigma[2]^{2}', 'mu[1]', 'mu[2]') 
transform = c('exp', 'exp', 'identity', 'identity')

marginSeq <- seq(1, batches, length.out = 9)
for(t in seq_along(marginSeq)){
  s <- ceiling(marginSeq[t])
  dens <- vbDensity(list(mean = fit[1:4, s], U = fit[5:20, s]), names = names, transform = transform, supports = supports)
  dens$t <- s
  dens$type <- rep(c('variance', 'mean'), rep(2000, 2))
  margins <- rbind(margins, dens)
}  


ggplot(margins) + geom_line(aes(support, density, colour = factor(t))) +
  geom_vline(data = data.frame(true = true, var = c(names[1], names[2], names[3], names[4])),#names),
            aes(xintercept = true)) + 
  facet_wrap(~var, scales = 'free', labeller = label_parsed)

accuracy <- rep(0, batches+1)
for(t in 1:batches){
  kFit <- c(0, N)
  for(i in 1:N){
    kFit[i] <- 1 + (fit[20 + i, t] < 0)
  }
  accuracy[t+1] <- max(sum(kFit == group), sum(kFit != group)) / N
}

ggplot() + geom_line(aes(data, accuracy)) +
  labs(x = 'T') + ylim(0, 1)
}

fitVBGRP <- function(data, lambda, K = 3, S = 25, maxIter = 5000, eta = 0.01, gamma = 0.1, tau = 1, threshold = 0.01){
  if(!is.matrix(lambda)){
    lambda <- matrix(lambda, ncol = 1)
  }
  N <- ncol(data)
  dimLambda <- length(lambda)
  sobol <- sobol_points(100+S, 2*K)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- numeric(dimLambda)
  kappa <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold){
    if(iter > maxIter){
      break
    }
    grad <- matrix(0, dimLambda, S)
    eval <- numeric(S)
    unif <- shuffle(sobol)
    unif[unif < 0.001] = 0.001
    unif[unif > 0.999] = 0.999
    if(S == 1){
      epsilon <- qnorm(unif[101,])
      L <- t(matrix(lambda[2*K+1:(4*K^2)], 2*K))
      theta <- lambda[1:(2*K)] + L %*% epsilon
      q <- sum(dnorm(epsilon, log=TRUE))
      SigmaSqrt <- array(0, dim = c(K, K, N))
      for(i in 1:N){
        alpha <- lambda[2*K*(2*K+1) + K*(i-1) + 1:K]
        pi <- MCMCpack::rdirichlet(1, alpha)
        mu <- digamma(alpha) - digamma(sum(alpha))
        Sigma <- diag(trigamma(alpha)) - trigamma(sum(alpha))
        eig <- eigen(Sigma)
        SigmaSqrt[,,i] <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
        epsPi <- solve(SigmaSqrt[,,i]) %*% t(log(pi) - mu)
        theta <- rbind(theta, t(pi))
        epsilon <- c(epsilon, epsPi)
        q <- q + log(MCMCpack::ddirichlet(pi, alpha))
      }
      logpj <- mixtureNormalGRP(data, theta, lambda, K, epsilon, SigmaSqrt, q)
      
      gradient <- logpj$grad
      LB[iter] <- logpj$eval
    } else {
      for(s in 1:S){
        epsilon <- qnorm(unif[s + 100,])
        
        L <- t(matrix(lambda[2*K+1:(4*K^2)], 2*K))
        theta <- lambda[1:(2*K)] + L %*% epsilon
        q <- sum(dnorm(epsilon, log=TRUE))
        SigmaSqrt <- array(0, dim = c(K, K, N))
        for(i in 1:N){
          alpha <- lambda[2*K*(2*K+1) + K*(i-1) + 1:K]
          pi <- MCMCpack::rdirichlet(1, alpha)
          mu <- digamma(alpha) - digamma(sum(alpha))
          Sigma <- diag(trigamma(alpha)) - trigamma(sum(alpha))
          eig <- eigen(Sigma)
          SigmaSqrt[,,i] <- eig$vectors %*% diag(sqrt(eig$values)) %*% t(eig$vectors)
          epsPi <- solve(SigmaSqrt[,,i]) %*% t(log(pi) - mu)
          theta <- rbind(theta, t(pi))
          epsilon <- c(epsilon, epsPi)
          q <- q + log(MCMCpack::ddirichlet(pi, alpha))
        }
        logpj <- mixtureNormalGRP(data, theta, lambda, K, epsilon, SigmaSqrt, q)
    
        eval[s] <- logpj$val
        grad[,s] <- logpj$grad
      }
      eval[eval == -Inf] = NA
      gradient <- rowMeans(grad, na.rm = TRUE)
      LB[iter] <- mean(eval, na.rm=TRUE) 
    }
    M <- gamma * gradient^2 + (1-gamma) * M
    rho <- eta * iter^(-0.5 + kappa) / (tau + sqrt(M))
    update <- rho * gradient
    if(any(is.na(update))){
      print('Break')
      break
    }
    lambda <- lambda + update
    if(iter %% 5 == 0){
      oldMeanLB <- meanLB
      meanLB <- mean(LB[iter:(iter-4)])
      diff <- abs(meanLB - oldMeanLB)
    } 
    if(iter %% 100 == 0){
      print(paste0('Iteration: ', iter, ' ELBO: ', LB[iter]))
    }
    iter <- iter + 1
  }
  return(list(lambda=lambda, LB = LB[1:min(iter-1, maxIter)], iter = min(maxIter, iter-1)))
}
ThreePlusGroups{

N <- 15
T <- 100
K <- 3
mu <- rnorm(K, 0, 2)
sigmaSq <- runif(K, 0.5, 2.5)
pi <- MCMCpack::rdirichlet(1, rep(10, K))
k <- rmultinom(N, 1, pi)
y <- matrix(0, T, N)
group <- rep(0, N)

for(i in 1:N){
  group[i] <- which(k[,i] == 1)
  y[,i] <- rnorm(T, mu[group[i]], sigmaSq[group[i]])
}
y %>%
  as.data.frame() %>%
  mutate(t = 1:T) %>%
  gather(var, val, -t) %>%
  mutate(var = rep(1:N, rep(T, N)),
         group = rep(group, rep(T, N))) %>%
  ggplot() + geom_line(aes(t, val, colour = factor(var))) + facet_wrap(~group) + 
  theme(legend.position = 'none')

y %>%
  as.data.frame() %>%
  mutate(t = 1:T) %>%
  gather(var, val, -t) %>%
  mutate(var = rep(1:N, rep(T, N)),
         group = rep(group, rep(T, N))) %>%
  group_by(group) %>%
  summarise(var = var(val), mean = mean(val)) %>%
  select(-group) %>%
  unlist() -> true

lambda <- matrix(c(rnorm(K, -1), rnorm(K, 0), rlnorm(K*N, 0, 0.5)), ncol = 1)
fit <- fitVBGRP(y, lambda, K, S = 20, eta = 0.01)

fit$lambda[1:6]

kFit <- c(0, N)
for(i in 1:N){
  kFit[i] <- which.max(fit$lambda[2*K*(2*K+1) + (i-1)*K + 1:K])
}
caret::confusionMatrix(factor(kFit), factor(group))$table


}
