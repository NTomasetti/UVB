fitVB <- function(data, lambda, model, S = 25, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01, 
                  dimTheta = 6,  ...){
  dimLambda <- length(lambda)
  sobol <- sobol_points(100+S, dimTheta)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- numeric(dimLambda)
  V <- numeric(dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  while(diff > threshold){
    if(iter > maxIter){
      break
    }
    grad <- matrix(0, dimLambda, S)
    eval <- numeric(S)
    q <- numeric(S)
    unif <- shuffle(sobol)
    unif[unif < 0.001] = 0.001
    unif[unif > 0.999] = 0.999
    if(S == 1){
      logpj <- model(data, lambda, unif[101,], ...)
      eval <- logpj$val
      grad <- logpj$grad
      q <- sum(dnorm(epsilon, log=TRUE))
      gradient <- grad
      gradientSq <- grad^2
      LB[iter] <- eval - q
    } else {
      for(s in 1:S){
        #logpj <- model(data, lambda, unif[s+100,], ...)   
        logpj <- gradAR2(data, lambda, qnorm(unif[s+100,]), prior$mean, prior$Linv)
        eval[s] <- logpj$val
        grad[,s] <- logpj$grad
        q <- sum(dnorm(qnorm(unif[s+100,]), log = TRUE))
      }
      eval[eval == -Inf] = NA
      gradient <- rowMeans(grad, na.rm = TRUE)
      gradientSq <- rowMeans(grad^2, na.rm=TRUE)
      LB[iter] <- mean(eval - q, na.rm=TRUE) 
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
      print(paste0('Iteration: ', iter, ' ELBO: ', LB[iter]))
    }
    iter <- iter + 1
  }
  return(list(lambda=lambda, LB = LB[1:min(iter-1, maxIter)], iter = min(maxIter, iter-1)))
}

library(Rcpp)
library(RcppArmadillo)
library(RcppEigen)
library(rstan)
sourceCpp('AR2.cpp')

mu <- 2
phi1 <- 0.4
phi2 <- -0.6
sigmaSq <- 2

T <- 500

gamma0 <- sigmaSq * (1 - phi2) / ((1+phi2) * ((1-phi2)^2 - phi1^2))
gamma1 <- sigmaSq * phi1 / ((1+phi2) * ((1-phi2)^2 - phi1^2))
Sigma <- matrix(c(gamma0, gamma1, gamma1, gamma0), 2)
z <- mvtnorm::rmvnorm(1, c(mu, mu), Sigma)

for(t in 3:500){
  zt <- mu + phi1 * (z[t-1] - mu) + phi2 * (z[t-2] - mu) + rnorm(1, 0, sqrt(sigmaSq))
  z <- c(z, zt)
}

L <- t(chol(diag(10, 4)))
Linv <- solve(L)
prior <- list(mean = rep(0, 4), Linv = Linv)
lambda <- matrix(c(prior$mean, prior$Linv), ncol = 1)


fitOnline <- matrix(0, 20, 10)
fitOffline <- matrix(0, 20, 10)
for(s in 1:10){
  
  if(s == 1){
    fitOffline[,s] <- fitVB(z[1:(50*s)], lambda, gradAR2, 
                            mean = prior$mean, Linv = prior$Linv, dimTheta = 4, S = 25)$lambda
    
  } else {
    fitOffline[,s] <- fitVB(z[1:(50*s)], matrix(fitOffline[,s-1], ncol = 1), gradAR2, 
                            mean = prior$mean, Linv = prior$Linv, dimTheta = 4, S = 25)$lambda
  }
 
  if(s == 1){
    fitOnline[,s] = fitOffline[,s]
  } else {
    priorMean <- fitOnline[1:4, s-1]
    priorL <- matrix(fitOnline[5:20, s-1], 4, byrow = TRUE)
    priorLinv <- solve(priorL)
    fitOnline[,s] <- fitVB(z[50 * (s-1) + (-1):50], matrix(fitOnline[,s-1], ncol = 1), gradAR2,
                           mean = priorMean, Linv = priorLinv, dimTheta = 4, S = 25, conditional = TRUE)$lambda
  }
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

density <- NULL
transform <- c('exp', rep('identity', 3))
names <- c('sigma^{2}', 'mu', 'phi[1]', 'phi[2]')

for(s in 1:10){
  fit <- list(mean = fitOnline[1:4, s],
              U = fitOnline[5:20, s])
  dens <- vbDensity(fit, transform, names)
  dens$s <- s
  dens$method <- 'Online'
  density <- rbind(density, dens)
  
  fit <- list(mean = fitOffline[1:4, s],
              U = fitOffline[5:20, s])
  dens %>%
    group_by(var) %>%
    mutate(n = 1:n()) %>% 
    select(-density, -method, -s) %>%
    spread(var, support) %>% 
    select(-n) %>%
    as.list() -> supports
  dens <- vbDensity(fit, transform, names, supports)
  dens$s <- s
  dens$method <- 'Offline'
  density <- rbind(density, dens)
}
density %>%
  mutate(s = s * 50) %>%
  ggplot() + geom_line(aes(support, density, colour = method)) + 
  facet_grid(s ~ var, scales = 'free') + 
  theme_bw() + 
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank())

fitVBScore <- function(data, lambda, model, S = 50, maxIter = 5000, alpha = 0.01, beta1 = 0.9, beta2 = 0.99, threshold = 0.01, ...){
  dimLambda <- nrow(lambda)
  sobol <- sobol_points(100+6*S, 4)
  diff <- threshold + 1
  iter <- 1
  LB <- numeric(maxIter)
  M <- rep(0, dimLambda)
  V <- rep(0, dimLambda)
  e <- 1e-8
  meanLB <- 0
  oldMeanLB <- 0
  first <- TRUE
  while(diff > threshold | iter < 100){
    if(iter > maxIter){
      break
    }
    grad <- matrix(0, dimLambda, S)
    eval <- numeric(S)
    z <- lambda[33:36]
    pi <- exp(z) / sum(exp(z))
    unif <- shuffle(sobol)
    s <- 0
    try <- 0
    epsilon <- qnorm(unif[101:(100+6*S), ])
    epsilon[epsilon < -3] = -3
    epsilon[epsilon > 3] = 3
    while(s < S){
      k <- sample(1:4, 1, prob=pi)
      Qmean <- lambda[(k-1)*4 + 1:4]
      Qsd <- exp(lambda[16 + (k-1)*4 + 1:4])
      theta <- c(Qmean + Qsd * epsilon[try+1,])
      derivs <- model(data, lambda, theta, ...)
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


L <- t(chol(diag(10, 4)))
Linv <- solve(L)
prior <- list(mean = rep(0, 4), Linv = Linv)
lambda <- matrix(c(rnorm(16, 0, 0.1), rnorm(20, -3, 0.2)), ncol = 1)


fitOnline <- matrix(0, 36, 10)
fitOffline <- matrix(0, 36, 10)
for(s in 1:5){
  fitOffline[,s] <- fitVBScore(z[1:(50*s)], lambda, singlePriorMixApprox,
                            mean = prior$mean, Linv = prior$Linv, mix = 4, S = 25)$lambda
 
  
  if(s == 1){
    fitOnline[,s] = fitOffline[,s]
  } else {
    priorMean <- matrix(fitOnline[1:16, s-1], 4)
    siginv <- array(0, dim = c(4, 4, 4))
    dets <- NULL
    for(k in 1:4){
      sd <- exp(fitOnline[16 + (k-1)*4 + 1:4, s-1])
      var <- diag(sd^2)
      siginv[,,k] <- solve(var)
      dets <- c(dets, 1 / prod(sd))
    }
    weights <- fitOnline[33:36, s-1]
    weights <- exp(weights) / sum(exp(weights))
    fitOnline[,s] <- fitVBScore(z[50 * (s-1) + (-1):50], lambda, mixPriorMixApprox,
                           mean = priorMean, SigInv = siginv, dets = dets, weights = weights)$lambda
  }
}

densityMix <- NULL
names <- c('sigma^{2}', 'mu', 'phi[1]', 'phi[2]')
support <- list(seq(1.2, 2.5, length.out = 1000),
                seq(-1, 3, length.out = 1000), 
                seq(-0.2, 1, length.out = 1000),
                seq(-1, 0.2, length.out = 1000))

for(s in 1:5){
  zi <- fitOffline[33:36,s]
  pi <- exp(zi) / sum(exp(zi))
  densMat <- matrix(0, 1000, 4)
  for(k in 1:4){
    mean <- fitOffline[(k-1)*4 + 1:4, s]
    sd <- exp(fitOffline[16 + (k-1)*4 + 1:4, s])
    densMat[,1] <- densMat[,1] + pi[k] * dlnorm(support[[1]], mean[1], sd[1])
    for(j in 2:4){
      densMat[,j] <- densMat[,j] + pi[k] * dnorm(support[[j]], mean[j], sd[j])
    }
  }
  dens <- data.frame(support = unlist(support),
                     density = c(densMat),
                     method = 'Offline',
                     s = s,
                     variable = rep(names, rep(1000, 4)))
  densityMix <- rbind(densityMix, dens)
 
  
  zi <- fitOnline[33:36,s]
  pi <- exp(zi) / sum(exp(zi))
  densMat <- matrix(0, 1000, 4)
  for(k in 1:4){
    mean <- fitOnline[(k-1)*4 + 1:4, s]
    sd <- exp(fitOnline[16 + (k-1)*4 + 1:4, s])
    densMat[,1] <- densMat[,1] + pi[k] * dlnorm(support[[1]], mean[1], sd[1])
    for(j in 2:4){
      densMat[,j] <- densMat[,j] + pi[k] * dnorm(support[[j]], mean[j], sd[j])
    }
  }
  dens <- data.frame(support = unlist(support),
                     density = c(densMat),
                     method = 'Online',
                     s = s,
                     variable = rep(names, rep(1000, 4)))
  densityMix <- rbind(densityMix, dens)
  
}
densityMix %>%
  filter(s <= 5) %>%
  mutate(s = s * 50) %>%
  ggplot() + geom_line(aes(support, density, colour = method)) + 
  facet_grid(s ~ variable, scales = 'free', labeller = label_parsed) + 
  theme_bw() + 
  theme(axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank())




