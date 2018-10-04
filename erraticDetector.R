library(tidyverse)
library(VBfuns)
Rcpp::sourceCpp('erraticDetector.cpp')

carsAug <- readRDS('../Cars/OHF/carsAug.RDS')

N <- 500

set.seed(1)

carsAug %>%
  group_by(ID) %>%
  filter(y > 190) %>%
  summarise(n = n()) %>% 
  filter(n > 602) %>%
  left_join(carsAug) %>%
  filter(y > 190) %>%
  group_by(ID) %>%
  mutate(n = seq_along(time),
         o = relD - lag(relD)) %>%
  filter(min(relV) > 0) -> carsNoStop

subID <- sample(unique(carsNoStop$ID), N) 

carsNoStop %>%
  filter(ID %in% subID & n > 2 & n <= 602) %>%
  select(-x, -y) %>%
  rename(v = relV, a = relA, x = relX, y = relY, d = relD) %>%
  select(ID, v, d, a, o, n, x, y, lane) %>%
  ungroup() -> carSubset

carSubset %>%
  group_by(ID) %>%
  summarise(startLane = head(lane, 1)) %>%
  right_join(carSubset) %>%
  group_by(ID) %>%
  mutate(changed = any(lane != startLane)) %>%
  ungroup() -> carSubset

carArray <- array(0, dim = c(600, 2, N))
for(i in 1:N){
  carArray[,,i] <- carSubset %>%
    filter(ID == subID[i]) %>%
    select(a, o) %>%
    as.matrix()
}


priorMean <- matrix(0, 2, N)
priorLinv <- array(0, dim = c(2, 2, N))

lambda <- NULL
for(i in 1:N){
  lambda <- c(lambda, 0, 0, diag(0.1, 2))
  priorLinv[,,i] <- solve(t(chol(diag(10, 2))))
}

fit <- reparamVB(data = carArray,
                 lambda = lambda,
                 model = erraticGrad,
                 S = 10,
                 dimTheta = 2 * N,
                 maxIter = 2000,
                 threshold = 0.01 * N,
                 priorMean = priorMean,
                 priorLinv = priorLinv)

ggplot() + geom_line(aes(1:fit$iter, fit$LB))

density <- NULL
params <- NULL
for(i in 1:N){
  mu <- fit$lambda[(i-1)*6 + 1:2]
  U <- matrix(fit$lambda[(i-1)*6 + 3:6], 2)
  Sigma <- t(U) %*% U
  sd <- sqrt(diag(Sigma))
  dens <- gaussianDensity(mu = mu,
                          sd = sd, 
                          transform = c('exp', 'exp'),
                          names = c('sigma[a]^{2}', 'sigma[omega]^{2}'))
  dens$ID <- subID[i]
  density <- rbind(density, dens)
  
  rho <- cov2cor(Sigma)[1, 2]
  params <- rbind(params,
                data.frame(meana = mu[1],
                           meano = mu[2],
                           sda = sd[1],
                           sdo = sd[2],
                           rho = rho,
                           ID = subID[i]))
  
  
}

ggplot(density) + geom_line(aes(support, density, group = ID)) + facet_wrap(~var, scales = 'free', labeller = label_parsed)

# UVB Loop

set.seed(5)
Tseq <- c(0, 100, seq(110, 600, 10))

priorMean <- matrix(0, 2, N)
priorLinv <- array(0, dim = c(2, 2, N))

lambda <- NULL
for(i in 1:N){
  lambda <- c(lambda, 0, 0, diag(0.1, 2))
  priorLinv[,,i] <- solve(t(chol(diag(10, 2))))
}
params <- tibble()

for(t in 2:length(Tseq)){
  print(paste(t, Sys.time()))
  
  fit <- reparamVB(data = carArray[(Tseq[t-1]+1):Tseq[t], , ],
                   lambda = lambda,
                   model = erraticGrad,
                   S = 10,
                   dimTheta = 2 * N,
                   maxIter = 2000,
                   threshold = 0.01 * N,
                   priorMean = priorMean,
                   priorLinv = priorLinv)

  lambda <- fit$lambda
  priorMean <- matrix(0, 2, N)
  priorLinv <- array(0, dim = c(2, 2, N))
  for(i in 1:N){
    mu <- fit$lambda[(i-1)*6 + 1:2]
    U <- matrix(fit$lambda[(i-1)*6 + 3:6], 2)
    Sigma <- t(U) %*% U
    sd <- sqrt(diag(Sigma))
    rho <- cov2cor(Sigma)[1, 2]
  
    params <- rbind(params,
                    data.frame(meana = mu[1],
                               meano = mu[2],
                               sda = sd[1],
                               sdo = sd[2],
                               rho = rho,
                               ID = subID[i],
                               T = Tseq[t]))
      
    priorMean[,i] <- mu
    priorLinv[,,i] <- solve(t(chol(Sigma)))
  }
}

params %>%
  rename('mu[a]' = meana, 
         'mu[omega]' = meano,
         'sigma[a]' = sda,
         'sigma[omega]' = sdo) %>%
  gather(var, value, -T, -ID) %>%
  ggplot() + geom_line(aes(T, value, group = ID)) +
    facet_wrap(~var, scales = 'free', labeller = label_parsed) + 
    theme_bw()


params %>%
  filter(T == 600) %>%
  select(meana, meano, sda, sdo, rho) %>%
  GGally::ggpairs()

params %>%
  filter(T == 600) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2)) %>%
  gather(param, mean, meano, meana) %>%
  ggplot() + geom_density(aes(mean)) + facet_wrap(~param, scales = 'free')

params %>%
  filter(T == 600) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2)) %>%
  .$meano %>%
  quantile(seq(0.2, 1, 0.2)) -> varO

varO

params %>%
  filter(T == 600) %>%
  mutate(meana = exp(meana + 0.5 * sda^2)) %>%
  .$meana %>%
  quantile(seq(0.2, 1, 0.2)) -> varA

carSubset %>%
  right_join(params %>%
               filter(T == 600)) %>%
  #right_join(params) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2), 
         OQ = case_when(meano <= varO[1] ~ 1L,
                        meano <= varO[2] ~ 2L,
                        meano <= varO[3] ~ 3L,
                        meano <= varO[4] ~ 4L,
                        TRUE ~ 5L),
         AQ = case_when(meana <= varA[1] ~ 1L,
                        meana <= varA[2] ~ 2L,
                        meana <= varA[3] ~ 3L,
                        meana <= varA[4] ~ 4L,
                        TRUE ~ 5L)) %>%
  ggplot() + geom_path(aes(n, x, group = ID, colour = changed)) + facet_grid(AQ~OQ)




carSubset %>%
  right_join(params %>%
               filter(T == 600)) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2), 
         OQ = case_when(meano <= varO[1] ~ 'Q1',
                        meano <= varO[2] ~ 'Q2',
                        meano <= varO[3] ~ 'Q3',
                        meano <= varO[4] ~ 'Q4',
                        TRUE ~ 'Q5'),
         AQ = case_when(meana <= varA[1] ~ 'Q1',
                        meana <= varA[2] ~ 'Q2',
                        meana <= varA[3] ~ 'Q3',
                        meana <= varA[4] ~ 'Q4',
                        TRUE ~ 'Q5')) %>%
  group_by(AQ, OQ) %>%
  summarise(ch = sum(changed) / (57 * 600)) 
  spread(AQ, ch) %>%
  rbind(data.frame(OQ = 'Total',
                   Q1 = sum(.[,2]),
                   Q2 = sum(.[,3]),
                   Q3 = sum(.[,4]),
                   Q4 = sum(.[,5]),
                   Q5 = sum(.[,6]))) %>%
  mutate(OQMarginal = Q1 + Q2 + Q3 + Q4 + Q5)
  


carSubset %>%
  right_join(params %>%
               filter(T == 600)) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2),
         OQ = case_when(meano < quantile(meano, 0.1) ~ 'Low',
                        meano < quantile(meano, 0.9) ~ 'Normal',
                        TRUE ~ 'High'),
         AQ = case_when(meana < quantile(meana, 0.1) ~ 'Low',
                        meana < quantile(meana, 0.9) ~ 'Normal',
                        TRUE ~ 'High')) %>%
  filter(OQ != 'Normal' & AQ != 'Normal') %>%
  ggplot() + geom_path(aes(y, x, group = ID)) + facet_grid(AQ~OQ)








