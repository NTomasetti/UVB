library(tidyverse)
library(VBfuns)
Rcpp::sourceCpp('erraticDetector.cpp')

carsAug <- readRDS('../Cars/OHF/carsAug.RDS')

N <- 200
maxT <- 600

#' Find cars that have:
#' A) Been in the sample for at least T observations
#' B) Not stopped inside the first T observations
#' C) Did not take any huge turns associated with bad measurements at low velocity
#' 2154 Vehicles remain

carsAug %>%
  group_by(ID) %>% 
  mutate(n = seq_along(time)) %>%
  filter(max(n) >= maxT & n <= maxT) %>%
  filter(min(relV) > 0 & max(abs(relD - pi/2)) < 1) -> carsNoStop


set.seed(1)
subID <- sample(unique(carsNoStop$ID), N) 

carsNoStop %>%
  filter(ID %in% subID) %>%
  select(-x, -y) %>%
  rename(v = relV, a = relA, x = relX, y = relY, d = relD, o = relO) %>%
  select(ID, v, d, a, o, n, x, y, lane) %>%
  ungroup() -> carSubset

# Currently the changed column in carsAug looks for lane changes in the entire dataset. Redo it so it only finds lane changes in the T observatiosn

carSubset %>%
  group_by(ID) %>%
  mutate(startLane = head(lane, 1),
         changed = any(lane != startLane)) %>%
  ungroup() -> carSubset

carArray <- array(0, dim = c(maxT, 2, N))
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

fitWhole <- reparamVB(data = carArray,
                 lambda = lambda,
                 model = erraticGrad,
                 S = 10,
                 dimTheta = 2 * N,
                 maxIter = 2000,
                 threshold = 0.01 * N,
                 priorMean = priorMean,
                 priorLinv = priorLinv)

# UVB Loop

set.seed(5)
Tseq <- c(0, 100, seq(110, maxT, 10))

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
                   priorLinv = priorLinv,
                   suppressProgress = TRUE)

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
  filter(T == maxT) %>%
  select(meana, meano, sda, sdo, rho) %>%
  GGally::ggpairs()

params %>%
  filter(T == maxT) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2)) %>%
  .$meano %>%
  quantile(c(0.01, 0.5, 0.8, 0.99, 1)) -> varO

params %>%
  filter(T == maxT) %>%
  mutate(meano = exp(meano + 0.5 * sdo^2),
         meana = exp(meana + 0.5 * sda^2)) %>%
  .$meana %>%
  quantile(c(0.01, 0.5, 0.8, 0.99, 1)) -> varA


params %>%
  filter(T == maxT) %>%
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
  select(ID, OQ, AQ) -> quantiles

carSubset %>%
  left_join(quantiles) %>%
  ggplot() + geom_path(aes(n, x, group = ID)) + facet_wrap(~OQ)


support <- seq(1e-8, 0.01, length.out = 1000)

autoSupportDlnorm <- function(logmean, logsd, length = 1000){
  mean <- exp(logmean + 0.5 * logsd^2)
  sd <- sqrt((exp(logsd^2) - 1) * exp(2 * logmean + logsd^2))
  support <- seq(max(1e-12, mean - 5 * sd), mean + 5 * sd, length.out = 1000)
  dens <- dlnorm(support, logmean, logsd)
  data.frame(support = support,
             density = dens)
}

set.seed(15)
quantiles %>%
  filter(OQ >= 4) %>% 
  .$ID %>% 
  sample(9) -> sample

params %>% 
  left_join(quantiles) %>%
  filter(ID %in% sample) %>%
  group_by(ID, T) %>%
  do(autoSupportDlnorm(.$meano, .$sdo)) -> posteriorSequence


density <- NULL
for(i in 1:N){
  if(!subID[i] %in% sample){
    next
  } 
  mu <- fit$lambda[(i-1)*6 + 1:2]
  U <- matrix(fit$lambda[(i-1)*6 + 3:6], 2)
  Sigma <- t(U) %*% U
  sd <- sqrt(diag(Sigma))
  dens <- gaussianDensity(mu = mu,
                          sd = sd, 
                          transform = c('exp', 'exp'),
                          names = c('sigma[a]^{2}', 'sigma[omega]^{2}'))
  dens$ID <- subID[i]
  density <- rbind(density, filter(dens, var == 'sigma[omega]^{2}'))
}

ggplot(posteriorSequence) + geom_line(aes(support, density, colour = T, group = T, alpha = T)) + 
  geom_line(data = density, aes(support, density)) +
  facet_wrap(~ID, scales = 'free') +
  theme_bw() + 
  scale_colour_gradient(high = '#743336', low = '#FF1C1C') + 
  scale_alpha(guide = 'none', range = c(0.1, 0.4)) + 
  labs(y = NULL, x = NULL) + 
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        strip.background = element_blank(),
        strip.text.x = element_blank())







carSubset %>%
  right_join(params %>%
               filter(T == maxT)) %>%
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
  summarise(ch = sum(changed) / (61 * maxT)) %>%
  spread(AQ, ch, fill = 0) %>%
  rbind(data.frame(OQ = 'Total',
                   Q1 = sum(.[,2]),
                   Q2 = sum(.[,3]),
                   Q3 = sum(.[,4]),
                   Q4 = sum(.[,5]),
                   Q5 = sum(.[,6]))) %>%
  mutate(OQMarginal = Q1 + Q2 + Q3 + Q4 + Q5)
  


carSubset %>%
  right_join(params %>%
               filter(T == maxT)) %>%
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








