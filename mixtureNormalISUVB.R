rm(list = ls())
repenv <- Sys.getenv("SLURM_ARRAY_TASK_ID")
rep <- as.numeric(repenv)

library(Rcpp, lib.loc = 'packages')
library(RcppArmadillo, lib.loc = 'packages')
library(RcppEigen, lib.loc = 'packages')
library(rstan, lib.loc = 'packages')
source('RFuns.R')
sourceCpp('mixtureNormal.cpp')

set.seed(rep)

N <- 100
T <- 100
initialBatch <- 10
updateSize <- 10
batches <- (T - initialBatch) / updateSize + 1
data <- seq(initialBatch, T, length.out = batches)
MCsamples <- 500

mu <- rnorm(2, 0, 0.25)
sigmaSq <- runif(2, 1, 2)
group <- sample(0:1, N, replace = TRUE)
y <- matrix(0, T, N)
results <- list()
counter <- 1

for(i in 1:N){
  y[,i] <- rnorm(T, mu[group[i]+1], sqrt(sigmaSq[group[i]+1]))
}

priorMean <- rep(0, 4)
priorVar <- diag(10, 4)
priorLinv <- solve(t(chol(priorVar)))
priorVarInv <- solve(priorVar)

for(K in 1:3){
  # Start Time
  VBTotal <- 0
  
    
  # Initial Lambda
  lambda <- c(rnorm(4*K, 0, 0.1), rnorm(4*K, -1, 0.5), rnorm(K))
    
  VBfit <- matrix(0, length(lambda), batches)
    
  for(t in 1:batches){
    start <- Sys.time()
    
    if(t == 1){
      # VB Fit = UVB fit when t = 1
      VB <- fitVBScoreMN(data = y[1:data[t],],
                         S = 25,
                         lambda = lambda,
                         mixPrior = 1,
                         mixApprox = K,
                         mean = matrix(priorMean, ncol = 1),
                         SigInv = array(priorVarInv, dim = c(4, 4, 1)),
                         probK = rep(0.5, N),
                         weights = 1,
                         dets = det(priorVarInv) ^ 0.5)
      VBfit[,t] <- VB$lambda
      ELBO <- VB$LB[VB$iter - 1]
    } else {
      # Calcualte density under the old distribution
      qVB <- rep(0, MCsamples)
      for(k in 1:K){
        qVB <- qVB + fitWeight[k] * mvtnorm::dmvnorm(draw, fitMean[,k], diag(fitSd[,k]^2))
      }
      # Calculate the log-joint distribution through the UVB prior recursion
      # log liklihood + log(p(k)) was calculated already for each group, subtract log(p(k)) and reconstruct with weights
      loglik <- rep(0, MCsamples)
      for(j in 1:N){
        loglik <- loglik + log((1 - norm1) * exp(probK[, 1, j]) + norm1 * exp(probK[, 2, j])) - log(0.5)
      }
      loglik <- loglik - max(loglik)
      # Prior
      prior <- rep(0, MCsamples)
      for(k in 1:K){
        prior <- prior + fitWeight[k] * mvtnorm::dmvnorm(draw, fitMean[,k], diag(fitSd[,k]^2))
      }
      # normalise a bit
      logjoint <- loglik + log(prior)
  
      # Run VB Update
      VB <- ISUVB(lambda = VBfit[,t-1],
                  qScore = mixNormScore,
                  samples = draw,
                  dSamples = qVB,
                  logjoint = logjoint,
                  maxIter = 2000,
                  mix = K)
      
      VBfit[,t] <- VB$lambda
      ELBO <- VB$LB[VB$iter - 1]
      
    }
      
    endVB <- Sys.time() - start
    if(attr(endVB, 'units') == 'mins'){
      endVB <- endVB * 60
    }
    VBTotal <- VBTotal + endVB
      
    draw <- matrix(0, MCsamples, 4)
    probK <- array(0, dim = c(MCsamples, 2, N))
    component <- rep(0, MCsamples)
      
    fitMean <- matrix(VBfit[1:(4*K), t], ncol = K)
    fitSd <- matrix(exp(VBfit[4*K + 1:(4*K), t]), ncol = K)
    fitZ <- VBfit[8*K + 1:K, t]
    fitWeight <- exp(fitZ) / sum(exp(fitZ))
      
    for(i in 1:MCsamples){
      component[i] <- sample(1:K, 1, prob = fitWeight)
      draw[i, ] <- fitMean[,component[i]] + fitSd[,component[i]] * rnorm(4)
      for(j in 1:N){
        probK[i, , j] <- probK2(y[1:data[t],j], draw[i,], c(log(0.5), log(0.5)))
      }
    }
      
    probKgroup1 <- apply(probK, 2:3, mean)
    norm1 <- apply(probKgroup1, 2, function(x) {y = x - max(x); exp(y[2]) / sum(exp(y))})
      
    class1 <- norm1 > 0.5
    score1 <- max(sum(class1 == group), sum(class1 != group)) / N
      
    results[[counter]] <-  data.frame(t = data[t],
                                      score = score1,
                                      inference = 'IS-UVB',
                                      K = K,
                                      runTime = VBTotal,
                                      ELBO = ELBO,
                                      id = rep)
    counter <- counter + 1
  }
}

results <- do.call(rbind.data.frame, results)

ggplot(results) + geom_line(aes(t, runTime, colour = factor(K)))

write.csv(results, paste0('mixNorm/N50_', rep, '_IS.csv'), row.names = FALSE)