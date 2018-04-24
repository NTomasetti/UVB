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

N <- 10
T <- 100
initialBatch <- 50
updateSize <- 25
batches <- (T - initialBatch) / updateSize + 1
data <- seq(initialBatch, T, length.out = batches)
MCsamples <- 5000

mu <- rnorm(2, 0, 0.25)
sigmaSq <- runif(2, 1, 2)
group <- sample(0:1, N, replace = TRUE)
y <- matrix(0, T, N)
results <- data.frame()

for(i in 1:N){
  y[,i] <- rnorm(T, mu[group[i]+1], sqrt(sigmaSq[group[i]+1]))
}

priorMean <- rep(0, 4)
priorVar <- diag(10, 4)
priorLinv <- solve(t(chol(priorVar)))
priorVarInv <- solve(priorVar)
priorPi <- rep(1, 2)


# MCMC
for(t in 1:batches){
  # Fit MCMC
  MCMCfit <- MixNormMCMC(y = y[1:data[t], ],
                         reps = 15000,
                         drawT = c(0.5, 0.5, 0, 0),
                         drawK = sample(0:1, N, replace = TRUE),
                         hyper = list(mean = priorMean, varInv = priorVarInv),
                         thin = 10,
                         stepsize = 0.01)
  keep <- floor(seq(1001, 1500, length.out = MCsamples))
  
  # Set up forecast density
  class <- rep(0, N)
  # Forecast
  for(j in 1:N){
    class1 <- 0
    for(i in 1:MCsamples){
      class1 <- class1 + MCMCfit$K[keep[i], j] / MCsamples
    }
    class[j] <- class1 > 0.5
  }
  score <- max(sum(class == group), sum(class != group)) / N
  
  results <- rbind(results,
                   data.frame(t = data[t],
                              score = score,
                              inference = 'MCMC',
                              K = 1:3,
                              runTime = NA,
                              ELBO = NA,
                              id = rep))
}

for(mix in 1:3){
  for(Update in 0:1){
    # Start Time
    VBTotal <- 0
    ISTotal <- 0
    
    
    # Initial Lambda
    lambda <- c(rnorm(4*mix, 0, 0.1), rnorm(4*mix, -1, 0.5), rnorm(mix))
    
    VBfit <- matrix(0, length(lambda), batches)
    
    for(t in 1:batches){
      startVB <- Sys.time()
      
      if(!Update | t == 1){
        # VB Fit = UVB fit when t = 1
        VB <- fitVBScoreMN(data = y[1:data[t],],
                           S = 50,
                           lambda = lambda,
                           mixPrior = 1,
                           mixApprox = mix,
                           mean = matrix(priorMean, ncol = 1),
                           SigInv = array(priorVarInv, dim = c(4, 4, 1)),
                           kPrior = rep(0.5, N),
                           weights = 1,
                           dets = det(priorVarInv) ^ 0.5)
        VBfit[,t] <- VB$lambda
        ELBO <- VB$LB[VB$iter - 1]
      } else {
        # UVB fits
        
        updateMean <- matrix(VBfit[1:(4*mix), t-1], ncol = mix)
        updateVarInv <- array(0, dim = c(4, 4, mix))
        dets <- rep(0, mix)
        for(k in 1:mix){
          sd <- exp(VBfit[4*mix + 4*(k-1) + 1:4, t-1])
          updateVarInv[,,k] <- diag(1/sd^2)
          dets[k] <- 1 / prod(sd)
        }
        updateZ <- VBfit[4*mix*2 + 1:mix, t-1] 
        updateWeight <- exp(updateZ) / sum(exp(updateZ))
        
        VB <- fitVBScoreMN(data = y[(data[t-1]+1):data[t],],
                           S = 50,
                           lambda = VBfit[,t-1],
                           mixPrior = mix,
                           mixApprox = mix,
                           mean = updateMean,
                           SigInv = updateVarInv,
                           piPrior = priorPi,
                           kPrior = probKgroup1,
                           weights = updateWeight,
                           dets = dets)
        
        VBfit[,t] <- VB$lambda
        ELBO <- VB$LB[VB$iter - 1]
        
      }
      
      endVB <- Sys.time() - startVB
      VBTotal <- VBTotal + as.numeric(endVB)
      
      drawVB <- matrix(0, MCsamples, 4)
      probK <- matrix(0, MCsamples, N)
      component <- rep(0, MCsamples)
      
      fitMean <- matrix(VBfit[1:(4*mix), t], ncol = mix)
      fitSd <- matrix(exp(VBfit[4*mix + 1:(4*mix), t]), ncol = mix)
      fitZ <- VBfit[8*mix + 1:mix, t]
      fitWeight <- exp(fitZ) / sum(exp(fitZ))
      
      for(i in 1:MCsamples){
        component[i] <- sample(1:mix, 1, prob = fitWeight)
        drawVB[i, ] <- fitMean[,component[i]] + fitSd[,component[i]] * rnorm(4)
        for(j in 1:N){
          probK[i, j] <- probK2(y[1:data[t],j], drawVB[i,], 0.5)
        }
      }
      
      ISTimeStart <- Sys.time()
      pTheta <- rep(0, MCsamples)
      qTheta <- rep(0, MCsamples)
      for(i in 1:MCsamples){
        pTheta[i] <- MNLogDens2(y[1:data[t], ], drawVB[i, ], probK[i, ], priorMean, priorVarInv)
        qTheta[i] <- mvtnorm::dmvnorm(drawVB[i,], fitMean[,component[i]], diag(fitSd[,component[i]]^2)) 
      }
      pTheta <- pTheta - max(pTheta)
      weights <- exp(pTheta) / qTheta
      weights <- weights / sum(weights)
      ISTimeFinish <- Sys.time() - ISTimeStart
      ISTotal <- ISTotal + as.numeric(ISTimeFinish)
      
      
      probKgroup1 <- colMeans(probK)
      probKgroup2 <- colSums(probK * weights)
      
      class1 <- probKgroup1 > 0.5
      class2 <- probKgroup2 > 0.5
      score1 <- max(sum(class1 == group), sum(class1 != group)) / N
      score2 <- max(sum(class2 == group), sum(class2 != group)) / N
      
      results <- rbind(results,
                       data.frame(t = data[t],
                                  score = c(score1, score2),
                                  inference = paste0(ifelse(Update, 'U', ''), c('VB', 'VB-IS')),
                                  K = mix,
                                  runTime = c(VBTotal, VBTotal + ISTotal), 
                                  ELBO = ELBO,
                                  id = rep))
    }
  }
}
write.csv(results, paste0('mixNorm/N50_', rep, '.csv'), row.names = FALSE)
