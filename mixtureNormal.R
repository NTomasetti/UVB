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
MCsamples <- 2000
ISsamples <- c(100, 500, 1000)
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

# MCMC
for(t in 1:batches){
  # Fit MCMC
  MCMCfit <- MixNormMCMC(y = y[1:data[t], ],
                         reps = 15000,
                         drawT = c(0.5, 0.5, 0, 0),
                         drawK = sample(0:1, N, replace = TRUE),
                         hyper = list(mean = priorMean, varInv = priorVarInv),
                         thin = 1,
                         stepsize = 0.01)
  keep <- floor(seq(10001, 15000, length.out = MCsamples))
  
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
  
  results[[counter]] <- data.frame(t = data[t],
                              score = score,
                              inference = 'MCMC',
                              K = 1:3,
                              runTime = NA,
                              ELBO = NA,
                              id = rep)
  counter <- counter + 1
}

for(K in 1:3){
  # Start Time
  VBTotal <- 0
  UVBTotal <- 0 
  ISTotal <- rep(0, 3)
    
  # Initial Lambda
  lambda <- c(rnorm(4*K, 0, 0.1), rnorm(4*K, -1, 0.5), rnorm(K, 0, 0.25))
    
  VBfit <- UVBfit <- matrix(0, length(lambda), batches)
  ISfit <- array(0, dim = c(length(lambda), batches, 3))
    
  for(t in 1:batches){

    
    if(t == 1){
      startVB <- Sys.time()
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
      endVB <- Sys.time() - startVB
      VBfit[,t] <- UVBfit[,t] <- ISfit[,t, 1] <- ISfit[,t, 2] <- ISfit[,t, 3] <- VB$lambda
      ELBO <- rep(VB$LB[VB$iter - 1], 5)
    } else {
      # VB
      startVB <- Sys.time()
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
      endVB <- Sys.time() - startVB
      VBfit[,t] <- VB$lambda
      ELBO <- VB$LB[VB$iter - 1]
      
      # UVB
      startUVB <- Sys.time()
      updateMean <- matrix(UVBfit[1:(4*K), t-1], ncol = K)
      updateSd <- matrix(exp(UVBfit[4*K + 1:(4*K), t-1]), ncol = K)
      updateVarInv <- array(0, dim = c(4, 4, K))
      dets <- rep(0, K)
      for(k in 1:K){
        updateVarInv[,,k] <- diag(1/updateSd[,k]^2)
        dets[k] <- 1 / prod(updateSd[,k])
      }
      updateZ <- UVBfit[8*K + 1:K, t-1]
      updateWeight <- exp(updateZ) / sum(exp(updateZ))
      
      UVB <- fitVBScoreMN(data = y[(data[t-1]+1):data[t],],
                         S = 25,
                         lambda = UVBfit[,t-1],
                         mixPrior = K,
                         mixApprox = K,
                         mean = updateMean,
                         SigInv = updateVarInv,
                         probK = probKUVB,
                         weights = updateWeight,
                         dets = dets)
      endUVB <- Sys.time() - startUVB
      UVBfit[,t] <- UVB$lambda
      ELBO <- c(ELBO, UVB$LB[UVB$iter - 1])
      
      # IS-UVB
      endIS <- rep(0, 3)
      for(isrep in 1:3){
        startIS <- Sys.time()
        
        ISMean <- matrix(ISfit[1:(4*K), t-1, isrep], ncol = K)
        ISSd <- matrix(exp(ISfit[4*K + 1:(4*K), t-1, isrep]), ncol = K)
        ISZ <- ISfit[2*4*K + 1:K, t-1, isrep]
        ISWeight <- exp(ISZ) / sum(exp(ISZ))
        ISdraw <- matrix(0, ISsamples[isrep], 4)
        for(i in 1:ISsamples[isrep]){
          groupIS <- sample(1:K, 1, prob = ISWeight)
          ISdraw[i, ] <- rnorm(4, ISMean[,groupIS], ISSd[,groupIS])
        }
        # Calcualte density under the old distribution
        qIS <- rep(0, ISsamples[isrep])
        for(k in 1:K){
          qIS <- qIS + ISWeight[k] * mvtnorm::dmvnorm(ISdraw, ISMean[,k], diag(ISSd[,k]^2))
        }
        # Calculate the log-joint distribution through the UVB prior recursion
        # log liklihood + log(p(k)) was calculated already for each group, subtract log(p(k)) and reconstruct with weights
        loglik <- rep(0, ISsamples[isrep])
        for(j in 1:N){
          ll <- matrix(0, ISsamples[isrep], 2)
          for(i in 1:ISsamples[isrep]){
            ll[i, ] <- probK2(y[(data[t-1]+1):data[t],j], ISdraw[i,], c(0, 0))
          }
          loglik <- loglik + log((1 - probKIS[j, isrep]) * exp(ll[,1]) +
                                   probKIS[j, isrep] * exp(ll[,2]))
        }
        loglik <- loglik - max(loglik)
        # Prior is qIS
        logjoint <- loglik + log(qIS)  
        # Run VB Update
        IS <- ISUVB(lambda = ISfit[,t-1, isrep],
                    qScore = mixNormScore,
                    samples = ISdraw,
                    dSamples = qIS,
                    logjoint = logjoint,
                    maxIter = 2000,
                    mix = K)
        istime <- Sys.time() - startIS
        if(attr(istime, 'units') == 'mins'){
          istime <- istime * 60
        }
        endIS[isrep] <- istime
        ISfit[,t, isrep] <- IS$lambda
        ELBO <- c(ELBO, IS$LB[IS$iter - 1])
      }
    
    }
      
    if(attr(endVB, 'units') == 'mins'){
      endVB <- endVB * 60
    }
    if(t == 1){
      endUVB <- endVB
      endIS <- rep(endVB, 3)
    } else {
      if(attr(endUVB, 'units') == 'mins'){
        endUVB <- endUVB * 60
      }
    }
   
    VBTotal <- VBTotal + as.numeric(endVB)
    UVBTotal <- UVBTotal + as.numeric(endUVB)
    ISTotal <- ISTotal + endIS
    
    # Draw samples
    VBMean <- matrix(VBfit[1:(4*K), t], ncol = K)
    VBSd <- matrix(exp(VBfit[4*K + 1:(4*K), t]), ncol = K)
    VBZ <- VBfit[2*4*K + 1:K, t]
    VBWeight <- exp(VBZ) / sum(exp(VBZ))
    VBdraw <- matrix(0, MCsamples, 4)
    for(i in 1:MCsamples){
      groupVB <- sample(1:K, 1, prob = VBWeight)
      VBdraw[i, ] <- mvtnorm::rmvnorm(1, VBMean[,groupVB], diag(VBSd[,groupVB]^2))
    }
    
    if(t > 1){
      UVBMean <- matrix(UVBfit[1:(4*K), t], ncol = K)
      UVBSd <- matrix(exp(UVBfit[4*K + 1:(4*K), t]), ncol = K)
      UVBZ <- UVBfit[2*4*K + 1:K, t]
      UVBWeight <- exp(UVBZ) / sum(exp(UVBZ))
      UVBdraw <- matrix(0, MCsamples, 4)
      for(i in 1:MCsamples){
        groupUVB <- sample(1:K, 1, prob = UVBWeight)
        UVBdraw[i, ] <- mvtnorm::rmvnorm(1, UVBMean[,groupUVB], diag(UVBSd[,groupUVB]^2))
      }
      
      ISdraw <- array(0, dim = c(MCsamples, 4, 3))
      for(isrep in 1:3){
        ISMean <- matrix(ISfit[1:(4*K), t, isrep], ncol = K)
        ISSd <- matrix(exp(ISfit[4*K + 1:(4*K), t, isrep]), ncol = K)
        ISZ <- ISfit[2*4*K + 1:K, t, isrep]
        ISWeight <- exp(ISZ) / sum(exp(ISZ))
        for(i in 1:MCsamples){
          groupIS <- sample(1:K, 1, prob = ISWeight)
          ISdraw[i, , isrep] <- mvtnorm::rmvnorm(1, ISMean[,groupIS], diag(ISSd[,groupIS]^2))
        }
      }
    }
    
    # Calculate p(k = 1)
    
    probKVB <- probKUVB <- array(0, dim = c(MCsamples, 2, N))
    probKIS <- array(0, dim = c(MCsamples, 2, N, 3))
    for(i in 1:MCsamples){
      for(j in 1:N){
        probKVB[i, , j] <- probK2(y[1:data[t],j], VBdraw[i,], c(log(0.5), log(0.5)))
        if(t > 1){
          probKUVB[i, , j] <- probK2(y[1:data[t],j], UVBdraw[i,], c(log(0.5), log(0.5)))
          for(isrep in 1:3){
            probKIS[i, , j, isrep] <- probK2(y[1:data[t],j], ISdraw[i,,isrep], c(log(0.5), log(0.5)))
          }
        }
      }
    }
    if(t == 1){
      probKUVB <- probKVB
      for(isrep in 1:3){
        probKIS[,,,isrep] <- probKVB
      }
    }
    
    probKVB <- apply(probKVB, 2:3, mean)
    probKUVB <- apply(probKUVB, 2:3, mean)
    probKIS <- apply(probKIS, 2:4, mean)
    
    probKVB <- apply(probKVB, 2, function(x) {y = x - max(x); exp(y[2]) / sum(exp(y))})
    probKUVB <- apply(probKUVB, 2, function(x) {y = x - max(x); exp(y[2]) / sum(exp(y))})
    probKIS <- apply(probKIS, 2:3, function(x) {y = x - max(x); exp(y[2]) / sum(exp(y))})
 
    classVB <- probKVB > 0.5
    classUVB <- probKUVB > 0.5
    classIS <- apply(probKIS, 2, function(x) x > 0.5)
    scoreVB <- max(sum(classVB == group), sum(classVB != group)) / N
    scoreUVB <- max(sum(classUVB == group), sum(classUVB != group)) / N
    scoreIS <- apply(classIS, 2, function(x) max(sum(x == group), sum(x != group)) / N)
    
      
    results[[counter]] <-  data.frame(t = data[t],
                                      score = c(scoreVB, scoreUVB, scoreIS),
                                      inference = c('VB', 'UVB', paste0('IS-UVB-', ISsamples)),
                                      K = K,
                                      runTime = c(VBTotal, UVBTotal, ISTotal), 
                                      ELBO = ELBO,
                                      id = rep)
    counter <- counter + 1
  }
}

results <- do.call(rbind.data.frame, results)
write.csv(results, paste0('mixNorm/N100_', rep, '.csv'), row.names = FALSE)
