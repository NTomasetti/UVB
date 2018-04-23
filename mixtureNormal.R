library(Rcpp)#, lib.loc = 'packages')
library(RcppArmadillo)#, lib.loc = 'packages')
library(RcppEigen)#, lib.loc = 'packages')
library(rstan)#, lib.loc = 'packages')
source('RFuns.R')
sourceCpp('mixtureNormal.cpp')

rep <- 1
set.seed(rep)

N <- 100
T <- 200
initialBatch <- 50
updateSize <- 25
batches <- (T - initialBatch) / updateSize + 1
data <- seq(initialBatch, T, length.out = batches)
MCsamples <- 500

mu <- rnorm(2, 0, 0.25)
sigmaSq <- runif(2, 1, 2)
group <- sample(1:2, N, replace = TRUE)
y <- matrix(0, T, N)
stats <- data.frame()
results <- data.frame()

for(i in 1:N){
  y[,i] <- rnorm(T, mu[group[i]], sqrt(sigmaSq[group[i]]))
}

priorMean <- rep(0, 4)
priorVar <- diag(10, 4)
priorLinv <- solve(t(chol(priorVar)))
priorPi <- matrix(1, N, 2)

draw <- c(0, 0, 0, 0, rep(1, 2*N))

# MCMC
for(t in seq_along(batches)){
  # Fit MCMC
  MCMCfit <- MixNormMCMC(data = y[1:data[t], ],
                         reps = 15000,
                         draw = draw,
                         hyper = list(mean = priorMean, varInv = priorVarInv))$draws[floor(seq(10001, 15000, length.out = MCsamples)), ]
  
  # Set up forecast density
  class <- rep(0, N)
  # Forecast
  for(j in 1:N){
    class1 <- 0
    class2 <- 0
    for(i in 1:MCsamples){
      class1 <- class1 + MCMCfit[i, 3 + 2*j] / MCsamples
      class2 <- class2 + MCMCfit[i, 4 + 2*j] / MCsamples
    }
    class[j] <- which.max(c(class1, class2))
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

for(K in 1:3){
  for(Update in 0:1){
    # Start Time
    startTimeVB <- Sys.time()
      
    # Initial Lambda
    if(K == 1){
      lambda <- c(rnorm(2), rnorm(2, -0.5, 0.2), diag(1, 4), rnorm(2*N, 0, 0.25))
    } else {
      lambda <- c(rnorm(2*K), rnorm(2*K, -1, 0.5), rnorm(K), rnorm(2*N, 0, 0.25))
    }
      
    VBfit <- matrix(0, length(lambda), batches)
      
    for(t in 1:batches){
      if(!Update){
      # VB Fits
      #If K = 1, use the reparam gradients
        if(K == 1){
          VB <- fitVB(data = x[(data[1]+1 - lags):data[t+1]],
                      lambda = lambda,
                      model = mixtureNormalSPSA,
                      S = 25,
                      dimTheta = 4+N,
                      mean = priorMean,
                      varInv = priorLinv,
                      piPrior = priorPi)
            
          VBfit[,t] <- VB$lambda
          ELBO <- VB$LB[VB$iter - 1]
          
        } else {
        # Otherwise use score gradients
          VB <- fitVBScoreMN(data = x[(data[1]+1 - lags):data[t+1]],
                             lambda = lambda,
                             model = mixtureNormalSPMA,
                             K = K,
                             S = 25,
                             mean = priorMean,
                             varInv = priorLinv,
                             piPrior = priorPi)
            
          VBfit[,t] <- VB$lambda
          ELBO <- VB$LB[VB$iter - 1]
            
        }
      } else {
        # UVB fits, At time 1, VB = UVB
        if(t == 1){
          if(K == 1){
            VB <- fitVB(data = x[(data[1]+1 - lags):data[t+1]],
                        lambda = lambda,
                        model = mixtureNormalSPSA,
                        S = 25,
                        dimTheta = 4+N,
                        mean = priorMean,
                        varInv = priorLinv,
                        piPrior = priorPi)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[VB$iter - 1]
            
          } else {
            # Otherwise use score gradients
            VB <- fitVBScoreMN(data = x[(data[1]+1 - lags):data[t+1]],
                               lambda = lambda,
                               model = mixtureNormalSPMA,
                               K = K,
                               S = 25,
                               mean = priorMean,
                               varInv = priorLinv,
                               piPrior = priorPi)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[VB$iter - 1]
            
          }
        } else {
          # Otherwise apply UVB
          if(K == 1){
            # Reparam gradients
            updateMean <- VBfit[1:4, t-1]
            updateU <- matrix(VBfit[5:20, t-1], ncol = 4)
            updateLinv <- t(solve(updateU))
            
            updatePi <- matrix(0, N, 2)
            updatePi[, 1] <- VBfit[20 + seq(1, 2*N-1, 2), t-1]
            updatePi[, 2] <- VBfit[20 + seq(2, 2*N, 2), t-1]^2
            
            VB <- fitVB(data = x[(data[1]+1 - lags):data[t+1]],
                        lambda = VBfit[,t-1],
                        model = mixtureNormalSPSA,
                        S = 25,
                        dimTheta = 4+N,
                        mean = updateMean,
                        varInv = updateLinv,
                        piPrior = updatePi)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[VB$iter - 1]
              
          } else {
            # Score gradients
            updateMean <- matrix(VBfit[1:(4*K), t-1], ncol = K)
            updateVarInv <- array(0, dim = c(dim, dim, K))
            dets <- rep(0, K)
            for(k in 1:K){
              sd <- exp(VBfit[4*K + 4*(k-1) + 1:4, t-1])
              updateVarInv[,,k] <- diag(1/sd^2)
              dets[k] <- 1 / prod(sd)
            }
            updateZ <- VBfit[4*K*2 + 1:K, t-1] 
            updateWeight <- exp(updateZ) / sum(exp(updateZ))
            
            updatePi <- VBfit[9*K + 1:(2*N), ncol = 2]
            
            VB<- fitVBScoreMN(data = x[(data[t]+1-lags):data[t+1]],
                              lambda = VBfit[,t-1],
                              model = mixtureNormalMPMA,
                              K = K,
                              S = 25,
                              mean = updateMean,
                              SigInv = updateVarInv,
                              dets = dets,
                              weights = updateWeight,
                              piPrior = updatePi)
            
            VBfit[,t] <- VB$lambda
            ELBO <- VB$LB[VB$iter - 1]
              
          }
            
        }
      }
        
    
      runTime <- Sys.time() - startTimeVB
      class(runTime) <- 'numeric'
      if(attr(runTime, 'units') == 'mins'){
        attr(runTime, 'units') = 'secs'
        runTime <- runTime * 60
      } else if(attr(runTime, 'units') == 'hours'){
        attr(runTime, 'units') = 'hours'
        runTime <- runTime * 3600
      }
      
      class <- rep(0, N)
      for(j in 1:N){
        class[j] <- which.max(VBfit[,t], VBfit[,t])
      }
      score <- max(sum(class == group), sum(class != group)) / N
      
      results <- rbind(results,
                       data.frame(t = data[t],
                                  score = score,
                                  inference = paste0(ifelse(Update, 'U', ''), 'VB'),
                                  K = K,
                                  runTime = runTIme,
                                  ELBO = ELBO,
                                  id = rep))
    }
  }
}

