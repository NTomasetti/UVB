updatePrior <- function(fit, var, switch, K, dim){
  # Calculate updated priors for theta parameters
  if(var){
    updateMean <- updateSd <- array(0, dim = c(dim, 48, K + switch))
    updateLinv <- array(0, dim = c(48, 48, K * (dim + switch)))
    updateL <- array(0, dim = c(4, 2, K))
    if(switch){
      updateMean[1:2, ki, 1] <- fit[(ki-1)*5 + 1:2]
      updateMean[3:4, ki, 1] <- fit[5*K + (ki-1)*5 + 1:2]
      updateL[1:2, 1:2, ki] <- matrix(c(fit[(ki-1)*5 + 3:4], 0, fit[(ki-1)*5 + 5]), 2)
      updateL[3:4, 1:2, ki] <- matrix(c(fit[K * 5 + (ki-1)*5 + 3:4], 0, fit[K * 5 + (ki-1)*5 + 5]), 2) 
      updateLinv[1:2, 1:2, ki] <- solve(updateL[1:2, 1:2, ki])
      updateLinv[3:4, 1:2, ki] <- solve(updateL[3:4, 1:2, ki])
    } else {
      for(ki in 1:K){
        updateMean[,,ki] <- matrix(fit[48*dim*2*(ki-1) + 1:(48*dim)], ncol = 48, byrow = TRUE)
        updateSd[,,ki] <- exp(matrix(fit[48*dim*(2*(ki-1)+1) + 1:(48*dim)], ncol = 48, byrow = TRUE))
        for(i in 1:dim){
          updateLinv[,,(ki-1)*dim + i] <- diag(1 / updateSd[i,,ki])
        }
      }
    }
    return(list(updateMean = updateMean,
                updateSd = updateSd,
                updateL = updateL,
                updateLinv = updateLinv))
    
  } else {
    updateMean <- array(0, dim = c(dim, K, 1 + switch))
    updateL <- updateLinv <- array(0, dim = c(dim, dim, K + K * switch))
    if(switch){
      for(ki in 1:K){
        updateMean[1:2, ki, 1] <- fit[(ki-1)*5 + 1:2]
        updateMean[3:4, ki, 1] <- fit[5*K + (ki-1)*5 + 1:2]
        updateL[1:2, 1:2, ki] <- matrix(c(fit[(ki-1)*5 + 3:4], 0, fit[(ki-1)*5 + 5]), 2)
        updateL[3:4, 1:2, ki] <- matrix(c(fit[K * 5 + (ki-1)*5 + 3:4], 0, fit[K * 5 + (ki-1)*5 + 5]), 2) 
        updateLinv[1:2, 1:2, ki] <- solve(updateL[1:2, 1:2, ki])
        updateLinv[3:4, 1:2, ki] <- solve(updateL[3:4, 1:2, ki])
        updateMean[,ki, 2] <- fit[10*K + (ki-1) * dim * (dim + 1) + 1:dim]
        updateL[,,K + ki] <- matrix(fit[10*K + (ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
        updateLinv[,,K + ki] <- solve(updateL[,,K + ki])
      }
    } else {
      for(ki in 1:K){
        updateMean[,ki, 1] <- fit[(ki-1) * dim * (dim + 1) + 1:dim]
        updateL[,, ki] <- matrix(fit[(ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
        updateLinv[,, ki] <- solve(updateL[,,ki])
      }
    }
    return(list(updateMean = updateMean,
                updateL = updateL,
                updateLinv = updateLinv))
  }
}

sampleTheta <- function(update, samples, var, switch, K, dim, statVars, weights = NULL){
  tDynFull <- array(0, dim = c(dim, ifelse(var, 48, 1), K, samples))
  tConsFull <- array(0, dim = c(4, K, samples))
  fcVar <- array(0, dim = c(48, ifelse(var, 48, 1), K, samples))
  
  for(i in 1:samples){
    if(is.null(weights)){
      updateMean <- update$updateMean
      updateL <- update$updateL
      updateLinv <- update$updateLinv
      if(var){
        updateSd <- update$updateSd
      }
    } else {
      #sample from mixture
      u <- runif(1)
      sumW <- cumsum(weights)
      group <- min(which(u <= sumW))
      updateMean <- update[[group]]$updateMean
      updateL <- update[[group]]$updateL
      updateLinv <- update[[group]]$updateLinv
      if(var){
        updateSd <- update[[group]]$updateSd
      }
    }
    
    if(switch){
      for(ki in 1:K){
        tConsFull[1:2, ki, i] <- updateMean[1:2, ki, 1] + updateL[1:2, 1:2, ki] %*% rnorm(2)
        tConsFull[3:4, ki, i] <- 1 / (1 + exp(-(updateMean[3:4, ki, 1] + updateL[3:4, 1:2, ki] %*% rnorm(2))))
      }
    }
    
    if(var){
      for(ki in 1:K){
        
        theta <- updateMean[,,ki]  + updateSd[,,ki] *  matrix(rnorm(48 * dim), ncol = 48)
        
        for(j in 1:48){
          root <- polyroot(c(1, -theta[statVars, j]))
          if(any(Mod(root) < 1)){
            stat <- FALSE
            try <- 0
            while(!stat){
              theta[, j] <- updateMean[, j, ki]  + updateSd[, j, ki] * rnorm(dim)
              root <- polyroot(c(1, -theta[statVars, j]))
              try <- try + 1
              if(all(Mod(root) > 1)){
                stat <- TRUE
              } else if(try > 10 & i > 1){
                theta[statVars, j] <- tDynFull[statVars,j, ki, i-1]
                stat <- TRUE
              } else if(try > 10 & i == 1){
                theta[statVars, j] <- 0
                stat <- TRUE
              }
            }
          }
          autocov <- ltsa::tacvfARMA(theta[statVars, j], sigma2 = exp(theta[1, j]), maxLag = 48)
          fcVar[, j, ki, i] <- ltsa::PredictionVariance(autocov, 48)
        }
        tDynFull[,, ki, i] <- theta
      }
    } else {
      theta <- matrix(0, dim, K)
      for(ki in 1:K){
        theta[, ki] <- updateMean[,ki, 1 + switch] + updateL[,,K * switch + ki] %*% rnorm(dim)
        
        root <- polyroot(c(1, -theta[statVars, ki]))
        if(any(Mod(root) < 1)){
          stat <- FALSE
          try <- 0
          while(!stat){
            theta[, ji] <- updateMean[,ki, 1 + switch] + updateL[,,K * switch + ki] %*% rnorm(dim)
            root <- polyroot(c(1, -theta[statVars, ki]))
            try <- try + 1
            if(all(Mod(root) > 1)){
              stat <- TRUE
            } else if(try > 10 & i > 1){
              theta[statVars, j] <- tDynFull[statVars, 1, ki, i-1]
              stat <- TRUE
            } else if(try > 10 & i == 1){
              theta[statVars, j] <- 0
              stat <- TRUE
            }
          }
        }
        autocov <- ltsa::tacvfARMA(theta[statVars, ki], sigma2 = exp(theta[1, ki]), maxLag = 48)
        fcVar[,1, ki, i] <- ltsa::PredictionVariance(autocov, 48)
        tDynFull[,1,ki,i] <- theta[,ki]
      }
    }
  }
  return(list(tDynFull = tDynFull,
              tConsFull = tConsFull,
              fcVar = fcVar))
}

  
  
  
  
  
  

