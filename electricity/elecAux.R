updatePrior <- function(fit, var, switch, K, dim){
  # Calculate updated priors for theta parameters
  if(var){
    updateMean <- updateSd <- array(0, dim = c(dim, 48, K + switch))
    updateLinv <- array(0, dim = c(48, 48, K * (dim + switch)))
    updateL <- arrary(0, dim = c(4, 2, K))
    if(switch){
      updateMean[1:2, ki, 1] <- fit[(ki-1)*5 + 1:2]
      updateMean[3:4, ki, 1] <- fit[5*K + (ki-1)*5 + 1:2]
      updateL[1:2, 1:2, ki] <- matrix(c(fit[(ki-1)*5 + 3:4], 0, fit[(ki-1)*5 + 5]), 2)
      updateL[3:4, 1:2, ki] <- matrix(c(fit[K * 5 + (ki-1)*5 + 3:4], 0, fit[K[k] * 5 + (ki-1)*5 + 5]), 2) 
      updateLinv[1:2, 1:2, ki] <- solve(updateL[1:2, 1:2, ki])
      updateLinv[3:4, 1:2, ki] <- solve(updateL[3:4, 1:2, ki])
    } else {
      for(ki in 1:K){
        updateMean[,,ki] <- matrix(var$lambda[48*dim*2*(ki-1) + 1:(48*dim)], ncol = 48, byrow = TRUE)
        updateSd[,,ki] <- matrix(var$lambda[48*dim*(2*(ki-1)+1) + 1:(48*dim)], ncol = 48, byrow = TRUE)
        for(i in 1:dim){
          updateLinv[,,(ki-1)*dim + i] <- diag(updateSd[i,,ki])
        }
      }
    }
    return(list(updateMean = updateMean,
                updateSd = updateSd,
                updateL = updateL,
                updateLinv = updateLinv))
    
  } else {
    updateMean <- array(0, dim = c(dim, 1, K + switch))
    updateL <- updateLinv <- array(0, dim = c(dim, dim, K + K * switch))
    if(switch){
      for(ki in 1:K){
        updateMean[1:2, ki, 1] <- fit[(ki-1)*5 + 1:2]
        updateMean[3:4, ki, 1] <- fit[5*K + (ki-1)*5 + 1:2]
        updateL[1:2, 1:2, ki] <- matrix(c(fit[(ki-1)*5 + 3:4], 0, fit[(ki-1)*5 + 5]), 2)
        updateL[3:4, 1:2, ki] <- matrix(c(fit[K[k] * 5 + (ki-1)*5 + 3:4], 0, fit[K[k] * 5 + (ki-1)*5 + 5]), 2) 
        updateLinv[1:2, 1:2, ki] <- solve(updateL[1:2, 1:2, ki])
        updateLinv[3:4, 1:2, ki] <- solve(updateL[3:4, 1:2, ki])
        updateMean[,ki, 2] <- fit[10*K + (ki-1) * dim * (dim + 1) + 1:dim]
        updateL[,,K + ki] <- matrix(fit[10*K + (ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
        updateLinv[,,K + ki] <- solve(updateL[,,K + ki])
      }
    } else {
      for(ki in 1:K){
        updateMean[,ki] <- fit[(ki-1) * dim * (dim + 1) + 1:dim]
        updateL[,, ki] <- matrix(fit[(ki-1) * dim * (dim + 1) + (dim+1):(dim + dim^2)], dim)
        updateLinv[,, ki] <- solve(updateL[,,ki])
      }
    }
    return(list(updateMean = updateMean,
                updateL = updateL,
                updateLinv = updateLinv))
  }
}

sampleTheta <- function(update, samples, var, switch, K, dim){
  tDynFull <- array(0, dim = c(dim, ifelse(var, 48, 1), K, samples))
  tConsFull <- array(0, dim = c(4, K, samples))
  fcVar <- arrary(0, dim = c(48, ifelse(var, 48, 1), K, samples))
  
  updateMean <- update$updateMean
  updateL <- update$updateL
  updateLinv <- update$updateLinv
  
  if(switch){
    for(i in 1:samples){
      for(ki in 1:K){
        for(ki in 1:K[k]){
          tConsFull[1:2, ki, i] <- updateMean[1:2, ki, 1] + updateL[1:2, 1:2, ki] %*% rnorm(2)
          tConsFull[3:4, ki, i] <- 1 / (1 + exp(-(updateMean[3:4, ki, 1] + updateL[3:4, 1:2, ki] %*% rnorm(2))))
        }
      }
    }
  }
  
  if(var){
    updateSd <- update$updateSd
    for(i in 1:samples){
      for(ki in 1:K){
   
        theta <- updateMean[,,ki]  + updateSd[,,ki] *  matrix(rnorm(48 * dim), ncol = 48)
        
        for(j in 1:48){
          root <- polyroot(c(1, -theta[11:13, j]))
          if(any(Mod(root) < 1)){
            if(i > 1){
              theta[,j] <- tDynFull[, j, ki, i-1]
            } else {
              stat <- FALSE
              try <- 0
              while(!stat){
                theta[, j] <- updateMean[, j, ki]  + updateSd[, j, ki] * rnorm(dim)
                root <- polyroot(c(1, -theta[11:13, j]))
                try <- try + 1
                if(all(Mod(root) > 1)){
                  stat <- TRUE
                } else if(try > 10){
                  theta[11:13, j] <- 0
                  stat <- TRUE
                }
              }
            }
          }
          autocov <- ltsa::tacvfARMA(theta[11:13, j], sigma2 = exp(theta[1, j]), maxLag = 48)
          fcVar[, j, ki, i] <- ltsa::PredictionVariance(autocov, 48)
        }
        tDynFull[,, ki, i] <- theta
      } 
    }
  } else {
    for(i in 1:samples){
       for(ki in 1:K[k]){
          theta[, ki] <- updateMean[,switch + ki] + updateL[,,K * switch + ki] %*% rnorm(dim)
    
          root <- polyroot(c(1, -theta[11:13, ki]))
          if(any(Mod(root) < 1)){
          if(i > 1){
            theta[,ki] <- tDynFull[,1,ki,i-1]
          } else {
            stat <- FALSE
            try <- 0
            while(!stat){
              theta[, ki] <- updateMean[, switch + ki] + updateL[,,K * switch + ki] %*% rnorm(dim)
              
              root <- polyroot(c(1, -theta[11:13, ki]))
              try <- try + 1
              if(all(Mod(root) > 1)){
                stat <- TRUE
              } else if(try > 10){
                theta[11:13, ki] <- 0
                stat <- TRUE
              }
            }
          }
        }
        
          autocov <- ltsa::tacvfARMA(theta[11:13, ki], sigma2 = exp(theta[1, ki]), maxLag = 48)
          fcVar[,1, ki, i] <- ltsa::PredictionVariance(autocov, 48)
          tDynFull[,1,ki,i] <- theta[,ki]
       }
     }
  }
  return(list(tDynFull = tDynFull,
              tConsFull = tConsFull,
              fcVar = fcVar))
}

  
  
  
  
  
  

