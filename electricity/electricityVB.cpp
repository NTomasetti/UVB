// [[Rcpp::depends(rstan)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <stan/math.hpp>
#include <Eigen/Dense>
#include <RcppEigen.h>
#include <Rcpp.h>
#include <boost/math/distributions.hpp> 
#include <boost/math/special_functions.hpp> 


using namespace arma;
using namespace boost::math;
using namespace std;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd; 
using Eigen::Map;

// [[Rcpp::export]]
mat sobol_points(int N, int D) {
  using namespace std;
  std::ifstream infile("new-joe-kuo-6.21201" ,ios::in);
  if (!infile) {
    cout << "Input file containing direction numbers cannot be found!\n";
    exit(1);
  }
  char buffer[1000];
  infile.getline(buffer,1000,'\n');
  
  // L = max number of bits needed 
  unsigned L = (unsigned)ceil(log((double)N)/log(2.0)); 
  
  // C[i] = index from the right of the first zero bit of i
  unsigned *C = new unsigned [N];
  C[0] = 1;
  for (unsigned i=1;i<=N-1;i++) {
    C[i] = 1;
    unsigned value = i;
    while (value & 1) {
      value >>= 1;
      C[i]++;
    }
  }
  
  // POINTS[i][j] = the jth component of the ith point
  //                with i indexed from 0 to N-1 and j indexed from 0 to D-1
  mat POINTS(N, D, fill::zeros);
  
  // ----- Compute the first dimension -----
  
  // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
  unsigned *V = new unsigned [L+1]; 
  for (unsigned i=1;i<=L;i++) V[i] = 1 << (32-i); // all m's = 1
  
  // Evalulate X[0] to X[N-1], scaled by pow(2,32)
  unsigned *X = new unsigned [N];
  X[0] = 0;
  for (unsigned i=1;i<=N-1;i++) {
    X[i] = X[i-1] ^ V[C[i-1]];
    POINTS(i, 0) = (double)X[i]/pow(2.0,32); // *** the actual points
    //        ^ 0 for first dimension
  }
  
  // Clean up
  delete [] V;
  delete [] X;
  
  
  // ----- Compute the remaining dimensions -----
  for (unsigned j=1;j<=D-1;j++) {
    
    // Read in parameters from file 
    unsigned d, s;
    unsigned a;
    infile >> d >> s >> a;
    unsigned *m = new unsigned [s+1];
    for (unsigned i=1;i<=s;i++) infile >> m[i];
    
    // Compute direction numbers V[1] to V[L], scaled by pow(2,32)
    unsigned *V = new unsigned [L+1];
    if (L <= s) {
      for (unsigned i=1;i<=L;i++) V[i] = m[i] << (32-i); 
    }
    else {
      for (unsigned i=1;i<=s;i++) V[i] = m[i] << (32-i); 
      for (unsigned i=s+1;i<=L;i++) {
        V[i] = V[i-s] ^ (V[i-s] >> s); 
        for (unsigned k=1;k<=s-1;k++) 
          V[i] ^= (((a >> (s-1-k)) & 1) * V[i-k]); 
      }
    }
    
    // Evalulate X[0] to X[N-1], scaled by pow(2,32)
    unsigned *X = new unsigned [N];
    X[0] = 0;
    for (unsigned i=1;i<=N-1;i++) {
      X[i] = X[i-1] ^ V[C[i-1]];
      POINTS(i, j) = (double)X[i]/pow(2.0,32); // *** the actual points
      //        ^ j for dimension (j+1)
    }
    
    // Clean up
    delete [] m;
    delete [] V;
    delete [] X;
  }
  delete [] C;
  
  return POINTS;
}

// [[Rcpp::export]]
mat shuffle(mat sobol){
  using namespace std;
  int N = sobol.n_rows;
  int D = sobol.n_cols;
  mat output(N, D, fill::zeros);
  // draw a random rule of: switch 1 and 0  /  do not switch for each binary digit.
  vec rule = randu<vec>(16);
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < D; ++j){
      // grab element of the sobol sequence
      double x = sobol(i, j);
      // convert to a binary representation
      uvec binary(16, fill::zeros);
      for(int k = 1; k < 17; ++k){
        if(x > pow(2, -k)){
          binary(k-1) = 1;
          x -= pow(2, -k);
        }
      }
      // apply the transform of tilde(x_k) = x_k + a_k mod 2, where a_k = 1 if rule_k > 0.5, 0 otherwise
      for(int k = 0; k < 16; ++k){
        if(rule(k) > 0.5){
          binary(k) = (binary(k) + 1) % 2;
        }
      }
      // reconstruct base 10 number from binary representation
      for(int k = 0; k < 16; ++k){
        if(binary(k) == 1){
          output(i, j) += pow(2, -(k+1));
        }
      }
    }
  }
  return output;
}

//  y: Data, rows: t, columns: i
// x: coefficient vector of intercept / temperature / day of week / holiday
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// pS1, prior probability of being in state 1
// priorMean: A cube, first slice, first two rows: Constant means, second two rows: Switching means, second slide: dynamic means. Columns: Each K 
// priorLinv: A cube, first K slices, first two rows: constant linv, second two rhos: rho linv. Second K slices: dynamic linv
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
// priorWeights: priorWeights of prior distribution, prior columns / slices are added as needed. Eg, first K columns for first component, second K for second...
struct electricitySwitching {
  const mat y;
  const mat x;
  const vec epsilon;
  const mat probK;
  const mat pS1;
  const cube priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  const bool uniformRho;
  const vec priorWeights;
  electricitySwitching(const mat& yIn, const mat& xIn, const vec& epsIn, const mat& probKIn, const mat& pS1In, const cube& priorMeanIn, const cube& priorLinvIn, 
                       const vec& orderIn, const int& TnIn, const bool& uRhoin, const vec& priorWeightsIn ) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), pS1(pS1In), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn), uniformRho(uRhoin), priorWeights(priorWeightsIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int numPars = 1 + order.n_elem + x.n_cols; // Parameters = Logvar, mean, beta + order parameters
    int lambdaPerK = numPars * (numPars + 1); // Size of variance matrix for each dynamic model
    
    // Create the constant varaiance and mean parameters and switching probabilities
    // Columns: Model, Rows: log var / mean, p01, p10
    Matrix<T, Dynamic, Dynamic> thetaC(2, K),  rhoUnnorm(2, K),  rho(2, K); 
    // Add parts to logdet J as we go
    T logdetJ = 0;
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < 2; ++i){
        thetaC(i, k) = lambda(k*5 + i);
        rhoUnnorm(i, k) = lambda(K*5 + k*5 + i);
        for(int j = 0; j <= i; ++j){
          thetaC(i, k) += lambda(k*5 + 2 + i + j) * epsilon(k*2 + j);
          rhoUnnorm(i, k) += lambda(K*5 + k*5 + 2 + i + j) * epsilon(K*2 + k*2 + j);
          if(i == j){
            logdetJ += log(fabs(lambda(k*5 + 2 + i + j))) + log(fabs(lambda(K*5 + k*5 + 2 + i + j)));
          }
          if(i == 0){
            logdetJ += thetaC(i, k);
          }
        }
        rho(i, k) = 1.0 / (1 + exp(-rhoUnnorm(i, k)));
        logdetJ += -rhoUnnorm(i, k) - 2 * log(exp(rhoUnnorm(i, k)) + 1);
      }
    }
    // Uses first 10*K lambda params and 4*K epsilon values
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    Matrix<T, Dynamic, Dynamic> thetaD (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        thetaD(i, k) = lambda(10*K + k * lambdaPerK + i);
        for(int j = 0; j <= i; ++j){
          thetaD(i, k) += lambda(10*K + k * lambdaPerK + numPars * (i + 1) + j) * epsilon(4*K + k*numPars + j);
          if(j == i){
            logdetJ += log(fabs(lambda(10*K + k * lambdaPerK + numPars * (i + 1) + j))); // add log(L_ii) to log det
          }
        }
        if(i == 0){
          logdetJ += thetaD(i, k); // Add log variance to log det
        }
      }
    }
    
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    T prior = 0;
    Matrix<T, Dynamic, 1> kernelCons(2);
    for(int k = 0; k < K; ++k){
      kernelCons.fill(0);
      for(int i = 0; i < 2; ++i){
        for(int j = 0; j <= i; ++j){
          kernelCons(i) += (thetaC(j, k) - priorMean(j, k, 0)) * priorLinv(i, j, k);
        }
        prior += - 0.5 * pow(kernelCons(i), 2);
      }
    }
    
    // Rho has a uniform prior on the first fit, but has a MVN prior for updates
    if(!uniformRho){
      for(int k = 0; k < K; ++k){
        kernelCons.fill(0);
        for(int i = 0; i < 2; ++i){
          for(int j = 0; j <= i; ++j){
            kernelCons(i) += (rhoUnnorm(j, k) - priorMean(2 + j, k, 0)) * priorLinv(2 + i, j, k);
          }
          prior += - 0.5 * pow(kernelCons(i), 2);
        }
      }
    }
    
    // Add the dynamic model priors, each independent normal (not identitcal)
    Matrix<T, Dynamic, Dynamic> kernelDyn (numPars, K);
    kernelDyn.fill(0);
    
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        for(int j = 0; j <= i; ++j){
          kernelDyn(i, k) += (thetaD(j, k) - priorMean(j, k, 1)) * priorLinv(i, j, K + k);
        }
        prior += - 0.5 * pow(kernelDyn(i, k), 2);
      }
    }
    // Calculate the log likelihood
    T logLik = 0;
    for(int i = 0; i < N; ++i){
      // Starting point is tau, will be part way through the data to deal with lagged variables
      int tau = y.n_rows - Tn;
      // Create likelihood vector as a sum of K individual likelihoods.
      Matrix<T, Dynamic, 1> likelihood(Tn);
      likelihood.fill(0);
      for(int k = 0; k < K; ++k){
        // Grab p(S_i,T = 1 | y_i, 1:T)
        T pS1modK = pS1(k, i);
        // components of the dynamic model
        T var = exp(thetaD(0, k));
        
        // Calculate dynamic yt ~ N(beta * x + sum_l (theta * (y_t-l - beta*x_t-l)), var)
        for(int t = tau; t < y.n_rows; ++t){
          T mean = 0;
          // Add B X_t
          for(int m = 0; m < x.n_cols; ++m){
            mean += thetaD(1 + m, k) * x(t, m);
          }
          // Deal with any AR terms 
          for(int l = 0; l < order.n_elem; ++l){
            T AR = y(t - order(l));
            // Subtract B X_{t-l}
            for(int m = 0; m < x.n_cols; ++m){
              AR -= thetaD(1 + m, k) * x(t - order(l), m);
            }
            mean += thetaD(1 + x.n_cols + l, k) * AR;
          }
          
          T eta1 = 1.0 / sqrt(2 * 3.14159 * var) * exp(-pow(y(t, i) - mean, 2) / (2 * var));
          T eta0 = 1.0 / sqrt(2 * 3.14159 * exp(thetaC(0, k))) * exp(-pow(y(t, i) - thetaC(1, k), 2) / (2 * exp(thetaC(0, k))));
          
          // Hamilton Filter Update
          T lik0 = (1 - pS1modK) * (1 - rho(0, k)) * eta0 + pS1modK * rho(1, k) * eta0;
          T lik1 = pS1modK * (1 - rho(1, k)) * eta1 + (1 - pS1modK) * rho(0, k) * eta1;
          likelihood(t - tau) += probK(k, i) * (lik0 + lik1);
          pS1modK = lik1 / (lik0 + lik1);
        }       
      }
      // Update loglikelihood now that all K terms in likelihood have been added
      for(int t = tau; t < y.n_rows; ++t){
        logLik += log(likelihood(t - tau));
      }
    }
    
    return prior + logLik + logdetJ;
  }
};

// Similar to the above switching model but with a different parameter for each half hour period (ie a restricted VAR).
// The dynamic model MVN approximaiton now have diagonal covariance, param: Mean / Log(Sd)
// y: Data, rows: t, columns: i
// x: regressor matrix
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// pS1, prior probability of being in state 1
// priorMean: A cube, first slice, first two rows: Constant means, second two rows: Switching means, next K slices: dynamic means. Columns: Each halfhour 
// priorLinv, a cube of the constant model + rho L inverse (one slice per group) + the dim * 48 theta^H inverses
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
// priorWeights: priorWeights of prior distribution, prior columns / slices are added as needed. Eg, first K columns for first component, second K for second...
struct electricitySwitchingVAR {
  const mat y;
  const mat x;
  const vec epsilon;
  const mat probK;
  const mat pS1;
  const cube priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  const bool uniformRho;
  const vec priorWeights;
  electricitySwitchingVAR(const mat& yIn, const mat& xIn, const vec& epsIn, const mat& probKIn, const mat& pS1In, const cube& priorMeanIn, 
                          const  cube& priorLinvIn, const vec& orderIn, const int& TnIn, const bool& uRhoin, const vec& priorWeightsIn) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), pS1(pS1In), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn), uniformRho(uRhoin), priorWeights(priorWeightsIn)  {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int dim = 1 + x.n_cols + order.n_elem;
    int numPars = 48 * dim; // Parameters = Logvar, mean, beta + order parameters
   
    
    // Create the constant varaiance and mean parameters and switching probabilities
    // Columns: Model, Rows: log var / mean, p01, p10
    Matrix<T, Dynamic, Dynamic> thetaC(2, K),  rhoUnnorm(2, K),  rho(2, K); 
    // Define the output variables
    T logdetJ = 0, prior = 0, logLik = 0;
    // We will add parts to logdet J as we go
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < 2; ++i){
        thetaC(i, k) = lambda(k*5 + i);
        rhoUnnorm(i, k) = lambda(K*5 + k*5 + i);
        for(int j = 0; j <= i; ++j){
          thetaC(i, k) += lambda(k*5 + 2 + i + j) * epsilon(k*2 + j);
          rhoUnnorm(i, k) += lambda(K*5 + k*5 + 2 + i + j) * epsilon(K*2 + k*2 + j);
          if(i == j){
            logdetJ += log(fabs(lambda(k*5 + 2 + i + j))) + log(fabs(lambda(K*5 + k*5 + 2 + i + j)));
          }
          if(i == 0){
            logdetJ += thetaC(i, k);
          }
        }
        rho(i, k) = 1.0 / (1 + exp(-rhoUnnorm(i, k)));
        logdetJ += -rhoUnnorm(i, k) - 2 * log(exp(rhoUnnorm(i, k)) + 1);
      }
    }
    // Create theta for the dynamic models, rows: Different parameters, cols: Different models
    Matrix<T, Dynamic, Dynamic> thetaD (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        thetaD(i, k) = lambda(10*K + k * 2 * numPars + i) + exp(lambda(10*K + (k * 2 + 1) * numPars + i)) * epsilon(4*K + k * numPars + i);
        logdetJ += lambda(10*K + (2 * k + 1) * numPars + i); // add log(L_ii) to log det
        if(i < 48){
          logdetJ += thetaD(i, k); // Add log variance to log det
        }
      }
    }
    
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    Matrix<T, Dynamic, 1> kernelCons(2);
    for(int k = 0; k < K; ++k){
      kernelCons.fill(0);
      for(int i = 0; i < 2; ++i){
        for(int j = 0; j <= i; ++j){
          kernelCons(i) += (thetaC(j, k) - priorMean(j, k, 0)) * priorLinv(i, j, k);
        }
        prior += - 0.5 * pow(kernelCons(i), 2);
      }
    }
    // Rho has a uniform prior on the first fit, but has a MVN prior for updates
    if(!uniformRho){
      for(int k = 0; k < K; ++k){
        kernelCons.fill(0);
        for(int i = 0; i < 2; ++i){
          for(int j = 0; j <= i; ++j){
            kernelCons(i) += (rhoUnnorm(j, k) - priorMean(2 + j, k, 0)) * priorLinv(2 + i, j, k);
          }
          prior += - 0.5 * pow(kernelCons(i), 2);
        }
      }
    }
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    // Add the dynamic model priors, each independent normal (not identitcal)
    // Each set of 48 thetas has its own 48*48 priorLinv, so we iterate over the groups, then theta blocks, then each halfhour parameter
    // Each L inverse is a banded matrix, containing terms only in the diagonal and row immediately below the diagonal.
    // So we only need to evaluate two columns per row instead of the whole lower triangle of the matrix.
    for(int k = 0; k < K; ++k){
      for(int d = 0; d < dim; ++d){
        for(int i = 0; i < 48; ++i){
          T kernelDyn = 0;
          for(int j = max(0, i-1); j <= i; ++j){
            kernelDyn += (thetaD(48*d + j, k) - priorMean(d, j, k)) * priorLinv(i, j, K + dim*k + d);
          }
          prior += - 0.5 * pow(kernelDyn, 2);
        }
      }
    }
    // Calculate the log likelihood
    Matrix<T, Dynamic, 1> pS1vec(N);
    for(int i = 0; i < N; ++i){
      // Starting point is tau, will be part way through the data to deal with lagged variables
      int tau = y.n_rows - Tn;
      // For each smart meter we create the likelihood as a sum of K many Hamilton Filtered likelihoods.
      Matrix<T, Dynamic, 1> likelihood(Tn);
      likelihood.fill(0);
      for(int k = 0; k < K; ++k){
        T pS1modK = pS1(k, i);
        // yt ~ N(beta * x + sum_l (theta * (y_t-l - beta*x_t-l)), var)
        for(int t = tau; t < y.n_rows; ++t){
          int halfhour = t % 48;
          // components of the dynamic model
          T var = exp(thetaD(halfhour, k));
          T mean = 0;
          // Add B X_t
          for(int m = 0; m < x.n_cols; ++m){
            mean += thetaD(48 * (1 + m) + halfhour, k) * x(t, m);
          }
          // Deal with any AR terms 
          for(int l = 0; l < order.n_elem; ++l){
            T AR = y(t - order(l));
            // Subtract B X_{t-l}
            for(int m = 0; m < x.n_cols; ++m){
              AR -= thetaD(48 * (1 + m) + halfhour, k) * x(t - order(l), m);
            }
            mean += thetaD(48 * (1 + x.n_cols + l) + halfhour, k) * AR;
          }
          
          
          T eta1 = 1.0 / sqrt(2 * 3.14159 * var) * exp(-pow(y(t, i) - mean, 2) / (2 * var));
          T eta0 = 1.0 / sqrt(2 * 3.14159 * exp(thetaC(0, k))) * exp(-pow(y(t, i) - thetaC(1, k), 2) / (2 * exp(thetaC(0, k))));
          
          // Hamilton Filter Update
          T lik0 = (1 - pS1modK) * (1 - rho(0, k)) * eta0 + pS1modK * rho(1, k) * eta0;
          T lik1 = pS1modK * (1 - rho(1, k)) * eta1 + (1 - pS1modK) * rho(0, k) * eta1;
          likelihood(t - tau) += probK(k, i) * (lik0 + lik1);
          pS1modK = lik1 / (lik0 + lik1);
        }      
      }
      for(int t = 0; t < Tn; ++t){
        logLik += log(likelihood(t));
      }
    }
    
    return prior + logLik + logdetJ;
  }
};


//  y: Data, rows: t, columns: i
// x: temperature vector with rows matching y time period
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// prior: All of the prior components to deal with
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
// priorWeights: priorWeights of prior distribution, prior columns / slices are added as needed. Eg, first K columns for first component, second K for second...
struct electricityStandard{
  const mat y;
  const mat x;
  const vec epsilon;
  const mat probK;
  const cube priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  const vec priorWeights;
  electricityStandard(const mat& yIn, const mat& xIn, const vec& epsIn, const mat& probKIn, const cube& priorMeanIn, const cube& priorLinvIn, const vec& orderIn, const int& TnIn, const vec& priorWeightsIn) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn), priorWeights(priorWeightsIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int numPars = 1 + x.n_cols + order.n_elem; // Parameters = Logvar, mean, beta + order parameters
    int lambdaPerK = numPars * (numPars + 1); // Size of parameter vector for each dynamic model
    int mix = priorWeights.n_elem; // number of components of prior distribution
    
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    T logdetJ = 0, prior = 0, logLik = 0;
    Matrix<T, Dynamic, Dynamic> theta (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        theta(i, k) = lambda(k * lambdaPerK + i);
        for(int j = 0; j <= i; ++j){
          theta(i, k) += lambda(k * lambdaPerK + numPars * (i + 1) + j) * epsilon(k*numPars + j);
          if(j == i){
            logdetJ += log(fabs(lambda(k * lambdaPerK + numPars * (i + 1) + j))); // add log(L_ii) to log det
          }
        }
        if(i == 0){
          logdetJ += theta(i, k); // Add log variance to log det
        }
      }
    }
    
    // Evaluate log(p(theta))
    // prior is a mixture of block diagonal normals (independence between different Ks)
    vec dets(mix, fill::ones);
    Matrix<T, Dynamic, 1> priorComp(mix);
    priorComp.fill(0);
    
    for(int m = 0; m < mix; ++m){
      Matrix<T, Dynamic, Dynamic> kernelDyn (numPars, K);
      kernelDyn.fill(0);
      for(int k = 0; k < K; ++k){
        for(int i = 0; i < numPars; ++i){
          for(int j = 0; j <= i; ++j){
            kernelDyn(i, k) += (theta(j, k) - priorMean(j, k, m)) * priorLinv(i, j, K*m + k);
          }
          dets(m) *= priorLinv(i, i, K*m + k);
          priorComp(m) += - 0.5 * pow(kernelDyn(i, k), 2);
        }
      }
      prior = prior + priorWeights(m) * dets(m) * exp(priorComp(m));
    }
    prior = log(prior);
    
  
    
    // Calculate the log likelihood
    // Starting point is tau, will be part way through the data as the first data points are lags.
    int tau = y.n_rows - Tn;
    
    for(int i = 0; i < N; ++i){
      for(int t = tau; t < y.n_rows; ++t){
        T likelihood = 0;
        for(int k = 0; k < K; ++ k){
          
          T mean = 0;
          // Add B X_t
          for(int m = 0; m < x.n_cols; ++m){
            mean += theta(1 + m, k) * x(t, m);
          }
          // Deal with any AR terms 
          for(int l = 0; l < order.n_elem; ++l){
            T AR = y(t - order(l));
            // Subtract B X_{t-l}
            for(int m = 0; m < x.n_cols; ++m){
              AR -= theta(1 + m, k) * x(t - order(l), m);
            }
            mean += theta(1 + x.n_cols + l, k) * AR;
          }
          
          // Calculate thel likelihood as the sum of each model likelihood * p(k_i = j)
          likelihood += probK(k, i) / sqrt(2 * 3.14159 * exp(theta(0, k))) * exp(-pow(y(t, i) - mean, 2) / (2 * exp(theta(0, k))));
        }
        logLik += log(likelihood);
      }
    }
    return prior + logLik + logdetJ;
  }
};

// VAR, no switching
//  y: Data, rows: t, columns: i
// x: temperature vector with rows matching y time period
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// priorMean, a 17 * 48 * K cube, one column per halfhour, rows are the means, slices are the groups.
// prior Linv, a 48*48*(K*17) cube. Slice 17*k + d is the L inverse matrix for the prior of the 48 elements of theta_d,k
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
// priorWeights: priorWeights of prior distribution, prior columns / slices are added as needed. Eg, first K columns for first component, second K for second...
struct electricityStandardVAR{
  const mat y;
  const mat x;    
  const vec epsilon;
  const mat probK;
  const cube priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  const vec priorWeights;
  electricityStandardVAR(const mat& yIn, const mat& xIn, const vec& epsIn, const mat& probKIn, const cube& priorMeanIn, const cube& priorLinvIn, const vec& orderIn, const int& TnIn, const vec& priorWeightsIn) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn), priorWeights(priorWeightsIn)  {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int dim = 1 + x.n_cols + order.n_elem;
    int numPars = 48 * dim; // Parameters = Logvar, mean, beta + order parameters
    int mix = priorWeights.n_elem; // number of components of prior distribution
    
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    T logdetJ = 0, prior = 0, logLik = 0;
    Matrix<T, Dynamic, Dynamic> theta (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        theta(i, k) = lambda(2 * k * numPars + i) + exp(lambda((2 * k + 1) * numPars + i)) * epsilon(k*numPars + i);
        logdetJ += lambda((2 * k + 1) * numPars + i); // add log(L_ii) to log det
        if(i < 48){
          logdetJ += theta(i, k); // Add log variances to log det
        }
      }
    }
    
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    // Add the dynamic model priors, each independent normal (not identitcal)
    // Each set of 48 thetas has its own 48*48 priorLinv, so we iterate over the groups, then theta blocks, then each halfhour parameter
    // Each L inverse is a banded matrix, containing terms only in the diagonal and row immediately below the diagonal.
    // So we only need to evaluate two columns per row instead of the whole lower triangle of the matrix.
    // In updates it is diagonal
    // Evaluate log(p(theta))
    // prior is a mixture of block diagonal normals (independence between different Ks)
    vec dets(mix, fill::ones);
    Matrix<T, Dynamic, 1> priorComp(mix);
    priorComp.fill(0);
    
    for(int m = 0; m < mix; ++m){
      for(int k = 0; k < K; ++k){
        for(int d = 0; d < dim; ++d){
          for(int i = 0; i < 48; ++i){
            T kernelDyn = 0;
            for(int j = max(0, i-1); j <= i; ++j){
              kernelDyn += (theta(48*d + j, k) - priorMean(d, j, m*K + k)) * priorLinv(i, j, m * dim * K + dim*k + d);
            }
            priorComp(m) += -0.5 * pow(kernelDyn, 2);
            dets(m) *= priorLinv(i, i, m * dim * K + dim * k + d);
          }
        }
      }
      prior += priorWeights(m) * dets(m) * exp(priorComp(m));
    }
    prior = log(prior);
    
    // Calculate the log likelihood
    // Starting point is tau, will be part way through the data as the first data points are lags.
    int tau = y.n_rows - Tn;
    for(int i = 0; i < N; ++i){
      for(int t = tau; t < y.n_rows; ++t){
        int halfhour = t % 48;
        T likelihood = 0;
        for(int k = 0; k < K; ++ k){
          T mean = 0;
          // Add B X_t
          for(int m = 0; m < x.n_cols; ++m){
            mean += theta(48 * (1 + m) + halfhour, k) * x(t, m);
          }
          // Deal with any AR terms 
          for(int l = 0; l < order.n_elem; ++l){
            T AR = y(t - order(l));
            // Subtract B X_{t-l}
            for(int m = 0; m < x.n_cols; ++m){
              AR -=  theta(48 * (1 + m) + halfhour, k) * x(t - order(l), m);
            }
            mean +=  theta(48 * (1 + x.n_cols + l) + halfhour, k) * AR;
          }
          
          likelihood += probK(k, i) / sqrt(2 * 3.14159 * exp(theta(halfhour, k))) * exp(-pow(y(t, i) - mean, 2) / (2 * exp(theta(halfhour, k))));
        }
        logLik += log(likelihood);
      }
    }
    return prior + logLik + logdetJ;
  }
};

// [[Rcpp::export]]
Rcpp::List elecModel(mat y, Rcpp::NumericMatrix lambdaIn, vec epsilon, mat x, mat probK, cube priorMean, cube priorLinv, vec order, vec priorWeights,
                     int Tn, mat ps1, bool uniformRho = true, bool switching = false, bool var = false) {
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  if(switching){
    if(var){
      electricitySwitchingVAR p(y, x, epsilon, probK, ps1, priorMean, priorLinv, order, Tn, uniformRho, priorWeights);
      stan::math::set_zero_all_adjoints();
      stan::math::gradient(p, lambda, eval, gradP);
    } else {
      electricitySwitching p(y, x, epsilon, probK, ps1, priorMean, priorLinv, order, Tn, uniformRho, priorWeights);
      stan::math::set_zero_all_adjoints();
      stan::math::gradient(p, lambda, eval, gradP);
    }
  } else {
    if(var){
      electricityStandardVAR p(y, x, epsilon, probK, priorMean, priorLinv, order, Tn, priorWeights);
      stan::math::set_zero_all_adjoints();
      stan::math::gradient(p, lambda, eval, gradP);
    } else {
      electricityStandard p(y, x, epsilon, probK, priorMean, priorLinv, order, Tn, priorWeights);
      stan::math::set_zero_all_adjoints();
      stan::math::gradient(p, lambda, eval, gradP);
    }
  }
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

// Calculates the likelihood of y_t | theta, y_{1:t-1} for a sarimax(3,0,0)(3,0,0)(1,0,0) model
// Supports VAR or non-VAR models
// Sarimas have a complex lag strucutre, there is quite a few cross product terms between each lag order. Easy to add terms to first to orders, more work to add extra second seasonality parameters
// y :Data
// t : Particular y to be evaluated
// theta parameters: logvar, mean, ar_order_1, ...
// order integer positions of lags
// log: boolean, return log likelihood if true
// var: boolean, return the var model with halfhours corresponding to columns of theta
// [[Rcpp::export]]
vec arLikelihood(vec y, mat x, mat theta, vec order, int T, bool log, bool var){
  int hh = 0, tau = y.n_elem - T;
  vec beta, phi, likelihood(T);
  
  if(!var){
    beta = theta(span(1, x.n_cols), 0);
    phi = theta(span(x.n_cols + 1, x.n_cols + order.n_elem), 0);
  }
  
  for(int t = tau; t < y.n_elem; ++t){
    if(var){
      int hh = t % 48;
      beta = theta(span(1, x.n_cols), hh);
      phi = theta(span(x.n_cols + 1, x.n_cols + order.n_elem), hh);
    }
    
    double mean = as_scalar(x.row(t) * beta);
    // Deal with any AR terms containing the non-seasonal AR components
    for(int l = 0; l < order.n_elem; ++l){
      mean += phi(l) * (y(t - order(l)) - as_scalar(x.row(t - order(l)) * beta));
    }
    // Finally calculate the likelihood
    if(log){
      likelihood(t - tau) =  - 0.5 * std::log(2 * 3.14159) - 0.5 * theta(0, hh) - pow(y(t) - mean, 2) / (2 * exp(theta(0, hh)));
    } else {
      likelihood(t - tau) = 1.0 / sqrt(2 * 3.14159 * exp(theta(0, hh))) * exp(-pow(y(t) - mean, 2) / (2 * exp(theta(0, hh))));
    }
  }
  return likelihood;
}

// Returns p(k = j | theta, y) up to proportionality for non switching models
// Passes the boolean var through tp arLikelihood
// Essentially a log version of Ar likelihood that includes the prior
// y: Data
// theta: parameters of dynamic model j, per halfhour columns if a var
// order: arima lags
// prior p(k = j) 
// T: amount of data to be evaluated
// [[Rcpp::export]]
double probK (vec y, mat x, mat theta, vec order, double prior, int T, bool var){
  double logdens = log(prior);
  vec loglik = arLikelihood(y, x, theta, order, T, true, var);
  logdens += sum(loglik);
  return logdens;
}

// Returns p(k = j | theta, y_{1:T}) and p(s_T = 1 | theta, K, y_{1:T})up to proportionality
// y: Data
// thetaD: parameters of dynamic model j with per halfhour columns
// thetaC: parameters of constant model j
// order: arima lags
// pS1: p(s_1 = 1 | everything), or eta
// rho: p01 and p10 in a vector for model j
// prior p(k = j) 
// T: size of data (excluding lags)
// var model, passed through to arlikelihood
// [[Rcpp::export]]
vec probSKHF (vec y, mat x, mat thetaD, vec thetaC, vec order, double pS1, double prior, int T, bool var){
  int tau = y.n_elem - T;
  vec eta1 = arLikelihood(y, x, thetaD, order, T, true, var);
  
  double logdens = log(prior);
  
  for(int t = tau; t < y.n_elem; ++t){
    double eta0 = -0.5 * log(2 * 3.14159) - 0.5 * thetaC(0) - pow(y(t) - thetaC(1), 2) / (2 * exp(thetaC(0)));
    
    double likelihood0 = exp(eta0) * ((1 - pS1) * (1 - thetaC(2)) + pS1 * thetaC(3));
    double likelihood1 = exp(eta1(t-tau)) * (pS1 * (1 - thetaC(3)) + (1 - pS1) * thetaC(2));
    
    logdens += log(likelihood0 + likelihood1);
    pS1 = likelihood1 / (likelihood0 + likelihood1);
  }
  return {logdens, pS1};
}

// Input data, theta draws and other relevant informaiton
// Will evaluate p(ki = j) and forecast
// Theta should be 17 * (48 or 1) * K, so groups in slices and var halfhours in columns
// fcVAR should work the same way, 48 periods ahead in rows, 1 or 48 per halfhour sets in columns, and groups in slices
// will forecast the number of rows in fcVar ahead
// [[Rcpp::export]]
mat forecastStandard (vec y, mat x, cube theta, vec order, vec pkNorm, cube fcVar, vec support, bool var){
  int M = support.n_elem;
  int H = fcVar.n_rows;
  int K = pkNorm.n_elem;
  int T = y.n_rows - H - 1;
  mat density(M, H, fill::zeros);
  
  // Forecast each model, skip components with zero probability
  for(int ki = 0; ki < K; ++ki){
    vec beta, phi, ylag = {y(T), y(T-1), y(T-2)};
    int hh = 0;
      
    for(int h = 1; h <= H; ++h){
      if(var){
        hh = (T + h) % 48;
      }
      vec beta = theta.slice(ki)(span(1, x.n_cols), hh);
      vec phi = theta.slice(ki)(span(x.n_cols+1, x.n_cols + order.n_elem), hh);
        
      double mean = as_scalar(x.row(T + h) * beta);
        
      // Deal with any AR terms containing the non-seasonal AR components
      for(int l = 0; l < 3; ++l){
        mean += phi(l) * (ylag(l) - as_scalar(x.row(T + h - order(l)) * beta));
      }
      for(int l = 3; l < order.n_elem; ++l){
        mean += phi(l) * (y(T + h - order(l)) - as_scalar(x.row(T + h - order(l)) * beta));
      }
      
      for(int l = 2; l > 0; --l){
        ylag(l) = ylag(l - 1);
      }
      ylag(0) = mean;
      for(int m = 0; m < M; ++m){
        density(m, h-1) += pkNorm(ki) / sqrt(2 * 3.14159 * fcVar(h - 1, hh, ki)) * exp(-pow(support(m) - mean, 2) / (2 * fcVar(h-1, hh, ki)));
      }
    }
  }
  
  return density;
  
}

// return the forecast density matrix (over the support grid and values of H) for a given p(S = 1 | y_{1:T}) and draw of k / theta.
// Will first evaluate the pk / ps probabilities, and then use these through a hamilton Filter based forecast
// Inputs are y, from the start of sample until the end of the forecasts (last values unused)
// x, temperature for the same period, will use the last values for forecasting
// ThetaC, the log variance and mean of the constant model plus rho01 and rho10
// Rho, the HF switching probabilities,
// ThetaD, a dim * (1 or 48 for var) * K parameter array
// fcVar, a H * (1 or 48) * K array of h step forecast variances
// Tn, the period evaluated for pk and ps
// and other standard inputs
// PS1 has three distinct time periods, the prior value, the posterior (p(S_T = 1 | y_{1:T})) and forecast values p(S_T+h  = 1 | y_1:T) for each group
// Forecasts are relatively short term, so seasonal AR components can use the true lagged value instead of a y-hat
// so ylag cycles through \hat{y}_{t-i} for i = 1, 2, 3 
// fcVar is the AR model prediction variance as a function of sigma^2 and the non-seasonal AR components, can be calculated ahead of time.
// [[Rcpp::export]]
mat forecastHF(vec y, mat x, mat thetaC, cube thetaD, cube fcVar, vec order, vec pS1prior, vec pkNorm, vec support, bool var){
  int M = support.n_elem;
  int H = fcVar.n_rows;
  int K = pkNorm.n_elem;
  int T = y.n_rows - H - 1;
  mat density(M, H, fill::zeros);
  // Evaluate p(k_i = j) and p(S_T = 1)
  // Forecast each model, skip ones with zero probability
  for(int ki = 0; ki < K; ++ki){
    if(pkNorm(ki) > 0){
      double pS1FC = pS1prior(ki);
      vec beta, phi, ylag = {y(T), y(T-1), y(T-2)};
      int hh = 0;
      
      for(int h = 1; h <= H; ++h){
        pS1FC = pS1FC * (1 - thetaC(3, ki)) + (1 - pS1FC) * thetaC(2, ki);
        if(var){
          hh = (T + h) % 48; 
        }
        beta = thetaD.slice(ki)(span(1, x.n_cols), hh);
        phi = thetaD.slice(ki)(span(1 + x.n_cols, x.n_cols + order.n_elem), hh);
        
        double mean = as_scalar(x.row(T + h) * beta);
        
        // Deal with any AR terms containing the non-seasonal AR components
        for(int l = 0; l < 3; ++l){
          mean += phi(l) * (ylag(l) - as_scalar(x.row(T + h - order(l)) * beta));
        }
        for(int l = 3; l < order.n_elem; ++l){
          mean += phi(l) * (y(T + h - order(l)) - as_scalar(x.row(T + h - order(l)) * beta));
        }
        
        
        for(int l = 2; l > 0; --l){
          ylag(l) = ylag(l - 1);
        }
        ylag(0) = mean;
        for(int m = 0; m < M; ++m){
          density(m, h-1) += (pS1FC / sqrt(2 * 3.14159 * exp(thetaC(0, ki))) * exp(-pow(support(m) - thetaC(1, ki), 2) / (2 * exp(thetaC(0, ki)))) +
            (1 - pS1FC) / sqrt(2 * 3.14159 * fcVar(h - 1, hh, ki)) * exp(-pow(support(m) - mean, 2) / (2 * fcVar(h-1, hh, ki)))) * pkNorm(ki);
        }
      }
    }
  }
  
  return density;
  
}
