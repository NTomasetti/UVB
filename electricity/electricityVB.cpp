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

// Calculates the likelihood of y_t | theta, y_{1:t-1} for an arbitary ar model
// y :Data
// t : Particular y to be evaluated
// theta parameters: logvar, mean, ar_order_1, ...
// order integer positions of lags
// [[Rcpp::export]]
vec arLikelihood(vec y, vec x, int T, vec theta, vec order){
  int tau = y.n_elem - T;
  double var = exp(theta(0)), mu = theta(1), beta = theta(2);
  vec likelihood(T);
  
  for(int t = tau; t < y.n_elem; ++t){
    double mean = mu + beta * x(t);
    for(int i = 0; i < order.n_elem; ++i){
      mean += theta(3 + i) * (y(t - order(i)) - mu -  beta * x(t- order(i)));
    }
    likelihood(t - tau) = 1.0 / sqrt(2 * 3.14159 * var) * exp(-pow(y(t) - mean, 2) / (2 * var));
  }
  return likelihood;
}

// Returns p(k = j | theta, y_{1:T}) and p(s_T = 1 | theta, K, y_{1:T})up to proportionality
// y: Data
// thetaK: parameters of dynamic model j
// thetaC: parameters of constant model
// order: arima lags
// pS1: p(s_1 = 1 | everything), or eta
// rho: p01 and p10 in a vector
// prior p(k = j) 
// T: size of data (excluding lags)
// [[Rcpp::export]]
vec probSKHF (vec y, vec x, vec thetaD, vec thetaC, vec order, double pS1, vec rho, double prior, int T){
  int tau = y.n_elem - T;
  vec eta1 = arLikelihood(y, x, T, thetaD, order);

  double logdens = log(prior);
  
  for(int t = tau; t < y.n_elem; ++t){
    double eta0 = 1.0 / sqrt(2 * 3.14159 * exp(thetaC(0))) * exp(-pow(y(t) - thetaC(1), 2) / (2 * exp(thetaC(0))));
    
    double likelihood0 = (1 - pS1) * (1 - rho(0)) * eta0 + pS1 * rho(1) * eta0;
    double likelihood1 = pS1 * (1 - rho(1)) * eta1(t - tau) + (1 - pS1) * rho(0) * eta1(t - tau);
    
    logdens += log(likelihood0 + likelihood1);
    pS1 = likelihood1 / (likelihood0 + likelihood1);
  }
  return {logdens, pS1};
}

// Returns p(k = j | theta, y) up to proportionality for non HF models
// y: Data
// theta: parameters of dynamic model j
// order: arima lags
// prior p(k = j) 
// T: size of data (excluding lags)
// [[Rcpp::export]]
double probK (vec y, vec x, vec theta, vec order, double prior, int T){
  int tau = y.n_elem - T;
  double logdens = log(prior);
  double var = exp(theta(0)), mu = theta(1), beta = theta(2);
  for(int t = tau; t < y.n_elem; ++t){
    double mean = mu + beta * x(t);
    for(int i = 0; i < order.n_elem; ++i){
      mean += theta(3 + i) * (y(t - order(i)) - mu - beta * x(t - order(i)));
    }
    logdens += -0.5 * theta(0) - pow(y(t) - mean, 2) / (2 * var);
  }

  return logdens;
}

//  y: Data, rows: t, columns: i
// x: temperature vector with rows matching y time period
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// pS1, prior probability of being in state 1
// priorMean: A matrix, first column: constant mean (pad with zeros), second column: rho mean, other columns dynamic mean 
// priorLinv: A cube, first slice: constant linv, second slice: rho linv, other: dynamic linv
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
struct electricitySwitching {
  const mat y;
  const vec x;
  const vec epsilon;
  const mat probK;
  const vec pS1;
  const mat priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  const bool uniformRho;
  electricitySwitching(const mat& yIn, const vec& xIn, const vec& epsIn, const mat& probKIn, const vec& pS1In, const mat& priorMeanIn, const cube& priorLinvIn, const vec& orderIn, const int& TnIn, const bool& uRhoin) :
   y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), pS1(pS1In), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn), uniformRho(uRhoin)  {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int numPars = 3 + order.n_elem; // Parameters = Logvar, mean, beta + order parameters
    int lambdaPerK = numPars * (numPars + 1); // Size of variance matrix for each dynamic model
    
    
    Matrix<T, Dynamic, 1> thetaC(2); // Create the constant varaiance and mean
    thetaC(0) = lambda(0) + lambda(2) * epsilon(0);
    thetaC(1) = lambda(1) + lambda(3) * epsilon(0) + lambda(4) * epsilon(1);
    
    T logdetJ = thetaC(0) + log(fabs(lambda(2))) + log(fabs(lambda(4))); // Add the relevant terms to log det j as we go
    
    // Create the unnormalised rho parameters, then apply logit to normalise
    Matrix<T, Dynamic, 1> rhoUnnorm(2), rho(2);
    rhoUnnorm(0) = lambda(5) + lambda(7) * epsilon(2);
    rhoUnnorm(1) = lambda(6) + lambda(8) * epsilon(2)  + lambda(9) * epsilon(3);
    rho(0) = 1.0 / (1 + exp(-rhoUnnorm(0)));
    rho(1) = 1.0 / (1 + exp(-rhoUnnorm(1)));
    
    // Add to log determinant
    logdetJ += log(fabs(lambda(7))) + log(fabs(lambda(9))) - rhoUnnorm(0) - rhoUnnorm(1) - 
      2 * log(exp(rhoUnnorm(0)) + 1)  -  2 * log(exp(rhoUnnorm(1)) + 1); 
    
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    Matrix<T, Dynamic, Dynamic> thetaJ (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        thetaJ(i, k) = lambda(10 + k * lambdaPerK + i);
        for(int j = 0; j <= i; ++j){
          thetaJ(i, k) += lambda(10 + k * lambdaPerK + numPars * (i + 1) + j) * epsilon(4 + k*numPars + j);
          if(j == i){
            logdetJ += log(fabs(lambda(10 + k * lambdaPerK + numPars * (i + 1) + j))); // add log(L_ii) to log det
          }
        }
        if(i == 0){
          logdetJ += thetaJ(i, k); // Add log variance to log det
        }
      }
    }
    
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    T prior = 0;
    Matrix<T, Dynamic, 1> kernelCons(2);
    kernelCons.fill(0);
    
    for(int i = 0; i < 2; ++i){
      for(int j = 0; j <= i; ++j){
        kernelCons(i) += (thetaC(j) - priorMean(j, 0)) * priorLinv(i, j, 0);
      }
      prior += - 0.5 * pow(kernelCons(i), 2);
    }
    
    // Rho has a uniform prior on the first fit, but has a MVN prior for updates
    if(!uniformRho){
      kernelCons.fill(0);
      for(int i = 0; i < 2; ++i){
        for(int j = 0; j <= i; ++j){
          kernelCons(i) += (rhoUnnorm(j) - priorMean(j, 1)) * priorLinv(i, j, 1);
        }
        prior += - 0.5 * pow(kernelCons(i), 2);
      }
    }
    
    // Add the dynamic model priors, each independent normal (not identitcal)
    Matrix<T, Dynamic, Dynamic> kernelDyn (numPars, K);
    kernelDyn.fill(0);
    
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        for(int j = 0; j <= i; ++j){
          kernelDyn(i, k) += (thetaJ(j, k) - priorMean(j, k+2)) * priorLinv(i, j, k+2);
        }
        prior += - 0.5 * pow(kernelDyn(i, k), 2);
      }
    }

    // Calculate the log likelihood
    T logLik = 0;
    Matrix<T, Dynamic, 1> pS1vec(N);
    
    for(int i = 0; i < N; ++i){
      pS1vec(i) = pS1(i);
      // Starting point is tau, will be part way through the data to deal with lagged variables
      int tau = y.n_rows - Tn;
      // For each smart meter we create eta1, the likelihood under the dynamic model as a sum over k of p(k) * p(y | theta, k)
      Matrix<T, Dynamic, 1> eta1(Tn);
      eta1.fill(0);
      for(int k = 0; k < K; ++k){
        
        // components of the dynamic model
        T var = exp(thetaJ(0, k)), mu = thetaJ(1, k), beta = thetaJ(2, k);
        
        // yt ~ N(beta * x + sum_l (theta * (y_t-l - beta*x_t-l)), var)
        for(int t = tau; t < y.n_rows; ++t){
          T mean = mu + beta * x(t);
          for(int l = 0; l < numPars - 3; ++l){
            mean += thetaJ(3 + l, k) * (y(t - order(l), i) - mu - beta * x(t - order(l)));
          }
          eta1(t - tau) += probK(k, i) / sqrt(2 * 3.14159 * var) * exp(-pow(y(t, i) - mean, 2) / (2 * var));
        }      
      }
      
      // Hamilton Filter recursion
      for(int t = tau; t < y.n_rows; ++t){
        T eta0 = 1.0 / sqrt(2 * 3.14159 * exp(thetaC(0))) * exp(-pow(y(t, i) - thetaC(1), 2) / (2 * exp(thetaC(0))));
        
        T lik0 = (1 - pS1vec(i)) * (1 - rho(0)) * eta0 + pS1vec(i) * rho(1) * eta0;
        T lik1 = pS1vec(i) * (1 - rho(1)) * eta1(t - tau) + (1 - pS1vec(i)) * rho(0) * eta1(t - tau);
        logLik += log(lik0 + lik1);
        pS1vec(i) = lik1 / (lik0 + lik1);
      }
    }
   
    return prior + logLik + logdetJ;
  }
};

// [[Rcpp::export]]
Rcpp::List elecSwitch(mat y, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec x, mat probK, vec pS1, mat priorMean, cube priorLinv, vec order, int Tn, bool uniformRho) {
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  electricitySwitching p(y, x, epsilon, probK, pS1, priorMean, priorLinv, order, Tn, uniformRho);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);

  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

//  y: Data, rows: t, columns: i
// x: temperature vector with rows matching y time period
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// prior: All of the prior components to deal with, 0: dynamic mean (matrix. one column per dynamic model), 1: dynamic linv (cube)
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
struct electricityStandard{
  const mat y;
  const vec x;
  const vec epsilon;
  const mat probK;
  const mat priorMean;
  const cube priorLinv;
  const vec order;
  const int Tn;
  electricityStandard(const mat& yIn, const vec& xIn, const vec& epsIn, const mat& probKIn, const mat& priorMeanIn, const cube& priorLinvIn, const vec& orderIn, const int& TnIn) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), priorMean(priorMeanIn), priorLinv(priorLinvIn), order(orderIn), Tn(TnIn)  {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int numPars = 3 + order.n_elem; // Parameters = Logvar, mean, beta + order parameters
    int lambdaPerK = numPars * (numPars + 1); // Size of parameter vector for each dynamic model
    
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    T logdetJ = 0, prior = 0, logLik = 0;
    Matrix<T, Dynamic, Dynamic> thetaJ (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        thetaJ(i, k) = lambda(k * lambdaPerK + i);
        for(int j = 0; j <= i; ++j){
          thetaJ(i, k) += lambda(k * lambdaPerK + numPars * (i + 1) + j) * epsilon(k*numPars + j);
          if(j == i){
            logdetJ += log(fabs(lambda(k * lambdaPerK + numPars * (i + 1) + j))); // add log(L_ii) to log det
          }
        }
        if(i == 0){
          logdetJ += thetaJ(i, k); // Add log variance to log det
        }
      }
    }

    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    // Add the dynamic model priors, each independent normal (not identitcal)
    Matrix<T, Dynamic, Dynamic> kernelDyn (numPars, K);
    kernelDyn.fill(0);
    
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        for(int j = 0; j <= i; ++j){
          kernelDyn(i, k) += (thetaJ(j, k) - priorMean(j, k)) * priorLinv(i, j, k);
        }
        prior += - 0.5 * pow(kernelDyn(i, k), 2);
      }
    }
  
    // Calculate the log likelihood
    // Starting point is tau, will be part way through the data as the first data points are lags.
    int tau = y.n_rows - Tn;
    
    for(int i = 0; i < N; ++i){
      for(int t = tau; t < y.n_rows; ++t){
        T likelihood = 0;
        for(int k = 0; k < K; ++ k){
          T mean = thetaJ(1, k) + thetaJ(2, k) * x(t);
          for(int l = 0; l < order.n_elem; ++l){
            mean += thetaJ(3 + l, k) * (y(t - order(l), i) - thetaJ(1, k) - thetaJ(2, k) * x(t - order(l)));
          }
          likelihood += probK(k, i) / sqrt(2 * 3.14159 * exp(thetaJ(0, k))) * exp(-pow(y(t, i) - mean, 2) / (2 * exp(thetaJ(0, k))));
        }
        logLik += log(likelihood);
      }
    }
    return prior + logLik + logdetJ;
  }
};

// [[Rcpp::export]]
Rcpp::List elecStd(mat y, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec x, mat probK, mat priorMean, cube priorLinv, vec order, int Tn) {
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  electricityStandard p(y, x, epsilon, probK, priorMean, priorLinv, order, Tn);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

// [[Rcpp::export]]
mat calcProb(cube pk, cube ps){
  int M = pk.n_slices;
  int N = pk.n_cols;
  int K = pk.n_rows;
  
  mat out(M, N, fill::zeros);
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      out(i, j) = as_scalar(pk.slice(i).col(j).t() * ps.slice(i).col(j));
    }
  }
  return out;
}


// Input data, theta draws and other relevant informaiton
// Will evaluate p(ki = j) and forecast
// [[Rcpp::export]]
Rcpp::List forecastStandard (vec y, vec x, mat theta, vec order, vec priorK, mat fcVar, vec support, int Tn){
  int M = support.n_elem;
  int H = fcVar.n_rows;
  int K = fcVar.n_cols;
  int T = y.n_rows - H - 1;
  mat density(M, H, fill::zeros);
  vec pk(K), pkNorm(K);

  // Evaluate p(k_i = j)
  if(K == 1){
      pkNorm.fill(1);
  } else {
    for(int ki = 0; ki < K; ++ki){
      pk(ki) = probK(y(span(0, T)), x(span(0, T)), theta.col(ki), order, priorK(ki), Tn);
    }
    
    // Normalise the weights
    double maxP = max(pk);
    for(int ki = 0; ki < K; ++ki){
      pk(ki) -= maxP;
    }
    pkNorm = exp(pk) / sum(exp(pk));
  }
  // Forecast each model
  for(int ki = 0; ki < K; ++ki){
    
    vec ylag = {y(T), y(T-1), y(T-2)};
    double mu = theta(1, ki), beta = theta(2, ki) ;
    for(int h = 1; h <= H; ++h){
      
      double mean = mu + beta * x(T + h);
      for(int l = 0; l < order.n_elem; ++l){
        if(l < 3){
          mean += theta(3 + l, ki) * (ylag(l) - mu - beta * x(T + h - order(l)));
        } else {
          mean += theta(3 + l, ki) * (y(T + h - order(l)) - mu - beta * x(T + h - order(l)));
        }
      }
      
      for(int l = 2; l > 0; --l){
        ylag(l) = ylag(l - 1);
      }
      ylag(0) = mean;
      for(int m = 0; m < M; ++m){
        density(m, h-1) = 1.0 / sqrt(2 * 3.14159 * fcVar(h - 1, ki)) * exp(-pow(support(m) - mean, 2) / (2 * fcVar(h-1, ki))) * pkNorm(ki);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("density") = density,
                            Rcpp::Named("pk") = pkNorm);
  
}

// return the forecast density matrix (over the support grid and values of H) for a given p(S = 1 | y_{1:T}) and draw of k / theta.
// Will first evaluate the pk / ps probabilities, and then use these through a hamilton Filter based forecast
// Inputs are y, from the start of sample until the end of the forecasts (last values unused)
// x, temperature for the same period, will use the last values for forecasting
// ThetaC, the log variance and mean of the constant model
// Rho, the HF switching probabilities,
// ThetaD, a matrix of parameters from the dynamic models,
// Tn, the period evaluated for pk and ps
// and other standard inputs
// PS1 has three distinct time periods, the prior value, the posterior (p(S_T = 1 | y_{1:T})) and forecast values p(S_T+h  = 1 | y_1:T) for each group
// Forecasts are relatively short term, so seasonal AR components can use the true lagged value instead of a y-hat
// so ylag cycles through \hat{y}_{t-i} for i = 1, 2, 3 
// fcVar is the AR model prediction variance as a function of sigma^2 and the non-seasonal AR components, can be calculated ahead of time.
// [[Rcpp::export]]
Rcpp::List forecastHF(vec y, vec x, vec thetaC, vec rho, mat thetaD, mat fcVar, vec order, double pS1prior, vec priorK, vec support, int Tn){
  int M = support.n_elem;
  int H = fcVar.n_rows;
  int K = fcVar.n_cols;
  int T = y.n_rows - H - 1;
  mat density(M, H, fill::zeros);
  vec pk(K), pkNorm(K), ps(K);
  double pS1posterior;
  
  // Evaluate p(k_i = j) and p(S_T = 1)
  for(int ki = 0; ki < K; ++ki){
    vec probs = probSKHF(y(span(0, T)), x(span(0, T)), thetaD.col(ki), thetaC, order, pS1prior, rho, priorK(ki), Tn);
    pk(ki) = probs(0);
    ps(ki) = probs(1);
  }
  // Normalise the weights
  double maxP = max(pk);
  for(int ki = 0; ki < K; ++ki){
    pk(ki) -= maxP;
  }
  pkNorm = exp(pk) / sum(exp(pk));
  pS1posterior = as_scalar(pkNorm.t() * ps);

  // Forecast each model
  for(int ki = 0; ki < K; ++ki){
    double pS1FC = pS1posterior;
    vec ylag = {y(T), y(T-1), y(T-2)};
    double mu = thetaD(1, ki), beta = thetaD(2, ki) ;
    for(int h = 1; h <= H; ++h){
      pS1FC = pS1FC * (1 - rho(1)) + (1 - pS1FC) * rho(0);
      
      double mean = mu + beta * x(T + h);
      for(int l = 0; l < order.n_elem; ++l){
        if(l < 3){
          mean += thetaD(3 + l, ki) * (ylag(l) - mu - beta * x(T + h - order(l)));
        } else {
          mean += thetaD(3 + l, ki) * (y(T + h - order(l)) - mu - beta * x(T + h - order(l)));
        }
      }
      for(int l = 2; l > 0; --l){
        ylag(l) = ylag(l - 1);
      }
      ylag(0) = mean;
      for(int m = 0; m < M; ++m){
        density(m, h-1) = (pS1FC / sqrt(2 * 3.14159 * exp(thetaC(0))) * exp(-pow(support(m) - thetaC(1), 2) / (2 * exp(thetaC(0)))) +
          (1 - pS1FC) / sqrt(2 * 3.14159 * fcVar(h - 1, ki)) * exp(-pow(support(m) - mean, 2) / (2 * fcVar(h-1, ki)))) * pkNorm(ki);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("density") = density,
                            Rcpp::Named("pk") = pkNorm,
                            Rcpp::Named("ps") = pS1posterior);
  
  
}

// Similar to the above switching model but with a different parameter for each half hour period (ie a restricted VAR).
// The dynamic model MVN approximaiton now have diagonal covariance, param: Mean / Log(Sd)
// y: Data, rows: t, columns: i
// x: temperature vector with rows matching y time period
// epsilon ~ Z
// probK, prior probability of k_i = j, rows: i, columns: j
// pS1, prior probability of being in state 1
// priorMean: A matrix, first column: constant mean (pad with zeros), second: rho mean, other columns dynamic mean 
// priorLinv, a cube of the constant model + rho L inverse
// priorSd: A matrix a dynamic model SD
// order, a vector of the lags of the dynamic model
// Tn: Size of current batch in terms of T (T is reserved in structures for stan::math::var)
struct electricitySwitchingVAR {
  const mat y;
  const vec x;
  const vec epsilon;
  const mat probK;
  const vec pS1;
  const mat priorMean;
  const cube priorLinv;
  const mat priorSd;
  const vec order;
  const int Tn;
  const bool uniformRho;
  electricitySwitchingVAR(const mat& yIn, const vec& xIn, const vec& epsIn, const mat& probKIn, const vec& pS1In, const mat& priorMeanIn, 
                          const  cube& priorLinvIn, const mat& priorSdIn, const vec& orderIn, const int& TnIn, const bool& uRhoin) :
    y(yIn), x(xIn), epsilon(epsIn), probK(probKIn), pS1(pS1In), priorMean(priorMeanIn),
    priorLinv(priorLinvIn), priorSd(priorSdIn), order(orderIn), Tn(TnIn), uniformRho(uRhoin)  {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = y.n_cols; // Number of units
    int K = probK.n_rows; // Number of dynamic models
    int numPars = 48 * (3 + order.n_elem); // Parameters = Logvar, mean, beta + order parameters
  
    Matrix<T, Dynamic, 1> thetaC(2); // Create the constant varaiance and mean
    thetaC(0) = lambda(0) + lambda(2) * epsilon(0);
    thetaC(1) = lambda(1) + lambda(3) * epsilon(0) + lambda(4) * epsilon(1);
    
    T logdetJ = thetaC(0) + log(fabs(lambda(2))) + log(fabs(lambda(4))); // Add the relevant terms to log det j as we go
    
    // Create the unnormalised rho parameters, then apply logit to normalise
    Matrix<T, Dynamic, 1> rhoUnnorm(2), rho(2);
    rhoUnnorm(0) = lambda(5) + lambda(7) * epsilon(2);
    rhoUnnorm(1) = lambda(6) + lambda(8) * epsilon(2)  + lambda(9) * epsilon(3);
    rho(0) = 1.0 / (1 + exp(-rhoUnnorm(0)));
    rho(1) = 1.0 / (1 + exp(-rhoUnnorm(1)));
    
    // Add to log determinant
    logdetJ += log(fabs(lambda(7))) + log(fabs(lambda(9))) - rhoUnnorm(0) - rhoUnnorm(1) - 
      2 * log(exp(rhoUnnorm(0)) + 1)  -  2 * log(exp(rhoUnnorm(1)) + 1); 
    
    // Create theta for the j'th dynamic model, rows: Different parameters, cols: Different models
    Matrix<T, Dynamic, Dynamic> thetaJ (numPars, K);
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        thetaJ(i, k) = lambda(10 + k * 2 * numPars + i) + exp(lambda(10 + (k * 2 + 1) * numPars + i)) * epsilon(4 + k * numPars + i);
        logdetJ += lambda(10 + (2 * k + 1) * numPars + i); // add log(L_ii) to log det
        if(i < 48){
          logdetJ += thetaJ(i, k); // Add log variance to log det
        }
      }
    }
    // Evaluate log(p(theta)), starting with log(var)_c, mu_c ~ N
    T prior = 0;
    Matrix<T, Dynamic, 1> kernelCons(2);
    kernelCons.fill(0);
    
    for(int i = 0; i < 2; ++i){
      for(int j = 0; j <= i; ++j){
        kernelCons(i) += (thetaC(j) - priorMean(j, 0)) * priorLinv(i, j, 0);
      }
      prior += - 0.5 * pow(kernelCons(i), 2);
    }
    
    // Rho has a uniform prior on the first fit, but has a MVN prior for updates
    if(!uniformRho){
      kernelCons.fill(0);
      for(int i = 0; i < 2; ++i){
        for(int j = 0; j <= i; ++j){
          kernelCons(i) += (rhoUnnorm(j) - priorMean(j, 1)) * priorLinv(i, j, 1);
        }
        prior += - 0.5 * pow(kernelCons(i), 2);
      }
    }
    
    // Add the dynamic model priors, each independent normal (not identitcal)
    for(int k = 0; k < K; ++k){
      for(int i = 0; i < numPars; ++i){
        prior += -0.5 * pow((thetaJ(i, k) - priorMean(i, k+2)) / priorSd(i, k), 2);
      }
    }
    
    // Calculate the log likelihood
    T logLik = 0;
    Matrix<T, Dynamic, 1> pS1vec(N);
    for(int i = 0; i < N; ++i){
      pS1vec(i) = pS1(i);
      // Starting point is tau, will be part way through the data to deal with lagged variables
      int tau = y.n_rows - Tn;
      // For each smart meter we create eta1, the likelihood under the dynamic model as a sum over k of p(k) * p(y | theta, k)
      Matrix<T, Dynamic, 1> eta1(Tn);
      eta1.fill(0);
      for(int k = 0; k < K; ++k){
        
        // yt ~ N(beta * x + sum_l (theta * (y_t-l - beta*x_t-l)), var)
        for(int t = tau; t < y.n_rows; ++t){
          int halfhour = t % 48;
          // components of the dynamic model
          T var = exp(thetaJ(halfhour, k)), mu = thetaJ(48 + halfhour, k), beta = thetaJ(96 + halfhour, k);
          T mean = mu + beta * x(t);
          for(int l = 0; l < order.n_elem ; ++l){
            mean += thetaJ((3 + l)*48 + halfhour, k) * (y(t - order(l), i) - mu - beta * x(t - order(l)));
          }
          eta1(t - tau) += probK(k, i) / sqrt(2 * 3.14159 * var) * exp(-pow(y(t, i) - mean, 2) / (2 * var));
        }      
      }
      
      // Hamilton Filter recursion
      for(int t = tau; t < y.n_rows; ++t){
        T eta0 = 1.0 / sqrt(2 * 3.14159 * exp(thetaC(0))) * exp(-pow(y(t, i) - thetaC(1), 2) / (2 * exp(thetaC(0))));
        
        T lik0 = (1 - pS1vec(i)) * (1 - rho(0)) * eta0 + pS1vec(i) * rho(1) * eta0;
        T lik1 = pS1vec(i) * (1 - rho(1)) * eta1(t - tau) + (1 - pS1vec(i)) * rho(0) * eta1(t - tau);
        logLik += log(lik0 + lik1);
        pS1vec(i) = lik1 / (lik0 + lik1);
      }
    }
    
    return prior + logLik + logdetJ;
  }
};

// [[Rcpp::export]]
Rcpp::List elecSwitchVAR(mat y, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec x, mat probK, vec pS1, mat priorMean, cube priorLinv, mat priorSd, vec order, int Tn, bool uniformRho) {
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  electricitySwitchingVAR p(y, x, epsilon, probK, pS1, priorMean, priorLinv, priorSd, order, Tn, uniformRho);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}
