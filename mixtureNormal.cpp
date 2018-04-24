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

// Sobol generation and shuffling

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

// Pieces for the mixture approximations are shared between both prior specifications

struct Qlogdens {
  const vec theta;
  const int mix;
  Qlogdens(const vec& thetaIn, const int& mixIn) :
    theta(thetaIn), mix(mixIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::lgamma;
    
    
    Matrix<T, Dynamic, 1> dets(mix);
    for(int k = 0; k < mix; ++k){
      dets(k) = exp(lambda(4*k + 4*mix));
      for(int i = 1; i < 4; ++i){
        dets(k) *= exp(lambda(4*mix + 4*k + i));
      }
      dets(k) = 1.0 / dets(k);
    }
    
    Matrix<T, Dynamic, 1> kernel(mix);
    kernel.fill(0);
    
    for(int k = 0; k < mix; ++k){
      for(int i = 0; i < 4; ++i){
        kernel(k) += pow((theta(i) - lambda(k*4 + i)) / exp(lambda(4*mix + 4*k + i)), 2);
      }
    }
    
    Matrix<T, Dynamic, 1> weights(mix);
    T sumExpZ = 0;
    for(int k = 0; k < mix; ++k){
      weights(k) = exp(lambda(2*4*mix + k));
      sumExpZ += weights(k);
    }
    weights /= sumExpZ;
    
    T density = 0;
    for(int k = 0; k < mix; ++k){
      density += weights(k) * dets(k) * pow(6.283185, -3) *  exp(-0.5 * kernel(k));
    }
    
    T logMVN = log(density);
    
    return logMVN;
  }
};

double pLogDens(mat y, vec theta, vec K, mat mean, cube SigInv, vec weights, vec dets){
  int N = y.n_cols;
  int T = y.n_rows;
  
  int mix = weights.n_rows;
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  double logPrior = log(prior);
  
  double logLik = 0;
  for(int i = 0; i < N; ++i){
    double mu = theta(2 + K(i));
    double var = exp(theta(K(i)));
    for(int t = 0; t < T; ++t){
      logLik += -0.5 * log(var) - pow(y(t, i) - mu, 2) / (2 * var);
    }
  }
  return logPrior + logLik;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List mixtureNormal(mat data, Rcpp::NumericMatrix lambdaIn, vec theta, vec K, mat mean, cube SigInv, vec weights, vec dets, int mix){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  Qlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDens(data, theta, K, mean, SigInv, weights, dets);
  double elbo = logp - qEval;
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}


// Metropolis Hastings Log Density Evaluation
// [[Rcpp::export]]
double MNLogDens(mat y, vec theta, vec K, vec mean, mat VarInv){
  int N = y.n_cols;
  int T = y.n_rows;
  
  double logPrior = - as_scalar((theta - mean).t() * VarInv * (theta - mean));
  
  double logLik = 0;
  for(int i = 0; i < N; ++i){
    double mu = theta(2 + K(i));
    double var = exp(theta(K(i)));
    for(int t = 0; t < T; ++t){
      logLik += - 0.5 * log(var) - pow(y(t, i) - mu, 2) / (2 * var);
    }
  }
  return logLik + logPrior;
}

// [[Rcpp::export]]
double probK1(vec y, vec theta, vec piPrior){
  double p0 = log(boost::math::beta(piPrior(0), piPrior(1) + 1));
  double p1 = log(boost::math::beta(piPrior(0) + 1, piPrior(1)));
  
  for(int t = 0; t < y.n_rows; ++t){
    p0 += - 0.5 * theta(0) - pow(y(t) - theta(2), 2) / (2 * exp(theta(0)));
    p1 += - 0.5 * theta(1) - pow(y(t) - theta(3), 2) / (2 * exp(theta(1)));
  }
  return exp(p1) / (exp(p0) + exp(p1));
}

// [[Rcpp::export]]
double MNLogDens2(mat y, vec theta, vec probK, vec mean, mat VarInv){
  int N = y.n_cols;
  int T = y.n_rows;
  
  double logPrior = - 0.5 * as_scalar((theta - mean).t() * VarInv * (theta - mean));
  
  double logLik = 0;
  for(int i = 0; i < N; ++i){
    for(int t = 0; t < T; ++t){
      logLik += log(
        (1 - probK(i)) * pow(2 * 3.141598 * exp(theta(0)), -0.5) * exp(-pow(y(t, i) - theta(2), 2) / 2 * exp(theta(0))) + 
        probK(i) * pow(2 * 3.141598 * exp(theta(1)), -0.5) * exp(-pow(y(t, i) - theta(3), 2) / 2 * exp(theta(1)))
      );
    }
  }
  return logLik + logPrior;
}
