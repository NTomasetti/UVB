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

// VB Gradient Estimator Components

struct SV1 {
  const double x;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  SV1(const double& xIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn) :
    x(xIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;

    Matrix<T, Dynamic, 1> theta(3);
    for(int i = 0; i < 3; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(3*(i+1) + j) * epsilon(j);
      }
    }
    T z1 = lambda(12) + lambda(13) * epsilon(3);
    
    T sig2 = exp(theta(0)), alpha = theta(1), beta = theta(2); 
    // Evaluate log(p(theta))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(3);
    kernel.fill(0);
    for(int i = 0; i < 3; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }

    // Evaluate log det J
    T logdetJ = theta(0) + log(fabs(lambda(3))) + log(fabs(lambda(7))) + log(fabs(lambda(11))) + log(fabs(lambda(13)));

    
    // Evaluate log likelihood
    T unconVar = sig2 / (1 - pow(beta, 2));
    T pZ1 = - 0.5 * log(unconVar) - pow(z1 - alpha, 2) / (2 * unconVar);
    T pX1 = - 0.5 * z1 - pow(x, 2) / (2 * exp(z1));
    
  
    return prior + logdetJ + pZ1 + pX1;
  }
};

struct SVt {
  const double x;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  const vec zStats;
  SVt(const double& xIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn, const vec& zStatsIn) :
    x(xIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn), zStats(zStatsIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    
    Matrix<T, Dynamic, 1> theta(3);
    for(int i = 0; i < 3; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(3*(i+1) + j) * epsilon(j);
      }
    }
    
    T zt = lambda(12) + lambda(13) * epsilon(3);
    
    T sig2 = exp(theta(0)), alpha = theta(1), beta = theta(2); 
    // Evaluate log(p(theta))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(3);
    kernel.fill(0);
    for(int i = 0; i < 3; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }
    // Evaluate log det J
    T logdetJ = theta(0) + log(fabs(lambda(3))) + log(fabs(lambda(7))) + log(fabs(lambda(11))) + log(fabs(lambda(13)));
    
    // Evaluate log likelihood
    T zMean = alpha + beta * (zStats(0)- alpha);
    T zVar = sig2 + pow(beta * zStats(1), 2);
    T pZ = - 0.5 * log(zVar) - pow(zt - zMean, 2) / (2 * zVar);
    T pX = - 0.5 * zt - pow(x, 2) / (2 * exp(zt));
    
    return prior + logdetJ + pZ + pX;
  }
};

struct SVtSmooth {
  const double x;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  const vec zMean;
  const vec zSd;
  SVtSmooth(const double& xIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn, const vec& zMeanIn, const vec& zSdIn) :
    x(xIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn), zMean(zMeanIn), zSd(zSdIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    
    Matrix<T, Dynamic, 1> theta(3);
    for(int i = 0; i < 3; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(3*(i+1) + j) * epsilon(j);
      }
    }
    int N = zMean.n_rows;
    Matrix<T, Dynamic, 1> z(N + 1);
    for(int i = 0; i < N+1; ++i){
      z(i) = lambda(12 + i) + lambda(13 + N + i) * epsilon(3 + i);
    }
    T sig2 = exp(theta(0)), alpha = theta(1), beta = theta(2); 
    // Evaluate log(p(theta))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(3);
    kernel.fill(0);
    for(int i = 0; i < 3; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }
    // Evaluate log det J
    T logdetJ = theta(0) + log(fabs(lambda(3))) + log(fabs(lambda(7))) + log(fabs(lambda(11)));
    for(int i = 0; i < N+1; ++i){
      logdetJ += log(fabs(lambda(13 + N + i)));
    }
    // Evaluate log likelihood
    T ztMean = alpha + beta * (zMean(N-1)- alpha);
    T ztVar = sig2 + pow(beta * zSd(N-1), 2);
    T pX = - 0.5 * z(N) - pow(x, 2) / (2 * exp(z(N)));
    T pZ = 0;
    for(int i = 0; i < N; ++i){
      pZ += - log(zSd(i)) - pow(z(i) - zMean(i), 2) / (2 * pow(zSd(i), 2));
    }
    pZ += - 0.5 * log(ztVar) - pow(z(N) - ztMean, 2) / (2 * ztVar);
    return prior + logdetJ + pZ + pX;
  }
};

// [[Rcpp::export]]
Rcpp::List SV1diff(double data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff

  SV1 p(data, epsilon, mean, Linv);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

// [[Rcpp::export]]
Rcpp::List SVtdiff(double data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, vec zStats){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  
  SVt p(data, epsilon, mean, Linv, zStats);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

// [[Rcpp::export]]
Rcpp::List SVtSmoothdiff(double data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, vec zMean, vec zSd){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  
  SVtSmooth p(data, epsilon, mean, Linv, zMean, zSd);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}


// Pieces for the mixture approximations are shared between both prior specifications

struct mixQlogdens {
  const vec theta;
  const int mix;
  mixQlogdens(const vec& thetaIn, const int& mixIn) :
    theta(thetaIn), mix(mixIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt;
    
    int dim = theta.n_rows;
    
    Matrix<T, Dynamic, 1> dets(mix);
    for(int k = 0; k < mix; ++k){
      dets(k) = exp(lambda(dim*k + dim*mix));
      for(int i = 1; i < dim; ++i){
        dets(k) *= exp(lambda(dim*mix + dim*k + i));
      }
      dets(k) = 1.0 / dets(k);
    }
    
    Matrix<T, Dynamic, 1> kernel(mix);
    kernel.fill(0);
    
    for(int k = 0; k < mix; ++k){
      for(int i = 0; i < dim; ++i){
        kernel(k) += pow((theta(i) - lambda(k*dim + i)) / exp(lambda(dim*mix + dim*k + i)), 2);
      }
    }
    
    Matrix<T, Dynamic, 1> pi(mix);
    T sumExpZ = 0;
    for(int k = 0; k < mix; ++k){
      pi(k) = exp(lambda(2*dim*mix + k));
      sumExpZ += pi(k);
    }
    pi /= sumExpZ;
    
    T density = 0;
    for(int k = 0; k < mix; ++k){
      density += pi(k) * dets(k) * pow(6.283185, -3) *  exp(-0.5 * kernel(k));
    }
    
    return log(density);
  }
};

double pLogDensMix(double x, vec theta, double zt, mat mean, cube SigInv, vec dets, vec weights, vec zStats){
  int mix = weights.n_rows;
  // Constrained Positive
  double sig2 = std::exp(theta(0)), alpha = theta(1), beta = theta(2); 
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * 
      exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  
  
  // Evaluate log likelihood
  double pZ = 0;
  for(int k = 0; k < mix; ++k){
    double zMean = alpha + beta * (zStats(2*k) - alpha);
    double zVar = sig2 + pow(beta * zStats(2*k + 1), 2);
    pZ += weights(k) / sqrt(2 * 3.141598 * zVar) * exp(-pow(zt - zMean, 2) / (2 * zVar));
  }
  double pX = - 0.5 * zt - pow(x, 2) / (2 * exp(zt));
  return std::log(prior) + log(pZ) + pX;
}

double pLogDensSingle(double x, vec theta, double z1, vec mean, mat Linv){
  // Constrained Positive
  double sig2 = std::exp(theta(0)), alpha = theta(1), beta = theta(2);
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  
  // Evaluate log(p(theta))
  double prior = 0;
  mat sigInv = Linv.t() * Linv;
  prior = - 0.5 * as_scalar((theta - mean).t() * sigInv * (theta - mean));
 
  // Evaluate log likelihood
  double unconVar = sig2 / (1 - pow(beta, 2));
  double pZ1 = - 0.5 * log(unconVar) - pow(z1 - alpha, 2) / (2 * unconVar);
  double pX1 = - 0.5 * z1 - pow(x, 2) / (2 * exp(z1));
  
  return prior + pZ1 + pX1;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List SV1mixdiff(double data, Rcpp::NumericMatrix lambdaIn, vec theta, vec mean, mat Linv, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  vec thetaSub = {theta(0), theta(1), theta(2)};
  double z1 = theta(3);
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensSingle(data, thetaSub, z1, mean, Linv);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List SVtmixdiff(double data, Rcpp::NumericMatrix lambdaIn, vec theta, mat mean, cube SigInv, vec dets, vec weights, vec zStats, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  vec thetaSub = {theta(0), theta(1), theta(2)};
  double zt = theta(3);
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensMix(data, thetaSub, zt, mean, SigInv, dets, weights, zStats);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

struct LG1 {
  const double x;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  LG1(const double& xIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn) :
    x(xIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    
    Matrix<T, Dynamic, 1> theta(4);
    for(int i = 0; i < 4; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(4*(i+1) + j) * epsilon(j);
      }
    }
    T z1 = lambda(20) + lambda(21) * epsilon(4);
    
    T sig2X = exp(theta(0)), sig2Z = exp(theta(1)), alpha = theta(2), beta = theta(3); 
    // Evaluate log(p(theta))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(4);
    kernel.fill(0);
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }
    
    // Evaluate log det J
    T logdetJ = theta(0) + theta(1) + log(fabs(lambda(4))) + log(fabs(lambda(9))) + 
      log(fabs(lambda(14))) + log(fabs(lambda(19))) + log(fabs(lambda(21)));
    
    
    // Evaluate log likelihood
    T unconVar = sig2Z / (1 - pow(beta, 2));
    T pZ1 = - 0.5 * log(unconVar) - pow(z1 - alpha, 2) / (2 * unconVar);
    T pX1 = - 0.5 * theta(0) - pow(x - z1, 2) / (2 * sig2X);
    
    
    return prior + logdetJ + pZ1 + pX1;
  }
};

struct LGt {
  const double x;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  const vec zStats;
  LGt(const double& xIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn, const vec& zStatsIn) :
    x(xIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn), zStats(zStatsIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    
    Matrix<T, Dynamic, 1> theta(4);
    for(int i = 0; i < 4; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(4*(i+1) + j) * epsilon(j);
      }
    }
    T zt = lambda(20) + lambda(21) * epsilon(4);
    
    T sig2X = exp(theta(0)), sig2Z = exp(theta(1)), alpha = theta(2), beta = theta(3); 
    // Evaluate log(p(theta))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(4);
    kernel.fill(0);
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }
    
    // Evaluate log det J
    T logdetJ = theta(0) + theta(1) + log(fabs(lambda(4))) + log(fabs(lambda(9))) + 
      log(fabs(lambda(14))) + log(fabs(lambda(19))) + log(fabs(lambda(21)));
    
    
    // Evaluate log likelihood
    T zMean = alpha + beta * (zStats(0)- alpha);
    T zVar = sig2Z + pow(beta * zStats(1), 2);
    T pZ = - 0.5 * log(zVar) - pow(zt - zMean, 2) / (2 * zVar);
    T pX = - 0.5 * theta(0) - pow(x - zt, 2) / (2 * sig2X);
    
    return prior + logdetJ + pZ + pX;
  }
};

// [[Rcpp::export]]
Rcpp::List LG1diff(double data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  
  LG1 p(data, epsilon, mean, Linv);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

// [[Rcpp::export]]
Rcpp::List LGtdiff(double data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, vec zStats){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff
  
  LGt p(data, epsilon, mean, Linv, zStats);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(p, lambda, eval, gradP);
  
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

double LGpMix(double x, vec theta, double zt, mat mean, cube SigInv, vec dets, vec weights, vec zStats){
  int mix = weights.n_rows;
  // Constrained Positive
  double sigmaSqX = std::exp(theta(0)), sig2 = std::exp(theta(1)), alpha = theta(2), beta = theta(3); 
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * 
      exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  
  // Evaluate log likelihood
  double pZ = 0;
  for(int k = 0; k < mix; ++k){
    double zMean = alpha + beta * (zStats(2*k) - alpha);
    double zVar = sig2 + pow(beta * zStats(2*k + 1), 2);
    pZ += weights(k) / sqrt(2 * 3.141598 * zVar) * exp(-pow(zt - zMean, 2) / (2 * zVar));
  }
  double pX = - 0.5 * theta(0) - pow(x - zt, 2) / (2 * sigmaSqX);
  return std::log(prior) + log(pZ) + pX;
}

double LGpSingle(double x, vec theta, double z1, vec mean, mat Linv){
  // Constrained Positive
  double sigmaSqX = std::exp(theta(0)), sig2 = std::exp(theta(1)), alpha = theta(2), beta = theta(3); 
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  
  // Evaluate log(p(theta))
  double prior = 0;
  mat sigInv = Linv.t() * Linv;
  prior = - 0.5 * as_scalar((theta - mean).t() * sigInv * (theta - mean));
  
  // Evaluate log likelihood
  double unconVar = sig2 / (1 - pow(beta, 2));
  double pZ1 = - 0.5 * log(unconVar) - pow(z1 - alpha, 2) / (2 * unconVar);
  double pX1 = - 0.5 * theta(0) - pow(x - z1, 2) / (2 * sigmaSqX);
  
  return prior + pZ1 + pX1;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List LG1mixdiff(double data, Rcpp::NumericMatrix lambdaIn, vec theta, vec mean, mat Linv, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  vec thetaSub = {theta(0), theta(1), theta(2), theta(3)};
  double z1 = theta(3);
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = LGpSingle(data, thetaSub, z1, mean, Linv);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List LGtmixdiff(double data, Rcpp::NumericMatrix lambdaIn, vec theta, mat mean, cube SigInv, vec dets, vec weights, vec zStats, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  vec thetaSub = {theta(0), theta(1), theta(2), theta(3)};
  double zt = theta(4);
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = LGpMix(data, thetaSub, zt, mean, SigInv, dets, weights, zStats);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}


double ARpMix(vec x, vec theta, mat mean, cube SigInv, vec dets, vec weights, int lags){
  int mix = weights.n_rows;
  // Constrained Positive
  double sigmaSq = exp(theta(0)), mu = theta(1);
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * 
      exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  
  // Evaluate log likelihood
  double logLik = 0;
  for(int i = lags; i < x.n_elem; ++i){
    double kernel = x(i) - mu;
    for(int j = 1; j <= lags; ++j){
      kernel -= theta(1 + j) * (x(i - j) - mu);
    }
    logLik += -0.5 * theta(0) - pow(kernel, 2) / (2 * sigmaSq);
  }
  return log(prior) + logLik;
}

double ARp(vec x, vec theta, vec mean, mat Linv, int lags){
  // Constrained Positive
  double sigmaSq = exp(theta(0)), mu = theta(1);
  // Evaluate log(p(theta))
  double prior = 0;
  mat sigInv = Linv.t() * Linv;
  prior = - 0.5 * as_scalar((theta - mean).t() * sigInv * (theta - mean));
  
  // Evaluate log likelihood
  double logLik = 0;
  for(int i = lags; i < x.n_elem; ++i){
    double kernel = x(i) - mu;
    for(int j = 1; j <= lags; ++j){
      kernel -= theta(1 + j) * (x(i - j) - mu);
    }
    logLik += -0.5 * theta(0) - pow(kernel, 2) / (2 * sigmaSq);
  }
  return prior + logLik;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List ARpdiff(vec data, Rcpp::NumericMatrix lambdaIn, vec theta, vec mean, mat Linv, int lags = 2, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = ARp(data, theta, mean, Linv, lags);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List ARpdiffMix(vec data, Rcpp::NumericMatrix lambdaIn, vec theta, mat mean, cube SigInv, vec dets, vec weights, int lags = 2, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = ARpMix(data, theta, mean, SigInv, dets, weights, lags);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}




double LGpMixFixed(double x, double zt, vec zStats, vec weights){
  int mix = weights.n_rows;

    // Evaluate log likelihood
  double pZ = 0;
  for(int k = 0; k < mix; ++k){
    double zMean = -2  +  0.95 * (zStats(2*k) + 2);
    double zVar = 0.1 + pow(0.95 * zStats(2*k + 1), 2);
    pZ += weights(k) / sqrt(2 * 3.141598 * zVar) * exp(-pow(zt - zMean, 2) / (2 * zVar));
  }
  double pX = - 0.5 * log(0.1) - pow(x - zt, 2) / (2 * 0.1);
  return log(pZ) + pX;
}

double LGpSingleFixed(double x, double z1){

  // Evaluate log likelihood
  double unconVar = 0.1 / (1 - pow(0.95, 2));
  double pZ1 = - 0.5 * log(unconVar) - pow(z1 + 2, 2) / (2 * unconVar);
  double pX1 = - 0.5 * log(0.1) - pow(x - z1, 2) / (2 * 0.1);
  
  return pZ1 + pX1;
}

struct mixQSingle {
  const double theta;
  const int mix;
  mixQSingle(const double& thetaIn, const int& mixIn) :
    theta(thetaIn), mix(mixIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt;
    
    Matrix<T, Dynamic, 1> dets(mix);
    for(int k = 0; k < mix; ++k){
      dets(k) = 1.0 / exp(lambda(mix + k));
    }
    
    Matrix<T, Dynamic, 1> kernel(mix);
    kernel.fill(0);
    
    for(int k = 0; k < mix; ++k){
      kernel(k) = pow((theta - lambda(k)) / exp(lambda(mix + k)), 2);
    }
    
    Matrix<T, Dynamic, 1> pi(mix);
    T sumExpZ = 0;
    for(int k = 0; k < mix; ++k){
      pi(k) = exp(lambda(2*mix + k));
      sumExpZ += pi(k);
    }
    pi /= sumExpZ;
    
    T density = 0;
    for(int k = 0; k < mix; ++k){
      density += pi(k) * dets(k) * pow(6.283185, -3) *  exp(-0.5 * kernel(k));
    }
    
    return log(density);
  }
};


// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List LG1mixdiffF(double data, Rcpp::NumericMatrix lambdaIn, double z, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQSingle logQ(z, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = LGpSingleFixed(data, z);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List LGtmixdiffF(double data, Rcpp::NumericMatrix lambdaIn, double z, vec weights, vec zStats, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQSingle logQ(z, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = LGpMixFixed(data, z, zStats, weights);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}











