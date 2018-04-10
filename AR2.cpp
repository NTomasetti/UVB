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

struct AR2 {
  const vec z;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  const bool conditional;
  AR2(const vec& zIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn, const bool& conditionalIn) :
    z(zIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn), conditional(conditionalIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs; using std::max;
    int N = z.n_rows;
    
    Matrix<T, Dynamic, 1> theta(4);
    for(int i = 0; i < 4; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(4*(i+1) + j) * epsilon(j);
      }
    }
    
    T sig2 = exp(theta(0)), mu = theta(1), phi1 = theta(2), phi2 = theta(3);
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
    T logdetJ = theta(0) + log(fabs(lambda(4))) + log(fabs(lambda(9))) + log(fabs(lambda(14))) + log(fabs(lambda(19)));

    
    // Evaluate log likelihood
    T logLik = 0;
    if(!conditional){
      T gamma0 = sig2 * (1 - phi2) / ((1+phi2) * (1 - phi1 - phi2) * (1 + phi1 - phi2)); //Create the variance matrix for the initial states: Sigma = (gamma0, gamma1 //gamma1, gamma0)
      T gamma1 = gamma0 * phi1 / (1 - phi2);
      T rho = phi1 / (1 - phi2);
      T cons = - log(2 * 3.14159 * gamma0 * sqrt(1-pow(rho, 2))); //bvn constant
      T z12 = pow(z(0) - mu, 2)/gamma0 + pow(z(1) - mu, 2)/gamma0 - 2*rho*(z(0)-mu)*(z(1) - mu)/ gamma0; //inside exp term
      logLik = cons - z12/(2*(1-pow(rho, 2))); // log likelihood of Z1, Z2
    }
    
    for(int t = 2; t < N; ++t){
      logLik += - 0.5 * theta(0) - pow(z(t) - mu - phi1 * (z(t-1)-mu) - phi2 * (z(t-2)-mu), 2) / (2 * sig2);
    }
    return prior + logLik + logdetJ;
  }
};

// [[Rcpp::export]]
Rcpp::List gradAR2(vec data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, bool conditional = false){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  // Autodiff

  AR2 p(data, epsilon, mean, Linv, conditional);
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

double pLogDensMix(vec z, vec theta, mat mean, cube SigInv, vec dets, vec weights){
  int N = z.n_rows;
  int mix = weights.n_rows;
  // Constrained Positive
  double sig2 = std::exp(theta(0)), mu = theta(1), phi1 = theta(2), phi2 = theta(3);
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  
  // Evaluate log likelihood
  double logLik = 0;
  for(int t = 2; t < N; ++t){
    logLik += - 0.5 * theta(0) - pow(z(t) - mu - phi1 * (z(t-1)-mu) - phi2 * (z(t-2)-mu), 2) / (2 * sig2);
  }
  
  return std::log(prior) + logLik;
}

double pLogDensSingle(vec z, vec theta, vec mean, mat Linv){
  int N = z.n_rows;
  // Constrained Positive
  double sig2 = std::exp(theta(0)), mu = theta(1), phi1 = theta(2), phi2 = theta(3);
  // Evaluate log(p(theta))
  double prior = 0;
  mat sigInv = Linv.t() * Linv;
  prior = - 0.5 * as_scalar((theta - mean).t() * sigInv * (theta - mean));
  
  // Evaluate log likelihood
  double gamma0 = sig2 * (1 - phi2) / ((1+phi2) * (1 - phi1 - phi2) * (1 + phi1 - phi2)); //Create the variance matrix for the initial states: Sigma = (gamma0, gamma1 //gamma1, gamma0)
  double gamma1 = gamma0 * phi1 / (1 - phi2);
  
  double rho = phi1 / (1 - phi2);
  double cons = - log(2 * 3.14159 * gamma0 * sqrt(1-pow(rho, 2))); //bvn constant
  double z12 = pow(z(0) - mu, 2)/gamma0 + pow(z(1) - mu, 2)/gamma0 - 2*rho*(z(0)-mu)*(z(1) - mu)/ gamma0; //inside exp term
  double logLik = cons - z12/(2*(1-pow(rho, 2))); // log likelihood of Z1, Z2
  
  for(int t = 2; t < N; ++t){
    logLik += - 0.5 * theta(0) - pow(z(t) - mu - phi1 * (z(t-1)-mu) - phi2 * (z(t-2)-mu), 2) / (2 * sig2);
  }
  // Evaluate log likelihood
  return prior + logLik;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List singlePriorMixApprox(vec data, Rcpp::NumericMatrix lambdaIn, vec theta, vec mean, mat Linv, int mix = 6){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensSingle(data, theta, mean, Linv);
  double elbo = logp - qEval;
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List mixPriorMixApprox(vec data, Rcpp::NumericMatrix lambdaIn, vec theta, mat mean, cube SigInv, vec dets, vec weights){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  int mix = weights.n_rows;
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixQlogdens logQ(theta, mix);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensMix(data, theta, mean, SigInv, dets, weights);
  double elbo = logp - qEval;
  
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}
