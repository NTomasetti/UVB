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

struct mixNormalSPSA {
  const mat data;
  const vec epsilon;
  const vec mean;
  const mat Linv;
  const mat piPrior;
  const bool update;
  mixNormalSPSA(const mat& dataIn, const vec& epsIn, const vec& meanIn, const mat& LinvIn, const mat& piPriorIn, const bool& updateIn) :
    data(dataIn), epsilon(epsIn), mean(meanIn), Linv(LinvIn), piPrior(piPriorIn), update(updateIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::lgamma; using std::fabs;
    
    int N = data.n_cols;
    int M = data.n_rows;
    
    Matrix<T, Dynamic, 1> theta(4);
    for(int i = 0; i < 4; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(4 + 4*i + j) * epsilon(j);
      }
    }
    
    T sigmaSq1 = exp(theta(0)), sigmaSq2 = exp(theta(1));
    Matrix<T, Dynamic, 1> z(N), pi(N);
    for(int i = 0; i < N; ++i){
      z(i) = lambda(20 + i) + lambda(20 + N + i) * epsilon(4 + i);
      pi(i) = 1.0 / (1 + exp(-z(i)));
    }
    
    T logPrior = 0;
    Matrix<T, Dynamic, 1> kernel(4);
    kernel.fill(0);
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      logPrior += - 0.5 * pow(kernel(i), 2);
    }
    if(update){
      for(int i = 0; i < N; ++i){
        logPrior += pow(z(i) - piPrior(i, 0), 2) / (2 * piPrior(i, 1));
      }
    } else {
      for(int i = 0; i < N; ++i){
        logPrior += (piPrior(i, 0) - 1) * log(pi(i)) + (piPrior(i, 1) - 1) * log(1 - pi(i));
      }
    }
  
    
    T logdetJ = theta(0) + theta(1) + log(fabs(lambda(4))) + log(fabs(lambda(9))) + log(fabs(lambda(14))) + log(fabs(lambda(19)));
    for(int i = 0; i < N; ++i){
      logdetJ += log(fabs(lambda(20 + N + i))) + z(i) - 2 * log(exp(z(i)) + 1);
    }
    
    T logLik = 0;
    for(int i = 0; i < N; ++i){
      for(int j = 0; j < M; ++j){
        logLik += log(
          pi(i) * pow(2 * 3.14159 * sigmaSq1, -0.5) * exp(-pow(data(j, i) - theta(2), 2) / (2 * sigmaSq1))  +
          (1 - pi(i)) * pow(2 * 3.14159 * sigmaSq2, -0.5) * exp(-pow(data(j, i) - theta(3), 2) / (2 * sigmaSq2))
        );
      }
    }
    return logPrior + logdetJ + logLik;
  }
};

// [[Rcpp::export]]
Rcpp::List mixtureNormalSPSA(mat data, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, mat piPrior, bool update = false){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  mixNormalSPSA logp(data, epsilon, mean, Linv, piPrior, update);
  
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logp, lambda, eval, grad);

  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = eval);
}


struct Qlogdens {
  const vec theta;
  const mat piK;
  const int K;
  Qlogdens(const vec& thetaIn, const mat& piIn, const int& KIn) :
    theta(thetaIn), piK(piIn), K(KIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::lgamma;
    
    int N = piK.n_rows;
    
    T determinant = exp(lambda(K));
    for(int i = 1; i < K; ++i){
      determinant *= exp(lambda(K + i));
    }
    determinant = 1.0 / determinant;

    T kernel = 0;
    for(int i = 0; i < K; ++i){
      kernel += pow((theta(i) - lambda(i)) / exp(lambda(K + i)), 2);
    }
    T logMVN = log(determinant) - 0.5 * kernel;
    
    T logDir = 0;
    Matrix<T, Dynamic, 1> lambdaSums(N);
    lambdaSums.fill(0);
    for(int i = 0; i < N; ++i){
      for(int j = 0; j < K; ++j){
        logDir += (exp(lambda(2*K +i*K + j)) - 1) * log(piK(i, j)) - lgamma(exp(lambda(2*K + i*K + j)));
        lambdaSums(i) += exp(lambda(2*K + i*K + j));
      }
      logDir += lgamma(lambdaSums(i));
    }
    return logMVN + logDir;
  }
};

double pLogDensSingle(mat y, vec theta, mat pi, vec mean, mat Linv, mat piPrior){
  int N = y.n_cols;
  int T = y.n_rows;

  // Evaluate log(p(theta))
  double logPrior = 0;
  for(int i = 0; i < 4; ++i){
    double kernel = 0;
    for(int j = 0; j <= i; ++j){
      kernel += (theta(j) - mean(j)) * Linv(i, j);
    }
    logPrior += - 0.5 * pow(kernel, 2);
  }
  for(int i = 0; i < N; ++i){
    logPrior += (piPrior(i, 0) - 1) * log(pi(i)) + (piPrior(i, 1) - 1) * log(1 - pi(i));
  }
  
  double logLik = 0;
  for(int i = 0; i < N; ++i){
    for(int t = 0; t < T; ++t){
      logLik += log(
        pi(i) * pow(2 * 3.14159 * exp(theta(0)), -0.5) * exp(-pow(y(t, i) - theta(2), 2) / (2 * exp(theta(0))))  +
          (1 - pi(i)) * pow(2 * 3.14159 * exp(theta(1)), -0.5) * exp(-pow(y(t, i) - theta(3), 2) / (2 * exp(theta(1))))
      );
    }
  }
  return logPrior + logLik;
  
  
  
}

double pLogDensMixture(mat y, vec theta, mat pi, mat mean, cube SigInv, mat piPrior, vec weights, vec dets){
  int N = y.n_cols;
  int T = y.n_rows;
  
  int mix = weights.n_rows;
  // Evaluate log(p(theta)), start by evaluative the quadratic in the MVN exponents
  double prior = 0;
  for(int k = 0; k < mix; ++k){
    prior += weights(k) * pow(6.283185, -3) * dets(k) * exp(-0.5 * as_scalar((theta - mean.col(k)).t() * SigInv.slice(k) * (theta - mean.col(k))));
  }
  double logPrior = log(prior);
  
  for(int i = 0; i < N; ++i){
    logPrior += (piPrior(i, 0) - 1) * log(pi(i)) + (piPrior(i, 1) - 1) * log(1 - pi(i));
  }
  
  double logLik = 0;
  for(int i = 0; i < N; ++i){
    for(int t = 0; t < T; ++t){
      logLik += log(
        pi(i) * pow(2 * 3.14159 * exp(theta(0)), -0.5) * exp(-pow(y(t, i) - theta(2), 2) / (2 * exp(theta(0))))  +
          (1 - pi(i)) * pow(2 * 3.14159 * exp(theta(1)), -0.5) * exp(-pow(y(t, i) - theta(3), 2) / (2 * exp(theta(1))))
      );
    }
  }
  return logPrior + logLik;
}

// These models are parameterised by the mean and log standard deviations
// [[Rcpp::export]]
Rcpp::List mixtureNormalSPMA(mat data, Rcpp::NumericMatrix lambdaIn, vec theta, mat pi, int K, vec mean, mat Linv, mat piPrior){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  Qlogdens logQ(theta, pi, K);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensSingle(data, theta, pi, mean, Linv, piPrior);
  double elbo = logp - qEval;
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}

// [[Rcpp::export]]
Rcpp::List mixtureNormalMPMA(mat data, Rcpp::NumericMatrix lambdaIn, vec theta, mat pi, int K, mat mean, cube SigInv, mat piPrior, vec weights, vec dets){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double qEval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  Qlogdens logQ(theta, pi, K);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logQ, lambda, qEval, grad);
  
  double logp = pLogDensMixture(data, theta, pi, mean, Linv, piPrior, weights, dets);
  double elbo = logp - qEval;
  for(int i = 0; i < dim; ++i){
    grad(i) *= elbo;
  }
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = elbo);
}




struct mixNormalP {
  const mat data;
  const int K;
  mixNormalP(const mat& dataIn, const int& kIn) :
    data(dataIn), K(kIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& theta)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt;
    
    int N = data.n_cols;
    int M = data.n_rows;
    
    Matrix<T, Dynamic, 1> sigmaSq(K), mu(K);
    Matrix<T, Dynamic, Dynamic> pi(N, K);
    for(int k = 0; k < K; ++k){
      sigmaSq(k) = exp(theta(k));
      mu(k) = theta(K + k);
      for(int i = 0; i < N; ++i){
        pi(i, k) = theta(2*K + i*K + k);
      }
    }
    
    T logPrior = 0;
    for(int k = 0; k < 2*K; ++k){
      logPrior += -pow(theta(k), 2) / 20;
    }
    
    T logLik = 0;
    for(int i = 0; i < N; ++i){
      for(int j = 0; j < M; ++j){
        T likelihood = 0;
        for(int k = 0; k < K; ++k){
          likelihood += pi(i, k) * pow(2 * 3.14159 * sigmaSq(k), -0.5) * exp(-pow(data(j, i) - mu(k), 2) / (2 * sigmaSq(k)));
        }
        logLik += log(likelihood);
      }
    }
    return logPrior + logLik;
  }
};

// [[Rcpp::export]]
Rcpp::List mixtureNormalGRP(mat data, Rcpp::NumericMatrix thetaIn, Rcpp::NumericMatrix lambdaIn,
                            int K, vec eps, cube SigmaSqrt, double logQ){
  Map<MatrixXd> theta(Rcpp::as<Map<MatrixXd> >(thetaIn));
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double evalP;
  int dimT = theta.rows(), dimL = lambda.rows(), N = data.n_cols;

  Matrix<double, Dynamic, 1> gradP(dimT);
  mixNormalP logp(data, K);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(logp, theta, evalP, gradP);

  vec grep(dimL, fill::zeros), gcorr(dimL, fill::zeros), H(dimL, fill::zeros);
  // Entropy derivatives
  for(int i = 0; i < K; ++i){
    H(i) = 1;
    for(int j = 0; j <= i; ++j){
      H(2*K + i*2*K + j) = eps(j);
    }
  }
  for(int i = 0; i < 2*K; ++i){
    H(2*K + (2*K + 1) * i) += 1.0 / lambda(2*K + (2*K + 1) * i);
  }
  
  // MVN Derivatives
  for(int i = 0; i < 2*K; ++i){
    if(i < K){
      grep(i) = theta(i) * gradP(i); // sigma Sq mean parameters
    } else {
      grep(i) = gradP(i); // Mu mean parameters
    }
    for(int j = 0; j <= i; ++j){
      grep(2*K + i*2*K + j) = eps(j) * grep(i); // Each L parameter is eps * mu parameter
    }
  }
  
  // Each alpha_ik Derivative
  for(int i = 0; i < N; ++i){
    // Extract components required
    vec alpha(K), pi(K), epsilon(K), dLogPdPi(K);
    for(int k = 0; k < K; ++k){
      alpha(k) = lambda(2 * K * (2 * K + 1) + K*i + k);
      pi(k) = theta(2*K + K*i + k);
      epsilon(k) = eps(2*K + K*i + k);
      dLogPdPi(k) = gradP(2*K + K*i + k);
    }
    double alpha0 = sum(alpha);
    // Matrix, element (i, j) = dmu_i / da_j
    mat dMudA(K, K);
    dMudA.fill(-boost::math::trigamma(alpha0));
    for(int k = 0; k < K; ++k){
      dMudA(k, k) +=  boost::math::trigamma(alpha(k));
    }
    
    // Cube, element (i, j, k) = dSigma^(1/2)_{i, j} / da_j
    cube dSigSqrtdA(K, K, K);
    for(int k = 0; k < K; ++k){
      mat dSigdA(K, K);
      dSigdA.fill(-boost::math::polygamma(2, alpha0));
      dSigdA(k, k) += polygamma(2, alpha(k));
      dSigSqrtdA.slice(k) = syl(SigmaSqrt.slice(i), SigmaSqrt.slice(i), -dSigdA);
    }
    
    // Matricies, hEpaA = h(epsilon, alpha) from the generalised reparam. gradient papr
    // components stores the sums of derivatives required in h(epsilon, alpha) and u(epsilon, alpha)
    mat hEpsA(K, K), components(K, K);
    for(int k = 0; k < K; ++k){
      for(int j = 0; j < K; ++j){
        components(k, j) = (as_scalar(dSigSqrtdA.slice(j).col(k).t() * epsilon) + dMudA(k, j));
        hEpsA(k, j) = pi(k) * components(k, j);
      }
    }
    // vectors of dlogq / dpi and dlogq / dalpha
    vec dqdz(K), dqda(K);
    for(int k = 0; k < K; ++k){
      dqdz(k) = (alpha(k) - 1) / pi(k);
      dqda(k) = boost::math::digamma(alpha0) - boost::math::digamma(alpha(k)) + log(pi(k));
    }
    // vector of u(epsilon, alpha)
    vec uEpsA(K);
    for(int k = 0; k < K; ++k){
      double logdet = trace(SigmaSqrt.slice(i) * dSigSqrtdA.slice(k));
      uEpsA(k) = logdet + sum(components.col(k));
    }
    
    grep(span(2*K*(2*K+1) + K*i, 2*K*(2*K+1) + K*(i+1) - 1)) = hEpsA.t() * dLogPdPi;
    gcorr(span(2*K*(2*K+1) + K*i, 2*K*(2*K+1) + K*(i+1) - 1)) = evalP * (hEpsA.t() * dqdz + dqda + uEpsA);
    H(span(2*K*(2*K+1) + K*i, 2*K*(2*K+1) + K*(i+1) - 1)) = dqdz * logQ;
  }
    
  double eval = evalP - logQ;
  vec grad = grep + gcorr + H;
  
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = eval);
  
}

