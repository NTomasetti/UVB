// [[Rcpp::depends(rstan)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <stan/math.hpp>
#include <Eigen/Dense>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace arma;
using namespace std;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::MatrixXd; 
using Eigen::Map;

struct erratic {
  const vec epsilon;
  const mat priorMean;
  const cube priorLinv;
  const cube data;
  erratic(const vec& epsIn, const mat& priorMeanIn, const cube& priorLinvIn, const cube& dataIn) :
    epsilon(epsIn), priorMean(priorMeanIn), priorLinv(priorLinvIn), data(dataIn) {}
  template <typename T> //
  T operator ()(const Matrix<T, Dynamic, 1>& lambda)
    const{
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::lgamma;
    
    int N = data.n_slices;
    int Tn = data.n_rows;
    
    T logdetJ = 0;
    Matrix<T, Dynamic, Dynamic> theta(N, 2);
    for(int n = 0; n < N; ++n){
      for(int i = 0; i < 2; ++i){
        theta(n, i) = lambda(n*6 + i);
        for(int j = 0; j <= i; ++j){
          theta(n, i) += lambda(6*n + 2*(i+1) + j) * epsilon(2*n + j);
          if(i == j){
            logdetJ += log(fabs(lambda(6*n + 2*(i+1) + j)));
          }
        }
        logdetJ += theta(n, i);
        theta(n, i) = exp(theta(n, i));
      }
    }
    
    T loglik = 0;
    T prior = 0;
    for(int n = 0; n < N; ++n){
      for(int i = 0; i < 2; ++i){
        T kernel = 0;
        for(int j = 0; j <= i; ++j){
          kernel += (log(theta(n, j)) - priorMean(j, n)) * priorLinv(i, j, n);
        }
        prior += -0.5 * pow(kernel, 2);
      }
      
      for(int t = 0; t < Tn; ++t){
        loglik += - 0.5 * log(theta(n, 0)) - pow(data(t, 0, n), 2) / (2 * theta(n, 0))
        - 0.5 * log(theta(n, 1)) - pow(data(t, 1, n), 2) / (2 * theta(n, 1));
      }
    }
    return loglik + logdetJ + prior;
  }
};

// [[Rcpp::export]]
Rcpp::List erraticGrad(cube data, Rcpp::NumericMatrix lambdaIn, vec epsilon, mat priorMean, cube priorLinv){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  double eval;
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1> grad(dim);
  
  erratic ELBO(epsilon, priorMean, priorLinv, data);
  stan::math::set_zero_all_adjoints();
  stan::math::gradient(ELBO, lambda, eval, grad);
  
  return Rcpp::List::create(Rcpp::Named("grad") = grad,
                            Rcpp::Named("val") = eval);
}
