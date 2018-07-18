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


/* VB Reparameterised Estimator for an AR2 Model
 * Inputs:
 * Z - vector of data
 * epsilon - standard normal noise for reparameterisation
 * mean - multivariate normal prior mean vector
 * Linv - Inverse Lower Triangular (Cholesky) of Variance Matrix of multivariate normal prior
 * conditional - If true evaluate likelihood conditional on first two observations, ie. p(z_3, ... z_T | theta, z_1, z_2)
 * if false use the unconditional likelihood, p(z_1, z_2, ...,  z_T | theta)
 */

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
    // Note the template type T, and T operator()
    // Need to explicitly state what std functions we are using inside the structure so stan can learn to take the derivative of these
    using std::log; using std::exp; using std::pow; using std::sqrt; using std::fabs;
    int N = z.n_rows;
    
    // Lambda is a 20 * 1 Matrix containing the length 4 mean vector of the multivariate normal approximation,
    // followed by the 16 elements of the upper triangular (cholesky) decomposition of the variance matrix
    // Transform theta = f(epsilon, lambda) = Mu + U * epsilon, location scale transform from a standard normal
    // Note that this is a matrix containing elements of type T, which will be replace with stan::math::var when the structure is called in stan::math::gradient
    Matrix<T, Dynamic, 1> theta(4);
    for(int i = 0; i < 4; ++i){
      theta(i) = lambda(i);
      for(int j = 0; j <= i; ++j){
        theta(i) += lambda(4*(i+1) + j) * epsilon(j);
      }
    }
    // Parameters of the AR2, z_t = mu + phi_1 (z_t-1 - mu) + phi_2 (z_t-2 - mu) + sigma * error
    // Keep everything as type T as these are included in the chain rule differentiation
    T sigmaSq = exp(theta(0)), mu = theta(1), phi1 = theta(2), phi2 = theta(3);
    // Evaluate log(p(theta)) as the log kernel of a multivariate normal is -0.5 * (L^-1(theta - mean))' (L^-1(theta - mean))
    T prior = 0;
    Matrix<T, Dynamic, 1> kernel(4);
    kernel.fill(0);
    for(int i = 0; i < 4; ++i){
      for(int j = 0; j <= i; ++j){
        kernel(i) += (theta(j) - mean(j)) * Linv(i, j);
      }
      prior += - 0.5 * pow(kernel(i), 2);
    }
    // Evaluate the log absolute value of the determinant of the jacobian of the transform, which is log(sigmaSq) + log (diagonal of cholesky decomposition of variance matrix)
    // std::fabs is a more stable version of std::abs
    T logdetJ = theta(0) + log(fabs(lambda(4))) + log(fabs(lambda(9))) + log(fabs(lambda(14))) + log(fabs(lambda(19)));

    
    // Evaluate log likelihood
    T logLik = 0;
    if(!conditional){
      // Unconditionally, the first two observations of an AR2 model have a bivariate normal distribution (assuming the error is normal)
      // Create the variance matrix for the initial states: Sigma = [gamma0, gamma1 // gamma1, gamma0]
      // Where gamma(k) is the k'th autocovariance
      T gamma0 = sigmaSq * (1 - phi2) / ((1+phi2) * (1 - phi1 - phi2) * (1 + phi1 - phi2));
      T gamma1 = gamma0 * phi1 / (1 - phi2);
      // Correlation between z_1 and z_2, equal to gamma1 / gamma0
      T rho = phi1 / (1 - phi2);
      T cons = - log(2 * 3.14159 * gamma0 * sqrt(1-pow(rho, 2))); //bvn constant, included as it depends on phi_1 and phi_2 via rho
      T z12 = pow(z(0) - mu, 2)/gamma0 + pow(z(1) - mu, 2)/gamma0 - 2*rho*(z(0)-mu)*(z(1) - mu)/ gamma0; //inside the exp term of the bivariate normal
      logLik = cons - z12/(2*(1-pow(rho, 2))); // log likelihood of Z1, Z2
    }
    
    // Loop through the log-likelihood of the remaining terms
    for(int t = 2; t < N; ++t){
      logLik += - 0.5 * theta(0) - pow(z(t) - mu - phi1 * (z(t-1)-mu) - phi2 * (z(t-2)-mu), 2) / (2 * sigmaSq);
    }
    return prior + logLik + logdetJ;
  }
};

/* This function is exported to R, and handles the actual differentiation
 * As we are differentiating with respect to lambda, it has to be an Eigen matrix, but everything else is constant so I prefer to keep them as arma objects
 * To create the Eigen lambda, we need to pass in an Rcpp matrix and map it to Eigen's MatrixXd
 * Input the data vector z, the parameter vector lambda, the noise vector epsilon, and the components of the prior distribution
 * It outputs a list of the value of the ELBO evaluated at this lambda / epsilon, and a vector of the gradients
 * I have an R gradient ascent wrapper that simulates epsilon, calls this function to get the gradients, then updates lambda via ADAM
 * This is all inside a while loop that stops if the absolute value of change in the ELBO is below a certain threshold, or the number of iterations is too high
 */
// [[Rcpp::export]]
Rcpp::List gradAR2(vec z, Rcpp::NumericMatrix lambdaIn, vec epsilon, vec mean, mat Linv, bool conditional = false){
  Map<MatrixXd> lambda(Rcpp::as<Map<MatrixXd> >(lambdaIn));
  // eval stores the result of the call to AR2
  double eval;
  // gradP stores the gradient vector, also an Eigen Matrix
  int dim = lambda.rows();
  Matrix<double, Dynamic, 1>  gradP(dim);
  
  // Autodiff
  // Create the structure of the AR2 model
  AR2 p(z, epsilon, mean, Linv, conditional);
  // Reset all of stans internal gradients to zero
  stan::math::set_zero_all_adjoints();
  // Pass lambda into p and take the derivative
  stan::math::gradient(p, lambda, eval, gradP);
  // Create an R list with the gradient and value
  return Rcpp::List::create(Rcpp::Named("grad") = gradP,
                            Rcpp::Named("val") = eval);
}

