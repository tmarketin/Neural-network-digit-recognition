#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <fstream>
#include <algorithm>

#include <cmath>
#include <ctime>

#include <armadillo>

#include "mnist.h"
#include "nnet.h"

using std::cout;
using std::endl;
using std::log;

using namespace arma;

// test network and compare
void NNetwork::TestNetwork(bool verbose) {
  size_t n_output = a.size() - 1; // index of output layer
  int count_correct = 0;
  arma::uword idx_max;

  cout << endl << "Testing network on training data, " << data_train.size() << " data points." << endl;
  count_correct = 0;
  for(size_t i = 0; i < data_train.size(); ++i) {
    EvaluateSingleDigit(data_train[i]);

    idx_max = 0;
    for(arma::uword j = 1; j < static_cast<arma::uword>(a[n_output].size()); ++j)
      if(a[n_output][j] > a[n_output][idx_max])
        idx_max = j;

    if(idx_max == static_cast<arma::uword>(data_train[i].GetDigit())) {
      if(verbose)
        cout << "Success!" << endl;
      count_correct++;
    }

    if(verbose) {
      cout << "Digit: " << data_train[i].GetDigit() << endl;
      for(arma::uword j = 0; j < static_cast<arma::uword>(a[n_output].size()); ++j)
        cout << a[n_output][j] << " ";
      cout << endl << endl;
    }
  }
  cout << "Final success rate: " << count_correct << "/" << data_train.size() << " = " << static_cast<double>(count_correct)/static_cast<double>(data_train.size()) << endl << endl;

  cout << "Testing network on test data, " << data_test.size() << " data points." << endl;
  count_correct = 0;
  for(size_t i = 0; i < data_test.size(); ++i) {
    EvaluateSingleDigit(data_test[i]);

    idx_max = 0;
    for(arma::uword j = 1; j < static_cast<arma::uword>(a[n_output].size()); ++j)
      if(a[n_output][j] > a[n_output][idx_max])
        idx_max = j;

    if(idx_max == static_cast<arma::uword>(data_test[i].GetDigit())) {
      if(verbose)
        cout << "Success!" << endl;
      count_correct++;
    }

    if(verbose) {
      cout << "Digit: " << data_test[i].GetDigit() << endl;
      for(arma::uword j = 0; j < static_cast<arma::uword>(a[n_output].size()); ++j)
        cout << a[n_output][j] << " ";
      cout << endl << endl;
    }
  }

  cout << "Final success rate: " << count_correct << "/" << data_test.size() << " = " << static_cast<double>(count_correct)/static_cast<double>(data_test.size()) << endl;

  return;
}

// calculates the value of the cost function and gradients for given data
double NNetwork::CostFunction(bool verbose) {
  size_t n_data = data_train.size();

  for(size_t i = 0; i < n - 1; ++i)
    theta_grad[i].zeros();

  double cost = 0.0;
  for(size_t i = 0; i < n_data; i++) {
    vec inputval(data_train[i].GetLabel());

    EvaluateSingleDigit(data_train[i]);
    BackpropagateSingleDigit(data_train[i]);
    
    if(verbose)
      cout << endl << "Data point: " << i << endl << a[n-1] << endl;

// cost function for single digit
    cost = cost + sum(-1.0*inputval % log(a[n-1]) - (1.0 - inputval) % (log(1.0-a[n-1])));

    if(verbose)
      cout << "Cost tmp value: " << cost << endl;   

    for(int layer = static_cast<int>(n - 2); layer >= 0; --layer) {
      theta_grad[static_cast<size_t>(layer)] = theta_grad[static_cast<size_t>(layer)] + delta[static_cast<size_t>(layer + 1)]*(a[static_cast<size_t>(layer)].t());
    }
  }

  for(size_t i = 0; i < n - 1; i++) // cost function regularization
    cost = cost + 0.5*lambda*accu(theta[i].submat(0,1,l_size[i+1]-1,l_size[i]) % theta[i].submat(0,1,l_size[i+1]-1,l_size[i]));
    
  cost = cost/static_cast<double>(n_data);    

  for(size_t i = 0; i < n - 1; ++i)
    theta_grad[i] = (theta_grad[i] + lambda*join_rows(zeros<vec>(theta[i].n_rows),theta[i].submat(span::all,span(1,theta[i].n_cols-1))))/static_cast<double>(n_data);

  return cost;
} 

// driver function for gradient checking

// =================================
// trebalo bi provjeriti relativna odstupanja svih elemenata matrica i ispisati range
// =================================

void NNetwork::CheckGradientDriver() {
  cout << "Cost function value: " << CostFunction(false) << endl;

  cout << endl << "Theta_grad 0 " << endl;
  cout << theta_grad[0].submat(span(0,9),span(0,1)) << endl;

  mat grad_exact = CheckGradient(0);

  cout << endl << "Theta_grad 0 exact: " << endl;
  cout << grad_exact << endl;

  cout << "Theta_grad 1 " << endl;
  cout << theta_grad[1].submat(span::all,span(0,1)) << endl;

  grad_exact = CheckGradient(1);

  cout << endl << "Theta_grad 0 exact: " << endl;
  cout << grad_exact << endl;

  return;
}

// calculate gradients directly and output
mat NNetwork::CheckGradient(size_t k) {
  double eps = 0.0001;
  double epsi = 1.0/eps;
  double fp,fm;
  mat grad = zeros<mat>(10,2);

  for(arma::uword i = 0; i < 2; ++i) {
    for(arma::uword j = 0; j < 10; ++j) {
      theta[k](j,i) += eps;
      fp = CostFunction(false);
      theta[k](j,i) -= 2*eps;
      fm = CostFunction(false);
      theta[k](j,i) += eps;

      grad(j,i) = 0.5*(fp - fm)*epsi;
    }
  }

  return grad;
} 


// after evaluating a digit, backpropagate to obtain gradients
// function assumes a and z layers have been determined
void NNetwork::BackpropagateSingleDigit(const Digit &input) {
  vec inputval(input.GetLabel());

  for(size_t i = 0; i < n; ++i)  
    delta[i].zeros();

  delta[n-1] = a[n-1] - inputval;
  for(size_t i = n - 2; i > 0; --i) {
    delta[i] = (theta[i].submat(span::all,span(1,l_size[i]))).t()*delta[i+1] % SigmoidGradient(i);
  }

  return;
}

// evaluate a single input digit with the current network and return output layer 
bool NNetwork::EvaluateSingleDigit(const Digit &input) {
  if(IsEmpty()) {
    cout << "NNetwork is empty!" << endl;
    return false;
  }

  if(input.GetValue().size() + 1 != this->a[0].size()) {
    cout << "Input vector size mismatch!" << endl;
    cout << "Input vector size: " << input.GetValue().size() << endl;
    cout << "Input layer size: " << a[0].size() << endl;
    return false;
  }

  SetInputLayer(input.GetValue());
  for(size_t i = 0; i < n - 2; ++i) {
    z[i+1] = theta[i]*a[i];
    a[i+1].subvec(1,a[i+1].size()-1) = Sigmoid(i+1);

    if(i < n - 2) // if not output layer, add bias
      a[i+1][0] = 1.0;
  } // output layer different due to no bias
  z[n-1] = theta[n-2]*a[n-2];
  a[n-1] = Sigmoid(n-1);

  return true;
} 

// variable metric method of minimization, see Numerical Recipes 10.7
// based on the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
double NNetwork::MinimizeNetwork() {
  const double eps_convergence = 1e-02;

  clock_t t;

  double stepmax;

// vectorise parameters
  vec p = MatrixToVector(theta);

  arma::uword dim = p.size();
  mat hesseinv = eye<mat>(dim,dim);

  vec pnew = zeros<vec>(dim); // new position
  vec dgrad = zeros<vec>(dim); // old gradient
  vec hdg = zeros<vec>(dim); // hessian times gradient difference

  double fval = CostFunction(false);
// vectorize gradients
  vec grad = MatrixToVector(theta_grad);
  vec xi = -grad;

  double fac1,fac2,fac3;
  double sumxi,sumdg;

  mat xit,hdgt,dgradt;

  t = clock();
  for(int iter = 0; iter < MIN_MAXITER; iter++) {
    cout << "Iteration: " << iter << "  " << fval;
    if(iter == 0)
      cout << endl;
    else {
      t = clock() - t;
      cout << "  " << static_cast<double>(t)/CLOCKS_PER_SEC << " s" << endl;
      t = clock();
    }

    stepmax = 100.0*std::max(sqrt(accu(p % p)),static_cast<double>(p.size()));
    fval = linesearch(p,fval,grad,xi,pnew,stepmax);

    xi = pnew - p;
    p = pnew;
//    cout << "Fval : " << fval << endl << p << endl;

    if(abs(xi).max() < eps_convergence)
      return fval; // convergence achieved

    dgrad = grad;
    grad = MatrixToVector(theta_grad);

    double den = std::max(fval,1.0);
    if((abs(grad).max())*(abs(p).max())/den < 1e-05)
      return fval; // gradient convergence achieved

    dgrad = grad - dgrad; // gradience difference
    hdg = hesseinv*dgrad; // H(f_i+1 - f_i)

    fac1 = accu(xi % dgrad);
    fac3 = accu(dgrad % hdg);
    sumxi = accu(xi % xi);
    sumdg = accu(dgrad % dgrad);

    if(fac1*fac1 > eps_convergence*sumdg*sumxi) {
      fac1 = 1.0/fac1;
      fac2 = 1.0/fac3;
      dgrad = fac1*xi - fac2*hdg; // u vector for BFGS

      UpdateHesseInverse(hesseinv,fac1,fac2,fac3,xi,hdg,dgrad);
    }

    xi = -hesseinv*grad;
  }

  return fval;
}

// update of the inverse of the Hesse matrix
// using dedicated function with direct memory access 
// for significantly enhanced performance 
void NNetwork::UpdateHesseInverse(mat &hesseinv,double fac1, double fac2, double fac3, vec &xi, vec &hdg, vec &dgrad) {
  double *h = hesseinv.memptr(); // hesse inverse

  double *ar = xi.memptr(); // position vector difference, rows
  double *ac = xi.memptr(); // position vector difference, columns

  double *br = hdg.memptr(); // hesse times gradient difference
  double *bc = hdg.memptr(); // hesse times gradient difference

  double *cr = dgrad.memptr(); // gradient difference
  double *cc = dgrad.memptr(); // gradient difference

  arma::uword len = xi.size();

  for(arma::uword col = 0; col < len; ++col) {
    for(arma::uword row = 0; row < len; ++row) {
      *h = *h + fac1*(*ar)*(*ac) - fac2*(*br)*(*bc) + fac3*(*cr)*(*cc);

      h++;

      ar++;
      br++;
      cr++;
    }
    ar = xi.memptr();
    br = hdg.memptr();
    cr = dgrad.memptr();

    ac++;
    bc++;
    cc++;
  }

  return;
}

// line search for minimization, see Numerical Recipes sec. 9.7
double NNetwork::linesearch(vec xold, double fold, vec gold, vec p, vec &xnew, double stepmax) {
  const double alpha = 1e-04;

  double loc_lambda = 1.0; // local lambda var
  double loc_lambdamin = 1e-07/fabs(p.max());

  double slope = accu(gold % p);

  double fnew,tmplambda;
  double fval2 = 0.0;
  double lambda2 = 0.0;
  double fold2 = 0.0;
  double dlambda,ll1,ll2;
  double afac,bfac,rhs1,rhs2;
  double disc; // discriminant

  if(sqrt(accu(p % p)) > stepmax)
    p = p*stepmax/sqrt(accu(p % p));

  for(;;) {
    xnew = xold + loc_lambda*p;
    VectorToMatrix(xnew,theta);
    fnew = CostFunction(false);

    if(loc_lambda < loc_lambdamin) {
      xnew = xold;
      return fold;
    }
    else if(fnew <= fold + alpha*loc_lambda*slope) {
      return fnew;
    }
    else {
      if(loc_lambda == 1.0)
        tmplambda = -0.5*slope/(fnew - fold - slope);
      else { // cubic formula
        rhs1 = fnew - fold - loc_lambda*slope;
        rhs2 = fval2 - fold2 - lambda2*slope;

        ll1 = loc_lambda*loc_lambda;
        ll2 = lambda2*lambda2;
        dlambda = loc_lambda - lambda2;
        afac = (rhs1/ll1 - rhs2/ll2)/dlambda;
        bfac = (-lambda2*rhs1/ll1 + loc_lambda*rhs2/ll2)/dlambda;
        if(afac == 0.0)
          tmplambda = -0.5*slope/bfac;
        else {
          disc = bfac*bfac - 3.0*afac*slope;
          if(disc < 0.0) {
            cout << "Error in linesearch - discriminant" << endl;
            return fold;
          }
          tmplambda = std::max((-bfac + sqrt(disc))/(3.0*afac),0.5*loc_lambda);
        }
      }
      lambda2 = loc_lambda;
      fval2 = fnew;
      fold2 = fold;
      loc_lambda = std::max(tmplambda,0.1*loc_lambda);
    }
  } // end for loop

  return fnew;
}

// set the value of the input layer, and take care of bias  
  bool NNetwork::SetInputLayer(std::vector<double> in) {
    a[0][0] = 1.0;

    for(arma::uword i = 1; i < a[0].size(); ++i)
      a[0][i] = in[i-1];

    return true;
  }

// get network name
  std::string NNetwork::GetName() {
    return this->name;
  }

// setup a network name
  void NNetwork::SetName(std::string new_name) {
    this->name = new_name;

    return;
  }  

// get number of layers
  size_t NNetwork::GetNumLayers() {
    return this->n;
  }

// check if network is initiaized
  bool NNetwork::IsEmpty() {
    if(this->n == 0)
      return true;
    else 
      return false;
  }

void NNetwork::PrintOutputLayer() {
  cout << a[a.size()-1] << endl;

  return;
}

void NNetwork::PrintStatus(bool verbose_layers, bool verbose_matrices) {
  if(this->IsEmpty()) {
    cout << "Neural network is not initialized!" << endl;
    return;
  }

  cout << "Neural network: " << this->GetName() << endl << endl;
  cout << "Number of layers: " << this->n << endl;

// output 'a' layers
  for(size_t i = 0; i < n; ++i) {
    cout << "Layer a, number " << i << ", size: " << size(a[i]) << endl;
    if(verbose_layers)
      cout << a[i] << endl;
  }

// output 'z' layers
  cout << endl;
  for(size_t i = 0; i < n; ++i) {
    cout << "Layer z, number " << i << ", size: " << size(z[i]) << endl;
    if(verbose_layers)
      cout << z[i] << endl;
  }

// output 'delta' layers
  cout << endl;
  for(size_t i = 0; i < n; ++i) {
    cout << "Layer delta, number " << i << ", size: " << size(delta[i]) << endl;
    if(verbose_layers)
      cout << delta[i] << endl;
  }

// output theta matrices
  cout << endl << "Number of parameter matrices: " << n - 1 << endl;
  for(size_t i = 0; i < n - 1; ++i) {
    cout << "Theta " << i << ", size: " << size(theta[i]) << ", # rows: " << theta[i].n_rows << ", # cols: " << theta[i].n_cols << endl;
    if(verbose_matrices)
      cout << theta[i] << endl;
  }

// output theta gradient matrices
  cout << endl << "Number of gradient matrices: " << n - 1 << endl;
  for(size_t i = 0; i < n - 1; ++i) {
    cout << "Theta_grad " << i << ", size: " << size(theta_grad[i]) << ", # rows: " << theta_grad[i].n_rows << ", # cols: " << theta_grad[i].n_cols << endl;
    if(verbose_matrices)
      cout << theta_grad[i] << endl;
  }

  cout << endl << "Number of training data points: " << data_train.size() << endl;
  cout << "Number of testing data points: " << data_test.size() << endl;

  cout << endl;

  return;
}

// load training data into network
void NNetwork::LoadTrainingData(std::vector<Digit> data) {
  data_train = data;

  return;
}

// load testing data into network
void NNetwork::LoadTestingData(std::vector<Digit> data) {
  data_test = data;

  return;
}

// save network 
void NNetwork::SaveNetwork(std::string fname) {
  std::ofstream outfile(fname, ios::out);

  size_t n_layer = l_size.size();

  outfile << l_size.size() << endl; // number of layers
  for(size_t i = 0; i < n_layer; ++i)
    outfile << l_size[i] << " "; // size of layers
  outfile << endl;

  for(size_t layer = 0; layer < n_layer - 1; ++layer) {
    arma::uword nr = theta[layer].n_rows; // size of each matrix
    arma::uword nc = theta[layer].n_cols;
    outfile << nr << " " << nc << endl;
    for(arma::uword row = 0; row < nr; ++row) {
      for(arma::uword col = 0; col < nc; ++col) // output of matrices
        outfile << std::setprecision(17) << theta[layer](row,col) << " " ;
      outfile << endl;
    }
  }

  return;
}


vec NNetwork::Sigmoid(size_t k) {
  vec t = z[k];
  for(arma::uword i = 0; i < t.size(); ++i)
    t[i] = exp(-t[i]);

  return 1.0/(1.0 + t);
}

vec NNetwork::SigmoidGradient(size_t k) {
  vec t = Sigmoid(k);

  return t % (1.0 - t);
}

// vectorise matrices (parameters or gradients)
vec NNetwork::MatrixToVector(std::vector<mat> m) {
  vec p;

  size_t mnum = m.size();
  if(mnum < 1) {
    cout << "MatrixToVector - Error in number of matrices." << endl;
    return p;
  }

  p = vectorise(m[0]);
  for(size_t i = 1; i < mnum; i++)
    p = join_cols(p,vectorise(m[i]));

  return p;
}

void NNetwork::VectorToMatrix(vec p, std::vector<mat> &m) {
  size_t ndim = m.size();

  arma::uword vidx = 0;
  for(size_t i = 0; i < ndim; ++i) {
    for(arma::uword col = 0; col < m[i].n_cols; ++col)
      for(arma::uword row = 0; row < m[i].n_rows; ++row)
        m[i].at(row,col) = p.at(vidx++);
  }

  return;
}
