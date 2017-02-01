#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <fstream>

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
  int n_data = data_test.size();
  int n_output = a.size() - 1; // index of output layer
  int count_correct = 0;
  int idx_max;

  cout << endl << "Testing network on training data, " << data_train.size() << " data points." << endl;
  count_correct = 0;
  for(int i = 0; i < data_train.size(); ++i) {
    EvaluateSingleDigit(data_train[i]);

    idx_max = 0;
    for(int j = 1; j < a[n_output].size(); ++j)
      if(a[n_output][j] > a[n_output][idx_max])
        idx_max = j;

    if(idx_max == data_train[i].GetDigit()) {
      if(verbose)
        cout << "Success!" << endl;
      count_correct++;
    }

    if(verbose) {
      cout << "Digit: " << data_train[i].GetDigit() << endl;
      for(int j = 0; j < a[n_output].size(); ++j)
        cout << a[n_output][j] << " ";
      cout << endl << endl;
    }
  }
  cout << "Final success rate: " << count_correct << "/" << data_train.size() << " = " << (double)count_correct/(double)data_train.size() << endl << endl;

  cout << "Testing network on test data, " << data_test.size() << " data points." << endl;
  count_correct = 0;
  for(int i = 0; i < data_test.size(); ++i) {
    EvaluateSingleDigit(data_test[i]);

    idx_max = 0;
    for(int j = 1; j < a[n_output].size(); ++j)
      if(a[n_output][j] > a[n_output][idx_max])
        idx_max = j;

    if(idx_max == data_test[i].GetDigit()) {
      if(verbose)
        cout << "Success!" << endl;
      count_correct++;
    }

    if(verbose) {
      cout << "Digit: " << data_test[i].GetDigit() << endl;
      for(int j = 0; j < a[n_output].size(); ++j)
        cout << a[n_output][j] << " ";
      cout << endl << endl;
    }
  }

  cout << "Final success rate: " << count_correct << "/" << data_test.size() << " = " << (double)count_correct/(double)data_test.size() << endl;

  return;
}

// calculates the value of the cost function and gradients for given data
double NNetwork::CostFunction(bool verbose) {
  int n_data = data_train.size();

  for(int i = 0; i < n - 1; ++i)
    theta_grad[i].zeros();

  double cost = 0.0;
  for(int i = 0; i < n_data; i++) {
    vec inputval(data_train[i].GetLabel());

    EvaluateSingleDigit(data_train[i]);
    BackpropagateSingleDigit(data_train[i]);
    
    if(verbose)
      cout << endl << "Data point: " << i << endl << a[n-1] << endl;

// cost function for single digit
    cost = cost + sum(-1.0*inputval % log(a[n-1]) - (1.0 - inputval) % (log(1.0-a[n-1])));

    if(verbose)
      cout << "Cost tmp value: " << cost << endl;   

    for(int i = n - 2; i >= 0; --i) {
      theta_grad[i] = theta_grad[i] + delta[i+1]*(a[i].t());
    }
  }

  for(int i = 0; i < n - 1; i++) // cost function regularization
    cost = cost + 0.5*lambda*accu(theta[i].submat(0,1,l_size[i+1]-1,l_size[i]) % theta[i].submat(0,1,l_size[i+1]-1,l_size[i]));
    
  cost = cost/static_cast<double>(n_data);    

  for(int i = 0; i < n - 1; ++i)
    theta_grad[i] = (theta_grad[i] + lambda*join_rows(zeros<vec>(theta[i].n_rows),theta[i].submat(span::all,span(1,theta[i].n_cols-1))))/(double)n_data;

  return cost;
} 

// driver function for gradient checking

// =================================
// trebalo bi provjeriti relativna odstupanja svih elemenata matrica i ispisati range
// =================================

void NNetwork::CheckGradientDriver(std::vector<Digit> data) {
  cout << "Cost function value: " << CostFunction(false) << endl;

  cout << endl << "Theta_grad 0 " << endl;
  cout << theta_grad[0].submat(span(0,9),span(0,1)) << endl;

  mat grad_exact = CheckGradient(data,0);

  cout << endl << "Theta_grad 0 exact: " << endl;
  cout << grad_exact << endl;

  cout << "Theta_grad 1 " << endl;
  cout << theta_grad[1].submat(span::all,span(0,1)) << endl;

  grad_exact = CheckGradient(data,1);

  cout << endl << "Theta_grad 0 exact: " << endl;
  cout << grad_exact << endl;

  return;
}

// calculate gradients directly and output
mat NNetwork::CheckGradient(std::vector<Digit> data, int k) {
  double eps = 0.0001;
  double epsi = 1.0/eps;
  double fp,fm;
  mat grad = zeros<mat>(10,2);

  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 10; ++j) {
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
void NNetwork::BackpropagateSingleDigit(Digit &input) {
  vec inputval(input.GetLabel());

  for(int i = 0; i < n; ++i)  
    delta[i].zeros();

  delta[n-1] = a[n-1] - inputval;
  for(int i = n - 2; i > 0; --i) {
    delta[i] = (theta[i].submat(span::all,span(1,l_size[i]))).t()*delta[i+1] % SigmoidGradient(i);
  }

  return;
}

// evaluate a single input digit with the current network and return output layer 
bool NNetwork::EvaluateSingleDigit(Digit &input) {
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
  for(int i = 0; i < n - 2; ++i) {
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

  clock_t t,t1;

  double stepmax;

// vectorise parameters
  vec p = MatrixToVector(theta);

  int dim = p.size();
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
      cout << "  " << (double)t/CLOCKS_PER_SEC << " s" << endl;
      t = clock();
    }

    stepmax = 100.0*max(sqrt(accu(p % p)),static_cast<double>(p.size()));
    fval = linesearch(p,fval,grad,xi,pnew,stepmax);

    xi = pnew - p;
    p = pnew;
//    cout << "Fval : " << fval << endl << p << endl;

    if(abs(xi).max() < eps_convergence)
      return fval; // convergence achieved

    dgrad = grad;
    grad = MatrixToVector(theta_grad);

    double den = max(fval,1.0);
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

  int len = xi.size();
  int nr = hesseinv.n_rows;
  int nc = hesseinv.n_cols;

  for(int col = 0; col < len; ++col) {
    for(int row = 0; row < len; ++row) {
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

  double lambda = 1.0;
  double lambdamin = 1e-07/fabs(p.max());

  double slope = accu(gold % p);

  double fnew,tmplambda;
  double fval2,lambda2;
  double dlambda,ll1,ll2;
  double a,b,rhs1,rhs2,fold2;
  double disc; // discriminant

  int counter = 0;
  if(sqrt(accu(p % p)) > stepmax)
    p = p*stepmax/sqrt(accu(p % p));

  for(;;) {
    xnew = xold + lambda*p;
    VectorToMatrix(xnew,theta);
    fnew = CostFunction(false);

    if(lambda < lambdamin) {
      xnew = xold;
      return fold;
    }
    else if(fnew <= fold + alpha*lambda*slope) {
      return fnew;
    }
    else {
      if(lambda == 1.0)
        tmplambda = -0.5*slope/(fnew - fold - slope);
      else { // cubic formula
        rhs1 = fnew - fold - lambda*slope;
        rhs2 = fval2 - fold2 - lambda2*slope;

        ll1 = lambda*lambda;
        ll2 = lambda2*lambda2;
        dlambda = lambda - lambda2;
        a = (rhs1/ll1 - rhs2/ll2)/dlambda;
        b = (-lambda2*rhs1/ll1 + lambda*rhs2/ll2)/dlambda;
        if(a == 0.0)
          tmplambda = -0.5*slope/b;
        else {
          disc = b*b - 3.0*a*slope;
          if(disc < 0.0) {
            cout << "Error in linesearch - discriminant" << endl;
            return fold;
          }
          tmplambda = max((-b + sqrt(disc))/(3.0*a),0.5*lambda);
        }
      }
      lambda2 = lambda;
      fval2 = fnew;
      fold2 = fold;
      lambda = max(tmplambda,0.1*lambda);
    }
  } // end for loop

  return fnew;
}

// set the value of the input layer, and take care of bias  
  bool NNetwork::SetInputLayer(std::vector<double> in) {
    a[0][0] = 1.0;

    for(int i = 1; i < a[0].size(); ++i)
      a[0][i] = in[i-1];

    return true;
  }

// get network name
  std::string NNetwork::GetName() {
    return this->name;
  }

// setup a network name
  void NNetwork::SetName(std::string name) {
    this->name = name;

    return;
  }  

// get number of layers
  int NNetwork::GetNumLayers() {
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
  for(int i = 0; i < n; ++i) {
    cout << "Layer a, number " << i << ", size: " << size(a[i]) << endl;
    if(verbose_layers)
      cout << a[i] << endl;
  }

// output 'z' layers
  cout << endl;
  for(int i = 0; i < n; ++i) {
    cout << "Layer z, number " << i << ", size: " << size(z[i]) << endl;
    if(verbose_layers)
      cout << z[i] << endl;
  }

// output 'delta' layers
  cout << endl;
  for(int i = 0; i < n; ++i) {
    cout << "Layer delta, number " << i << ", size: " << size(delta[i]) << endl;
    if(verbose_layers)
      cout << delta[i] << endl;
  }

// output theta matrices
  cout << endl << "Number of parameter matrices: " << n - 1 << endl;
  for(int i = 0; i < n - 1; ++i) {
    cout << "Theta " << i << ", size: " << size(theta[i]) << ", # rows: " << theta[i].n_rows << ", # cols: " << theta[i].n_cols << endl;
    if(verbose_matrices)
      cout << theta[i] << endl;
  }

// output theta gradient matrices
  cout << endl << "Number of gradient matrices: " << n - 1 << endl;
  for(int i = 0; i < n - 1; ++i) {
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

  int n_layer = l_size.size();

  outfile << l_size.size() << endl; // number of layers
  for(int i = 0; i < n_layer; ++i)
    outfile << l_size[i] << " "; // size of layers
  outfile << endl;

  for(int layer = 0; layer < n_layer - 1; ++layer) {
    int nr = theta[layer].n_rows; // size of each matrix
    int nc = theta[layer].n_cols;
    outfile << nr << " " << nc << endl;
    for(int row = 0; row < nr; ++row) {
      for(int col = 0; col < nc; ++col) // output of matrices
        outfile << std::setprecision(17) << theta[layer](row,col) << " " ;
      outfile << endl;
    }
  }

  return;
}


vec NNetwork::Sigmoid(int k) {
  vec t = z[k];
  for(int i = 0; i < t.size(); ++i)
    t[i] = exp(-t[i]);

  return 1.0/(1.0 + t);
}

vec NNetwork::SigmoidGradient(int k) {
  vec t = Sigmoid(k);

  return t % (1.0 - t);
}

// vectorise matrices (parameters or gradients)
vec NNetwork::MatrixToVector(std::vector<mat> m) {
  vec p;

  int mnum = m.size();
  if(mnum < 1) {
    cout << "MatrixToVector - Error in number of matrices." << endl;
    return p;
  }

  p = vectorise(m[0]);
  for(int i = 1; i < mnum; i++)
    p = join_cols(p,vectorise(m[i]));

  return p;
}

void NNetwork::VectorToMatrix(vec p, std::vector<mat> &m) {
  int ndim = m.size();

  int vidx = 0;
  for(int i = 0; i < ndim; ++i) {
    for(int col = 0; col < m[i].n_cols; ++col)
      for(int row = 0; row < m[i].n_rows; ++row)
        m[i].at(row,col) = p.at(vidx++);
  }

  return;
}

double NNetwork::max(double a, double b) {
  if(a > b)
    return a;
  else
    return b;
}