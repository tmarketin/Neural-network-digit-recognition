#ifndef NNET_H_
#define NNET_H_

#include <vector>
#include <string>
#include <fstream>

#include <armadillo>

#include "mnist.h"

using namespace arma;

class NNetwork {
private:
// name of the neural network
  std::string name; 

// sizes of the individual layers, including input and output layers, not including bias
// but bias is included by default
  std::vector<int> l_size; 

// number of layers, equal to l_size.size()
  int n;

// layers z, a and delta (for backpropagation), vectors of double
  std::vector<vec> z;
  std::vector<vec> a;
  std::vector<vec> delta;

// matrices between layers, double type
  std::vector<mat> theta;  
  std::vector<mat> theta_grad;

// learning parameter for regularization
  double lambda; 

// data for training and testing
  std::vector<Digit> data_train;
  std::vector<Digit> data_test;

// minimization parameter
  int MIN_MAXITER;

// apply sigmoid function to k-th z layer
  vec Sigmoid(int k);
  vec SigmoidGradient(int k);

// max function, returns greater
  double max(double a, double b);

// calculates the value of the cost function for given data
  double CostFunction(bool verbose);  

//  evaluate a single digit with current network and output result
  bool EvaluateSingleDigit(Digit &input);

// after evaluating a digit, backpropagate to obtain gradients
// function assumes a and z layers have been determined
  void BackpropagateSingleDigit(Digit &input);

// calculate gradients directly and output
  mat CheckGradient(std::vector<Digit> data, int k);  

// vectorise and reshape theta and theta_grad
  vec MatrixToVector(std::vector<mat> m);
  void VectorToMatrix(vec p, std::vector<mat> &m); 

// minimization functions - variable metric method
// from Numerical Recipes, chap. 10.7
  double linesearch(vec xold, double fold, vec gold, vec p, vec &xnew, double stepmax);

// update of the inverse of the Hesse matrix
// using dedicated function for performance 
  void UpdateHesseInverse(mat &hesseinv,double fac1, double fac2, double fac3, vec &xi, vec &hdg, vec &dgrad);

public:
  NNetwork(std::string name, std::vector<int> len) {
    this->name = name;

    if(len.size() < 3) {
      cout << "Invalid network initialization." << endl;
      cout << "Need at least three layers: one input layer, one hidden layer and one output layer." << endl;
      return;
    }
    l_size = len;
    n = l_size.size();

// set regularization parameter
    lambda = 0.1;

// setup layers
    for(int i = 0; i < n - 1; ++i) {
      z.push_back(zeros<vec>(l_size[i]));
      a.push_back(zeros<vec>(l_size[i] + 1));
      delta.push_back(zeros<vec>(l_size[i]));
    }
    z.push_back(zeros<vec>(l_size[n-1])); // no bias in output layer
    a.push_back(zeros<vec>(l_size[n-1]));
    delta.push_back(zeros<vec>(l_size[n-1]));

// setup matrices
//    arma_rng::set_seed_random(); // init rng
    for(int i = 0; i < n - 1; ++i) {
      theta.push_back(randu<mat>(l_size[i + 1],l_size[i] + 1));
      theta[i] = 2*theta[i] - 1.0;
      theta_grad.push_back(zeros<mat>(l_size[i + 1],l_size[i] + 1));
    }

// setup minimization parameter
    MIN_MAXITER = 1000;    
  }

// load network from file  
  NNetwork(std::string fname) {
    std::ifstream infile(fname, ios::in);

    int tmpval;

    infile >> n;
    for(int i = 0; i < n; ++i) {
      infile >> tmpval;
      l_size.push_back(tmpval);
    }

// set regularization parameter
    lambda = 0.1;

// setup layers
    for(int i = 0; i < n - 1; ++i) {
      z.push_back(zeros<vec>(l_size[i]));
      a.push_back(zeros<vec>(l_size[i] + 1));
      delta.push_back(zeros<vec>(l_size[i]));
    }
    z.push_back(zeros<vec>(l_size[n-1])); // no bias in output layer
    a.push_back(zeros<vec>(l_size[n-1]));
    delta.push_back(zeros<vec>(l_size[n-1]));

// setup matrices
    int nr,nc;
    for(int i = 0; i < n - 1; ++i) {
      infile >> nr >> nc;
      theta.push_back(zeros<mat>(l_size[i+1],l_size[i] + 1));
      for(int row = 0; row < nr; ++row)
        for(int col = 0; col < nc; ++col)
          infile >> theta[i].at(row,col);
      theta_grad.push_back(zeros<mat>(l_size[i + 1],l_size[i] + 1));
    }

// setup minimization parameter
    MIN_MAXITER = 1000;    
  }

  ~NNetwork() {
  }

// setup a network name
  void SetName(std::string name);

// get network name
  std::string GetName();

// get number of layers
  int GetNumLayers();

// check if network is initiaized
  bool IsEmpty();

// output network status to terminal
  void PrintStatus(bool verbose_layers,bool verbose_matrices);

// set the value of the input layer, and take care of bias  
  bool SetInputLayer(std::vector<double> in);

// print output layer to terminal
  void PrintOutputLayer();  

// driver function for gradient checking
  void CheckGradientDriver(std::vector<Digit> data);

// minimization algorithm
  double MinimizeNetwork();

// test network and compare
  void TestNetwork(bool verbose);  
  
// load training data into network
  void LoadTrainingData(std::vector<Digit> data);

// load testing data into network
  void LoadTestingData(std::vector<Digit> data);

// save network 
  void SaveNetwork(std::string fname);

};

#endif