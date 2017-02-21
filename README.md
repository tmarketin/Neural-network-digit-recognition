A simple neural network implementation for digit recognition

This repository contains the a simple implementation of handwritten digit recognition using neural network. Data is taken from the MNIST database at <a href="http://yann.lecun.com/exdb/mnist/"> MNIST website</a>. Training and test files should be places in a "Data" subdirectory. 

There are two options to set uo the network:
1. Construct the <em>NNetwork</em> class with a string and a vector of integers. The string represents the name of the network, and the integers represent the sizes of the netwrk layers. Length of the vector corresponds to the number of layers (including input and output layers). 
2. Construct the class with a string which represents the file name from which to load the entire network. The file can be generated after minimization with the <code>.SaveNetwork("filename.dat")</code> method. 

<code>.TestNetwork(bool)</code> evaluates all the training and test digits with the network as it is, and outputs the results. If the bool is set to <em>true</em> additional output is generated for each digit - it is very verbose and useless for larger data sets. 

<code>.MinimizeNetwork()</code> starts the minimization procedure. The cost function is simply the sum of deviations for each training data point plus regularization of all parameters. By default, the regularization parameter lambda is set to 0.1, but may be changed in the network constructors. Minimization procedure uses the variable metrich method based on the chapter 10.7 of Numerical Recipes.

Linear algebra is handled by the <a href="http://arma.sourceforge.net/">Armadillo linear algebra library</a>. It is generally included in most linux distributions (though not necessarily the latest version). During compilation the <code>-larmadillo</code> switch is used.

As it is, the code will construct a neural network with two hidden layers of size 15, minimize the netowrk and test on all the training and testing data. One should get around 95% accuracy on the training data, and 94% on the testing data. 

While writing I wasn't perfectly consistent in naming variables and functions, and the style is sometimes problematic (member variables without prefixes and similar). 