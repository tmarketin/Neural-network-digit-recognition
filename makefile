CXX = g++
OPTS = -std=c++11 -c -O3 -ftree-vectorize
OPTSARMA = -larmadillo
DEPS = mnist.h nnet.h

nnet_main: nnet_main.o nnet.o mnist.o
	$(CXX) -std=c++11 -O3 -ftree-vectorize -o run_nnet nnet_main.o nnet.o mnist.o $(OPTSARMA)

nnet_main.o: nnet_main.cc mnist.h nnet.h
	$(CXX) $(OPTS) nnet_main.cc

mnist.o: mnist.cc mnist.h
	$(CXX) $(OPTS) mnist.cc

nnet.o: nnet.cc nnet.h mnist.h
	$(CXX) $(OPTS) $(OPTSARMA) nnet.cc
