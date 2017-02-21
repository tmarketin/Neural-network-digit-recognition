CXX = g++
OPTS = -std=c++11 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -Wold-style-cast -Wcast-align -Wunused -Wconversion -Wsign-conversion -O3 -ftree-vectorize
OPTSARMA = -larmadillo
DEPS = mnist.h nnet.h

all: nnet_main.o nnet.o mnist.o
	$(CXX) $(OPTS) -o run_nnet nnet_main.o nnet.o mnist.o $(OPTSARMA)

nnet_main.o: nnet_main.cc mnist.h nnet.h
	$(CXX) $(OPTS) -c nnet_main.cc

mnist.o: mnist.cc mnist.h
	$(CXX) $(OPTS) -c mnist.cc

nnet.o: nnet.cc nnet.h mnist.h
	$(CXX) $(OPTS) $(OPTSARMA) -c nnet.cc

clean:
	rm *.o run_nnet