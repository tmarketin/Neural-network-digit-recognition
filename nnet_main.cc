#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

#include "nnet.h"
#include "mnist.h"

using std::cout;
using std::endl;
using std::setw;
using std::ifstream;
using std::ios;

int main() {
  const char *train_labels = "Data/train-labels-idx1-ubyte";
  const char *train_images = "Data/train-images-idx3-ubyte";
  const char *test_labels = "Data/t10k-labels-idx1-ubyte";
  const char *test_images = "Data/t10k-images-idx3-ubyte";

  unsigned char tmplabel;
  std::vector<unsigned char> tmpvec;
  Digit tmpdigit;

  std::vector<Digit> data_train;
  std::vector<Digit> data_test;

  ifstream ftrain_labels(train_labels, ios::in | ios::binary);
  ifstream ftrain_images(train_images, ios::in | ios::binary);
  ifstream ftest_labels(test_labels, ios::in | ios::binary);
  ifstream ftest_images(test_images, ios::in | ios::binary);

  unsigned int n_train_images,n_train_labels,n_test_images,n_test_labels;
  unsigned int n_image_x,n_image_y;

  for(int i = 0; i < 4; i++) {
    ftrain_labels.get();
    ftrain_images.get();
    ftest_images.get();
    ftest_labels.get();
  }

// read training data
// number of labels in the file
  ftrain_labels.read(reinterpret_cast<char *>(&n_train_labels),sizeof(n_train_labels));
  n_train_labels = __builtin_bswap32(n_train_labels); // convert endian due to file format

// number of images in the file
  ftrain_images.read(reinterpret_cast<char *>(&n_train_images),sizeof(n_train_images));
  n_train_images = __builtin_bswap32(n_train_images); // convert endian due to file format

// number of pixels in x dimension per image
  ftrain_images.read(reinterpret_cast<char *>(&n_image_x),sizeof(n_image_x));
  n_image_x = __builtin_bswap32(n_image_x); // convert endian due to file format

// number of pixels in y dimension per image
  ftrain_images.read(reinterpret_cast<char *>(&n_image_y),sizeof(n_image_y));
  n_image_y = __builtin_bswap32(n_image_y); // convert endian due to file format

  for(unsigned int i = 0; i < n_train_labels; i++) {
    tmplabel = static_cast<unsigned char>(ftrain_labels.get());

    tmpvec.clear();
    for(unsigned int row = 0; row < n_image_y; ++row)
      for(unsigned int col = 0; col < n_image_x; ++col)
        tmpvec.push_back(static_cast<unsigned char>(ftrain_images.get()));

    tmpdigit.SetLabel(tmplabel);
    tmpdigit.SetSize(n_image_x,n_image_y);
    tmpdigit.SetValue(tmpvec);

    data_train.push_back(tmpdigit);
  }

// read testing data
// number of labels in the file
  ftest_labels.read(reinterpret_cast<char *>(&n_test_labels),sizeof(n_test_labels));
  n_test_labels = __builtin_bswap32(n_test_labels); // convert endian due to file format

// number of images in the file
  ftest_images.read(reinterpret_cast<char *>(&n_test_images),sizeof(n_test_images));
  n_test_images = __builtin_bswap32(n_test_images); // convert endian due to file format

// number of pixels in x dimension per image
  ftest_images.read(reinterpret_cast<char *>(&n_image_x),sizeof(n_image_x));
  n_image_x = __builtin_bswap32(n_image_x); // convert endian due to file format

// number of pixels in y dimension per image
  ftest_images.read(reinterpret_cast<char *>(&n_image_y),sizeof(n_image_y));
  n_image_y = __builtin_bswap32(n_image_y); // convert endian due to file format

  for(unsigned int i = 0; i < n_test_labels; ++i) {
    tmplabel = static_cast<unsigned char>(ftest_labels.get());

    tmpvec.clear();
    for(unsigned int row = 0; row < n_image_y; ++row)
      for(unsigned int col = 0; col < n_image_x; ++col)
        tmpvec.push_back(static_cast<unsigned char>(ftest_images.get()));

    tmpdigit.SetLabel(tmplabel);
    tmpdigit.SetSize(n_image_x,n_image_y);
    tmpdigit.SetValue(tmpvec);

    data_test.push_back(tmpdigit);
  }

  std::vector<arma::uword> lsizes {784,25,10};

  NNetwork neuralnet("Test network",lsizes);

  neuralnet.LoadTrainingData(data_train);
  neuralnet.LoadTestingData(data_test);

  neuralnet.PrintStatus(false,false);

  neuralnet.TestNetwork(false);

  neuralnet.MinimizeNetwork();

  neuralnet.TestNetwork(false);

  neuralnet.SaveNetwork("network_15_15.dat");

  return 0;
}