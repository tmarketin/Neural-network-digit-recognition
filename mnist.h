#ifndef MNIST_H_
#define MNIST_H_

#include <iostream>
#include <fstream>
#include <vector>

class Digit {
private:
  unsigned char label;
  unsigned char x_size,y_size;
  std::vector<double> num;
  std::vector<double> layer;

public:
  Digit() : label {255}, x_size {0}, y_size {0}, num {}, layer {} {
  }

  int GetDigit();

  void SetLabel(unsigned char k);
  std::vector<double> GetLabel() const;

  void SetSize(unsigned int x, unsigned int y);

  bool SetValue(std::vector<unsigned char> v);
  std::vector<double> GetValue() const;

  friend std::ostream& operator<< (std::ostream &out, const Digit &dig);
};

#endif