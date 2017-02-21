#include <iostream>
#include <iomanip>
#include <fstream>

#include "mnist.h"

using std::cout;
using std::endl;
using std::setw;

void Digit::SetLabel(unsigned char k) {
  this->label = k;

  this->layer.clear();
  for(int i = 0; i < 10; i++)
    this->layer.push_back(0.0);

  this->layer[k] = 1.0;

  return;
}

std::vector<double> Digit::GetLabel() const {
  return this->layer;
}

int Digit::GetDigit() {
  return static_cast<int>(this->label);
}

void Digit::SetSize(unsigned int x, unsigned int y) {
  x_size = static_cast<unsigned char>(x);
  y_size = static_cast<unsigned char>(y);

  return;
}

bool Digit::SetValue(std::vector<unsigned char> v) {
  size_t len = v.size();

  if(len != x_size*y_size) {
    cout << "Mismatch in vector size!" << endl;
    return false;
  }

  std::vector<double> tmp;
  tmp.clear();

  for(size_t i = 0; i < v.size(); i++)
    tmp.push_back(static_cast<double>(v[i])/255.0);

  this->num = tmp;

  return true;
}

std::vector<double> Digit::GetValue() const {
  return this->num;
}

std::ostream& operator<< (std::ostream &out, const Digit &digit) {
  out << "Value: " << static_cast<int>(digit.label) << endl;
  for(size_t i = 0; i < digit.layer.size(); i++)
    out << digit.layer[i] << " ";
  out << endl;

  for(size_t i = 0; i < digit.y_size; i++) {
    for(size_t j = 0; j < digit.x_size; j++)
      out << setw(4) << static_cast<int>(digit.num[i*digit.x_size + j]*255);
    out << endl;
  }

  return out;
}