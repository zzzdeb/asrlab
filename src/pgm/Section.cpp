#include "Section.hpp"

#include <cmath>

Vector Section::log() { 
    std::vector<double> ret(begin, end);
    std::for_each(ret.begin(), ret.end(), [](auto& v) { v = std::log(v); }); 
    return Vector(std::move(ret));
}
Vector Section::square() {
    return *this * *this;
}

Vector operator/(const Section& a, const double& b) {
    return operator_b<std::divides<double>>(a, b);
}
Vector operator+(const Section& a, const double& b) {
    return operator_b<std::plus<double>>(a, b);
}
Vector operator*(const Section& a, const double& b) {
    return operator_b<std::multiplies<double>>(a, b);
}

Vector operator+(const Section& a, const Section& b) {
    return operator_b<std::plus<double>>(a, b);
}
Vector operator*(const Section& a, const Section& b) {
    return operator_b<std::multiplies<double>>(a, b);
}
Vector operator-(const Section& a, const Section& b) {
    return operator_b<std::minus<double>>(a, b);
}