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

Vector Section::sqrt() {
    std::vector<double> ret(begin, end);
    std::for_each(ret.begin(), ret.end(), [](auto& v) { v = std::sqrt(v); }); 
    return Vector(std::move(ret));
}

Vector operator/(const Section& a, const double& b) {
    if (b == 0)
        throw std::invalid_argument("Can not divide by zero.");
    return operator_b<std::divides<double>>(a, b);
}
Vector operator+(const Section& a, const double& b) {
    return operator_b<std::plus<double>>(a, b);
}
Vector operator*(const Section& a, const double& b) {
    return operator_b<std::multiplies<double>>(a, b);
}
Vector operator*(const double& a, const Section& b) {
    return b * a;
}

Vector operator/(const double& a, const Section& b) {
    std::vector<double> ret(b.end - b.begin);
    std::transform(b.begin, b.end, ret.begin(), [&a](const auto& v){ return std::divides<double>{}(a, v); });
    return Vector(std::move(ret));
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