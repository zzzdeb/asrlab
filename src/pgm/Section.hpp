#ifndef __SECTION_HPP__
#define __SECTION_HPP__

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>

class Vector;

/* Interface to work on between two iterators end and begin */
class Section {
public:
    using ItType = std::vector<double>::iterator;
    Section(ItType begin, ItType end) : begin(begin), end(end) {}
    size_t size() const { return end - begin; }
    Section& operator=(const Section& other)
    {
        if (other.size() != size())
            throw std::invalid_argument("Dimension does not match.");
        std::copy(other.begin, other.end, begin);
        return *this;
    }
    auto operator+=(const Section& other) {
        return std::transform(begin, end, other.begin, begin, std::plus<double>{});
    }
    Vector log();
    Vector square();
    Vector sqrt();
    double sum() { return std::accumulate(begin, end, 0.);}

    ItType begin;
    ItType end;
};

class Vector : public Section {
public:
    Vector(std::vector<double> v):
        Section(v.begin(), v.end()), v(std::move(v)) {} 
    template<class T>
    Vector(T* begin, T* end): Section(v.begin(), v.end()), v(begin, end) {
        this->begin = v.begin();
        this->end = v.end();
    }
    std::vector<double> v;
};

template<class BinaryF>
Vector operator_b(const Section& a, const double& b) {
    std::vector<double> ret(a.end - a.begin);
    std::transform(a.begin, a.end, ret.begin(), [&b](const auto& v){ return BinaryF{}(v, b); });
    return Vector(std::move(ret));
}
Vector operator/(const Section& a, const double& b);
Vector operator+(const Section& a, const double& b);
Vector operator*(const Section& a, const double& b);
Vector operator*(const double& a, const Section& b);
Vector operator/(const double& a, const Section& b);

template<class BinaryF>
Vector operator_b(const Section& a, const Section& b) {
    std::vector<double> ret(b.begin, b.end);
    std::transform(a.begin, a.end, ret.begin(), ret.begin(), BinaryF{});
    return Vector(std::move(ret));
}
Vector operator+(const Section& a, const Section& b);
Vector operator*(const Section& a, const Section& b);
Vector operator-(const Section& a, const Section& b);

#endif /* __SECTION_HPP__ */