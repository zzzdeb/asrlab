#ifndef __SECTION_HPP__
#define __SECTION_HPP__

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cassert>
#include <cmath>

#include <iostream>

template <typename IterT>
std::ostream& cprint(const IterT& begin, const IterT& end, std::ostream& out)
{
    out << "[";
    for (auto iter = begin; iter < end; iter++)
        out << *iter << ",";
    out << "]";
    return out;
}
template <typename T>
std::ostream& cprint(const std::vector<T> &container, std::ostream& out = std::cout)
{
    return cprint(container.cbegin(), container.cend(), out);
}

class Vector;

template <class ItType>
class SectionBase {
    public:
    SectionBase(ItType begin, ItType end) : begin(begin), end(end) {}
    SectionBase(double* begin, double* end) : begin(begin), end(end) {}
    size_t size() const { return end - begin; }
    double operator[] (size_t i)
    {
        assert(i < end - begin);
        return *(begin + i);
    }
    Vector operator-() const;
    operator std::vector<double>() const { return {begin, end};}
    Vector log() const;
    Vector exp() const;
    Vector square() const;
    Vector nonzero() const;
    Vector sqrt() const;
    Vector sub(size_t s, size_t e) const;
    double sum() const { return std::accumulate(begin, end, 0.);}
    ItType min() const { return std::min_element(begin, end);}

    ItType begin;
    ItType end;

};

class ConstSection : public SectionBase<std::vector<double>::const_iterator> {
public:
    using BaseT = SectionBase<std::vector<double>::const_iterator>;
    using BaseT::BaseT;
};

/* Interface to work on between two iterators end and begin */
class Section : public SectionBase<std::vector<double>::iterator> {
public:
    using BaseT = SectionBase<std::vector<double>::iterator>;
    using BaseT::BaseT;
    size_t size() const { return end - begin; }
    Section& operator=(const Section& other)
    {
        if (other.size() != size())
            throw std::invalid_argument("Dimension does not match.");
        std::copy(other.begin, other.end, begin);
        return *this;
    }
    bool operator==(const double v) {
        for(auto it = begin; it != end; it++)
            if (*it != v)
                return false;
        return true;
    }
    auto operator+=(const Section& other) {
        return std::transform(begin, end, other.begin, begin, std::plus<double>{});
    }
    auto operator/=(const double& d) {
        return std::for_each(begin, end, [&d](auto& v) { v /= d;});
    }
    double &operator[](size_t i)
    {
        assert(i < end - begin);
        return *(begin + i);
    }
    Vector operator-() const;
    operator std::vector<double>() const { return {begin, end};}
};

template<class T>
std::ostream& operator<<(std::ostream& os, const SectionBase<T>& s) {
    return cprint(s.begin, s.end, os);
}

class Vector : public Section {
public:
    template<typename... Params>
    Vector(Params... args) : Section(begin, end), v(std::forward<Params>(args)...) {
        this->begin = v.begin();
        this->end = v.end();
    }
    Vector& operator=(const Vector& other) {
        Section::operator=(other);
        return *this;
    }
    std::vector<double> v;
};

template<class BinaryF, class T>
Vector operator_b(const SectionBase<T>& a, const double& b) {
    std::vector<double> ret(a.end - a.begin);
    std::transform(a.begin, a.end, ret.begin(), [&b](const auto& v){ return BinaryF{}(v, b); });
    return Vector(std::move(ret));
}
Vector operator/(const Section& a, const double& b);
Vector operator+(const Section& a, const double& b);
Vector operator-(const Section& a, const double& b);
Vector operator*(const Section& a, const double& b);
Vector operator*(const double& a, const Section& b);
Vector operator/(const double& a, const Section& b);

template<class BinaryF, class T, class T2>
Vector operator_b(const SectionBase<T>& a, const SectionBase<T2>& b) {
    std::vector<double> ret(b.begin, b.end);
    std::transform(a.begin, a.end, ret.begin(), ret.begin(), BinaryF{});
    return Vector(std::move(ret));
}
template<class T, class T2>
Vector operator+(const SectionBase<T>& a, const SectionBase<T2>& b) {
    return operator_b<std::plus<double>>(a, b);
}
template<class T, class T2>
Vector operator*(const SectionBase<T>& a, const SectionBase<T2>& b) {
    return operator_b<std::multiplies<double>>(a, b);
}
template<class T, class T2>
Vector operator-(const SectionBase<T>& a, const SectionBase<T2>& b) {
    return operator_b<std::minus<double>>(a, b);
}

template<class T, class T2>
bool operator==(const SectionBase<T>& a, const SectionBase<T2>& b) {
    if (a.size() != b.size())
        return false;
    auto bit = b.begin;
    for (auto it = a.begin; it != a.end; it++, bit++)
        if (*it != *bit)
            return false;
    return true;
}

template<class T>
bool operator==(const std::vector<double>& a, const SectionBase<T>& b) {
    return ConstSection(a.cbegin(), a.cend()) == b;
}

template <class T>
Vector SectionBase<T>::log() const { 
    std::vector<double> ret(size());
    std::transform(begin, end, ret.begin(), [](const auto& v){return std::log(v);}); 
    return Vector(std::move(ret));
}
template <class T>
Vector SectionBase<T>::exp() const { 
    std::vector<double> ret(size());
    std::transform(begin, end, ret.begin(), [](const auto& v){return std::exp(v);}); 
    return Vector(std::move(ret));
}

template <class T>
Vector SectionBase<T>::square() const {
    return *this * *this;
}

template <class T>
Vector SectionBase<T>::nonzero() const {
    std::vector<double> ret(begin, end);
    std::for_each(ret.begin(), ret.end(), [](auto& v) { if (std::abs(v -0) < 0.0025) v = 0.0025; }); 
    return Vector(std::move(ret));
}

template <class T>
Vector SectionBase<T>::sqrt() const {
    std::vector<double> ret(begin, end);
    std::for_each(ret.begin(), ret.end(), [](auto& v) { v = std::sqrt(v); }); 
    return Vector(std::move(ret));
}
template <class T>
Vector SectionBase<T>::sub(size_t s, size_t e) const {
    std::vector<double> ret(begin+s, begin+e);
    return Vector(std::move(ret));
}

#endif /* __SECTION_HPP__ */
