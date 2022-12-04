//
// Created by zzz on 12/4/22.
//
#include "LinAlg.h"
#include <valarray>
#include <cassert>

namespace linalg {

    void LinObj::tanh() {
        std::valarray<float>  v = get();
        get() = std::tanh(v);
    }
    void LinObj::dtanh() {
        std::valarray<float>  v = get();
        get() = 1 - v * v;
    }
    void LinObj::relu() {
        BaseT d = get();
        for(auto& v : d)
            if (v < 0)
                v = 0;
        get() = d;
    }
    void LinObj::drelu() {
        BaseT d = get();
        for(auto& v : d)
            if (v <= 0)
                v = 0;
            else
                v = 1;
        get() = d;
    }
    void LinObj::sigmoid() {
        // 1/(1+e^-x)
        std::valarray<float>  v = get();
        get() = 1 / (1 + std::exp(-v));
    }
    void LinObj::dsigmoid() {
        std::valarray<float>  v = get();
        get() = v * (1 - v);
    }

    float& Matrix::at(size_t i, size_t j) {
        return (*data)[slice.start() + i * slice.stride()[0] + j * slice.stride()[1]];
    }
    Vector Matrix::operator[](size_t j) const {
        return {data, std::gslice{slice.start() + slice.stride()[0] * j, {shape()[1]}, {slice.stride()[1]}}};
    }
    Vector Matrix::col(size_t j) const {
        return {data, std::gslice(slice.start() + slice.stride()[1] * j, {shape()[0]}, {slice.stride()[0]})};
    }

    const Tensor Tensor::operator() (const Sect& a, const Sect& b, const Sect& c) const {
        std::valarray<size_t> size(3);
        size_t startl = slice.start() + a.s * slice.stride()[0] + b.s * slice.stride()[1] + c.s * slice.stride()[2];
        size[0] = a == ALL ? slice.size()[0] : a.e - a.s;
        size[1] = b == ALL ? slice.size()[1] : b.e - b.s;
        size[2] = c == ALL ? slice.size()[2] : c.e - c.s;
        return {data, std::gslice(startl, size, slice.stride())};
    }
    Tensor Tensor::operator() (const Sect& a, const Sect& b, const Sect& c) {
        std::valarray<size_t> size(3);
        size_t startl = slice.start() + a.s * slice.stride()[0] + b.s * slice.stride()[1] + c.s * slice.stride()[2];
        size[0] = a == ALL ? slice.size()[0] : a.e - a.s;
        size[1] = b == ALL ? slice.size()[1] : b.e - b.s;
        size[2] = c == ALL ? slice.size()[2] : c.e - c.s;
        return {data, std::gslice(startl, size, slice.stride())};
    }

    auto find(const std::valarray<size_t>& s, size_t v) {
        auto b = begin(s);
        while(b != end(s)) {
            if (*b == v)
                break;
            b++;
        }
        return b;
    }
    std::valarray<size_t> remove(const std::valarray<size_t>& va, size_t dist) {
        std::valarray<size_t> size(va.size() - 1);
        auto b = begin(va);
        std::copy(b, b + dist, begin(size));
        std::copy(b + dist + 1, end(va), begin(size) + dist);
        return size;
    }

    Tensor::operator Matrix() const {
        auto it = find(slice.size(), 1ul);
        assert(it != end(slice.size()));
        auto dist = it - begin(slice.size());
        std::valarray<size_t> size = slice.size();
        size = remove(size, dist);
        std::valarray<size_t> stride = slice.stride();
        stride = remove(stride, dist);
        return Matrix(data, std::gslice(slice.start(), size, stride));
    }

    float Vector::dot(const Vector& v) const {
        return (std::valarray<float>(get()) * std::valarray<float>(v.get())).sum();
    }

    Matrix operator+(const Matrix& m, const Vector& v) {
        Matrix ret(m);
        for (int i = 0; i < ret.shape()[1]; ++i) {
            ret.col(i) += v;
        }
        return ret;
    }

    Vector Tensor::at(size_t i, size_t j) const {
        return {data, std::gslice{slice.start() + i * slice.size()[0] + j * slice.size()[1], {slice.size()[2]}, {slice.stride()[2]}}};
    }

    Matrix Vector::outer(const Vector& v) const {
        Matrix ret({shape()[0], v.shape()[0]});
        for(size_t i = 0; i < shape()[0]; i++) {
            BaseT arr = v.get();
            ret[i].get() = at(i) * arr;
        }
        return ret;
    }

    Vector Tensor::sumx() const {
        Vector ret({slice.size()[2]});
        ret.get() = 0;
        for(size_t i = 0; i < slice.size()[0]; i++)
            for(size_t j = 0; j < slice.size()[1]; j++) {
                BaseT tmp = at(i, j).get();
                ret.get() += tmp;
            }
        return ret;
    }

    std::ostream& operator<<(std::ostream& out, const Vector& v) {
        BaseT tmp = v.get();
        for(const auto& a: tmp)
            out << a << " ";
        return out;
    }
}