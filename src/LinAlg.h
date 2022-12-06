//
// Created by zzz on 12/4/22.
//

#ifndef ASRLAB_LINALG_H
#define ASRLAB_LINALG_H

#include <utility>
#include <valarray>
#include <memory>
#include <array>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <limits>

namespace {
    bool operator==(const std::valarray<size_t>& a, const std::valarray<size_t>& b) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }
}

namespace linalg {

using BaseT = std::valarray<float>;

class Matrix;
class Vector;


class LinObj {
public:
    struct Sect {
        Sect(size_t s, size_t e) : s(s), e(e) {}
        Sect(size_t i) : s(i), e(i+1) {}
        bool operator==(const Sect& o) const {return s == o.s && e == o.e; }

        size_t s;
        size_t e;
    };

    LinObj(std::shared_ptr<BaseT> data, std::valarray<size_t> shape, const size_t& start = 0) : data(std::move(data)) {
        slice_from_param(std::move(shape), start);
    }
    LinObj(const std::valarray<size_t>& shape) :
        LinObj(std::make_shared<BaseT>(std::accumulate(begin(shape), end(shape), 1, std::multiplies<size_t>{})),
               shape)
        {
        }
    LinObj(std::shared_ptr<BaseT> data, std::gslice const& slice) : data(data), slice(slice) {
    }
    LinObj& operator=(const LinObj& other) {
        assert(other.slice.size().size() == slice.size().size());
        for (size_t i = 0; i < slice.size().size(); i++)
            assert(other.slice.size()[i] == slice.size()[i]);
        BaseT tmp = other.get();
        get() = tmp;
        return *this;
    }
    void reset(std::shared_ptr<BaseT> datap, std::valarray<size_t> shape, const size_t& start = 0) {
        data = std::move(datap);
        slice_from_param(std::move(shape), start);
    }
    void slice_from_param(std::valarray<size_t> shape, const size_t& start = 0) {
        size_t stride = 1;
        std::valarray<size_t> strides(shape.size());
        for (size_t i = shape.size(); i > 0; i--) {
            strides[i -1] = stride;
            stride *= shape[i -1];
        }
        slice = std::gslice(start, shape, strides);
    }

    std::gslice_array<float> get() const { return (*data)[slice]; }
    std::valarray<size_t> shape() const { return slice.size();}

    LinObj& operator*=(const LinObj& other) {
        assert(shape() == other.shape());
        get() *= other.get();
        return *this;
    }
    LinObj& operator+=(const LinObj& other) {
        assert(shape() == other.shape());
        get() += other.get();
        return *this;
    }
    void tanh();
    void relu();
    void sigmoid();

    void dtanh();
    void drelu();
    void dsigmoid();
    LinObj operator*(const LinObj& other) const {
        BaseT ret = get();
        ret *= BaseT(other.get());
        return {std::make_shared<BaseT>(std::move(ret)), shape()};
    }
    LinObj operator-(const LinObj& other) const {
        BaseT ret = get();
        ret -= BaseT(other.get());
        return {std::make_shared<BaseT>(std::move(ret)), shape()};
    }
    void nonzero() {
        BaseT tmp = get();
        for(auto& v : tmp)
            if (v==0)
                v = 0.0000001;
        get() = tmp;
    }

    std::gslice slice;
    std::shared_ptr<BaseT> data;
};

static const LinObj::Sect ALL(0, std::numeric_limits<size_t>::infinity());

class Vector : public LinObj {
public:
    using LinObj::LinObj;
//    Vector(std::shared_ptr<BaseT> data, const std::slice& slicep) : LinObj(data, std::gslice(slicep.start(), {slicep.size()}, {slicep.stride()})) {}
    float dot(const Vector& v) const;
    Vector(const LinObj& other) : LinObj(other) {
        assert(slice.size().size() == 1);
    }
    Vector dot(const Matrix& m) const;
    void softmax() {
        BaseT v = get();
        v = std::exp(v);
        v /= v.sum();
        get() = v;
    }
    const float& at(size_t i) const { return (*data)[slice.start() + i * slice.stride()[0]]; }
    Matrix outer(const Vector& v) const;
};

std::ostream& operator<<(std::ostream& out, const Vector& v);

class Matrix : public LinObj {
public:
    using LinObj::LinObj;
//    Matrix(std::shared_ptr<BaseT> data, const std::array<size_t, 2>& dims, const size_t& start = 0) : LinObj(data) {}
    Matrix(const LinObj& other) : LinObj(other) {
        assert(slice.size().size() == 2);
    }

    void softmax() {
        for (int i = 0; i < shape()[0]; ++i) {
            Vector v = operator[](i);
            v.softmax();
        }
    }
    Matrix& radd(const Vector& v) {
        for (int i = 0; i < shape()[0]; ++i)
            operator[](i) += v;
        return *this;
    }

    float& at(size_t i, size_t j);
    Vector operator[](size_t j) const;
    Vector col(size_t j) const;

    Matrix dot(const Matrix& m) const {
        assert(slice.size()[1] == m.slice.size()[0]);
        Matrix ret({shape()[0], m.shape()[1]});
        for (size_t j = 0; j < m.shape()[1]; j++)
            for (size_t i = 0; i < shape()[0]; i++)
                ret.at(i, j) = operator[](i).dot(m.col(j));
        return ret;
    }
    Matrix transpose() const {
        return {data, {slice.start(), {slice.size()[1], slice.size()[0]}, {slice.stride()[1], slice.stride()[0]}}};
    }
    Vector sumy() const {
        Vector ret({slice.size()[0]});
        for (size_t i = 0; i < slice.size()[1]; i++) {
            ret += col(i);
        }
        return ret;
    }
};

Matrix operator+(const Matrix& m, const Vector& v);

class Tensor : public LinObj {
public:
    using LinObj::LinObj;
    const Tensor operator() (const Sect& a, const Sect& b, const Sect& c) const;
    Tensor operator() (const Sect& a, const Sect& b, const Sect& c);
    void resize(size_t x, size_t y, size_t z) {
        data->resize(x * y * z);
        slice = std::gslice(0, {x, y, z}, {z*y,z,1});
    }

    Matrix mat() const;

    //   auto dot = [&](std::valarray<float>& W, std::valarray<float>& x, std::valarray<float>& out) {
    //     const auto& outlen = output_size_;
    //     for (size_t i = 0; i < outlen; i++)
    //     {
    //       out[i] = (W[std::slice(i, 1, outlen)] * x).sum();
    //     }
    //   };
    //   output[std::slice(s, output_size_, 1)] = dot(W, x, out);
    //   output[std::slice(s, output_size_, 2)] += b;
    // }


    // size_t index =   slice.start()
    //                + (center_time + delta - trunc_offsets.first)          * slice.stride()[0]
    //                + i                                                    * slice.stride()[1]
    //                + ((delta + context_frames_) * base_feature_size_ + f) * slice.stride()[2];
    // float val = (*iters.first)[f];
    // output[index] = val;

    // output[slice] = res;
    // params_
    // input_buffer_
    // input;

    // std::gslice(offset,
    //             {   max_seq_length_, batch_size_, feature_size_},
    //             {batch_size_ * fdim,        fdim,             1})));
};
}

#endif //ASRLAB_LINALG_H
