//
// Created by zzz on 12/7/22.
//
#include "Types.hpp"

Vect valarr_to_vect(const std::valarray<float>& v) {
    Vect ret(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        ret[i] = v[i];
    return ret;
}
Matr valarr_to_matr(const std::valarray<float>& v, const std::gslice& slice) {
    Matr mat(slice.size()[0], slice.size()[1]);
    for (size_t i = 0; i < slice.size()[0]; ++i)
        for (size_t j = 0; j < slice.size()[1]; ++j)
            mat(i, j) = v[slice.start() + i * slice.stride()[0] + j * slice.stride()[1]];
    return mat;
}

std::valarray<float> matr_to_valarr(const Matr& m) {
    std::valarray<float> ret(m.rows() * m.cols());
    size_t k = 0;
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j, k++)
            ret[k] = m(i, j);
    return ret;
}

std::ostream& operator<<(std::ostream& out, const std::valarray<float>& val) {
    for (int i = 0; i < 10; ++i) {
        out << val[i] << " ";
    }
}
