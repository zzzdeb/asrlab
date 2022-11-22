#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include "Section.hpp"

#include <string>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace pgm
{

    class Matrix
    {
    public:
        Matrix() = default;

        Matrix(size_t rows, size_t columns, double init = 0) : 
            rows_(rows),
            columns_(columns),
            data(rows_ * columns_, init) {}

        Matrix& operator=(const double& v)
        {
            std::fill(data.begin(), data.end(), v);
            return *this;
        }

        void from_file(const std::string &path);
        void to_file(const std::string &path) const;
        void add_row();
        void add_row(std::vector<double> &row, size_t times = 1);
        void add_row(const Section& row, size_t times = 1);
        void transpose();

        auto row_begin(size_t i) { return data.begin() + columns_ * i; }
        auto row_end(size_t i) { return data.begin() + columns_ * (i + 1); }
        void resize(const std::pair<size_t, size_t>& size) { rows_ = size.first; columns_ = size.second; data.resize(rows_ * columns_);}
        std::pair<size_t, size_t> size() const { return {rows_, columns_};}
        auto& operator()(size_t i, size_t j) { return data.at(i * columns_ + j); }
        const auto& operator()(size_t i, size_t j) const { return data.at(i * columns_ + j); }
        Section operator[](size_t i) { assert(i < rows_); return {row_begin(i), row_end(i)};}

        auto row_cbegin(size_t i) const { return data.cbegin() + columns_ * i; }
        auto row_cend(size_t i) const { return data.cbegin() + columns_ * (i + 1); }
        ConstSection operator[](size_t i) const { assert(i < rows_); return {row_cbegin(i), row_cend(i)};}

        size_t rows_{0};
        size_t columns_{0};
        std::vector<double> data{};
    };

    std::ostream& operator<<(std::ostream& os, const Matrix& m);
}

#endif /* __MATRIX_HPP__ */