#include "Matrix.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

namespace pgm
{
    namespace
    {
        // Based on the implementation detail of to_file, counts '\n' and ' '
        void get_shape(const std::stringstream &ss, size_t &rows, size_t &colums)
        {
            std::string content = ss.str();
            std::string line("");
            rows = std::count(content.cbegin(), content.cend(), '\n');
            colums = std::count(content.cbegin(), content.cend(), ' ') / rows;
        }
    }

    void Matrix::from_file(const std::string &path)
    {
        std::ifstream infile(path);
        std::stringstream ss;
        ss << infile.rdbuf();
        // get shape
        get_shape(ss, rows_, columns_);
        // Following lines : data
        for (size_t row = 0; row < rows_; ++row)
            for (size_t col = 0; col < columns_; ++col)
                ss >> data.at(row * columns_ + col);
        infile.close();
    }

    void Matrix::to_file(const std::string &path) const
    {
        std::ofstream outfile(path);
        for (size_t row = 0; row < rows_; ++row)
        {
            for (size_t col = 0; col < columns_; ++col)
                outfile << data[row * columns_ + col] << " ";
            outfile << std::endl;
        }
        outfile.close();
    }

    void Matrix::transpose()
    {
        std::vector<double> ndata(data.size());
        for (size_t i = 0; i < rows_; i++)
            for (size_t j = 0; j < columns_; j++)
                ndata.at(j * rows_ + i) = data.at(i * columns_ + j);
        data = std::move(ndata);
        std::swap(rows_, columns_);
    }

    void Matrix::add_row(std::vector<double> &row, size_t times)
    {
        add_row(Section(row.begin(), row.end()), times);
    }
    void Matrix::add_row() {
        data.resize(data.size() + columns_);
        std::fill(data.end() - columns_, data.end(), 0);
        rows_++;
    }
    void Matrix::add_row(const Section& row, size_t times){
        if (columns_ == 0)
            columns_ = row.size();
        if (columns_ != row.size())
            throw std::invalid_argument("row size does not match.");
        std::vector<double> v{row.begin, row.end};
        data.resize(data.size() + times * row.size());
        for (size_t i = 0; i < times; i++)
        {
            std::copy(v.begin(), v.end(), data.begin() + columns_ * rows_);
            rows_++;
        }
    }

    std::vector<double> Matrix::operator()(size_t i)
    {
        if (i >= rows_)
            throw std::invalid_argument("Index is too big.");
        auto first = data.cbegin() + columns_ * i;
        auto last = data.cbegin() + columns_ * (i + 1);

        return std::vector<double>(first, last);
    }
}