#include "Matrix.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

namespace pgm
{
    // Based on the implementation detail of to_file, counts '\n' and ' '
    void get_shape(const std::stringstream &ss, size_t &height, size_t &width)
    {
        std::string content = ss.str();
        std::string line("");
        height = std::count(content.cbegin(), content.cend(), '\n');
        width = std::count(content.cbegin(), content.cend(), ' ') / height;
    }

    void Matrix::from_file(const std::string &path)
    {
        std::ifstream infile(path);
        std::stringstream ss;
        ss << infile.rdbuf();
        // get shape
        get_shape(ss, height, width);
        // Following lines : data
        for (size_t row = 0; row < height; ++row)
            for (size_t col = 0; col < width; ++col)
                ss >> data.at(row * width + col);
        infile.close();
    }

    void Matrix::to_file(const std::string &path) const
    {
        std::ofstream outfile(path);
        for (size_t row = 0; row < height; ++row)
        {
            for (size_t col = 0; col < width; ++col)
                outfile << data[row * width + col] << " ";
            outfile << std::endl;
        }
        outfile.close();
    }

    void Matrix::transpose()
    {
        std::vector<double> ndata(data.size());
        for (size_t i = 0; i < height; i++)
            for (size_t j = 0; j < width; j++)
                ndata.at(j * height + i) = data.at(i * width + j);
        data = std::move(ndata);
        std::swap(height, width);
    }

    void Matrix::add_row(const std::vector<double> &row, size_t times)
    {
        if (width == 0)
            width = row.size();
        if (width != row.size())
            throw std::invalid_argument("row size does not match.");

        data.resize(data.size() + times * row.size());
        for (size_t i = 0; i < times; i++)
        {
            std::copy(row.cbegin(), row.cend(), data.begin() + width * height);
            height++;
        }
    }
}