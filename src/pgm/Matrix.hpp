#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <string>
#include <stdexcept>
#include <vector>

namespace pgm
{

    class Matrix
    {
    public:
        Matrix() = default;

        void from_file(const std::string &path);
        void to_file(const std::string &path) const;

        void add_row(const std::vector<double> &row, size_t times = 1);

        void transpose();

        size_t get_height() const { return height; }

        // Returns row i
        std::vector<double> operator()(size_t i)
        {
            if (i >= height)
                throw std::invalid_argument("Index is too big.");
            auto first = data.cbegin() + width * i;
            auto last = data.cbegin() + width * (i + 1);

            return std::vector<double>(first, last);
        };

    protected:
        size_t width{0};
        size_t height{0};
        std::vector<double> data{};
    };

}

#endif /* __MATRIX_HPP__ */