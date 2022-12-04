#include "Section.hpp"


namespace pgm {
    Vector Section::operator-() const {
        Vector ret({size()});
        std::transform(begin, end, ret.begin, std::negate<double>{});
        return ret;
    }


    Vector operator/(const Section &a, const double &b) {
        if (b == 0)
            throw std::invalid_argument("Can not divide by zero.");
        return operator_b<std::divides<double>>(a, b);
    }

    Vector operator+(const Section &a, const double &b) {
        return operator_b<std::plus<double>>(a, b);
    }

    Vector operator-(const Section &a, const double &b) {
        return operator_b<std::minus<double>>(a, b);
    }

    Vector operator*(const Section &a, const double &b) {
        return operator_b<std::multiplies<double>>(a, b);
    }

    Vector operator*(const double &a, const Section &b) {
        return b * a;
    }

    Vector operator/(const double &a, const Section &b) {
        std::vector<double> ret(b.end - b.begin);
        std::transform(b.begin, b.end, ret.begin(), [&a](const auto &v) { return std::divides<double>{}(a, v); });
        return Vector(std::move(ret));
    }
}