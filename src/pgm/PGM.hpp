#ifndef __PGM_HPP__
#define __PGM_HPP__

#include <string>
#include <stdexcept>
#include <vector>

namespace pgm
{

  class PGM
  {
  public:
    PGM() = default;

    enum PgmType
    {
      P2,
      P5,
    };

    void from_file(const std::string &path);
    void to_file(const std::string &path, PgmType type = P2, bool scale = true, bool invert = false) const;

    void add_row(const std::vector<double> &row, size_t times = 1);

    void transpose();

  private:
    size_t width{0};
    size_t height{0};
    std::string comment{""};
    std::vector<double> data{};
  };

}

#endif /* __PGM_HPP__ */
