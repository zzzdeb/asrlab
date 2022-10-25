#ifndef __PGM_HPP__
#define __PGM_HPP__

#include <string>
#include <stdexcept>
#include <vector>

namespace pgm {

class PGM {
public:

  static const std::string WS;

  PGM() = default;

  enum PgmType {
    P2,
    P5,
  };

  void from_file(const std::string& path);
  void to_file(const std::string& path, PgmType type=P2, bool scale = true) const;

  void add_row(const std::vector<double>& row) {
    if (width == 0)
      width = row.size();
    if (width != row.size())
      throw std::invalid_argument("row size does not match.");

    data.resize(data.size() + row.size());
    std::copy(row.cbegin(), row.cend(), data.begin() + width*height);
    height++;
  }

private:
  size_t width{0};
  size_t height{0};
  std::string comment{""};
  std::vector<double> data{};
};

}


#endif /* __PGM_HPP__ */
