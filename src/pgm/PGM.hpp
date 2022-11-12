#ifndef __PGM_HPP__
#define __PGM_HPP__

#include "Matrix.hpp"

#include <string>
#include <stdexcept>
#include <vector>

namespace pgm
{

  class PGM : public Matrix
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

  private:
    std::string comment{""};
  };

}

#endif /* __PGM_HPP__ */
