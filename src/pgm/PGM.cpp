#include "PGM.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>

namespace pgm
{

  bool almost_equal(double x, double y)
  {
    return std::fabs(x - y) <= 5 * std::numeric_limits<double>::epsilon();
  }

  PGM::PgmType str_to_type(const std::string &typeStr)
  {
    if (typeStr.compare("P2") == 0)
      return PGM::P2;
    else if (typeStr.compare("P5") == 0)
      return PGM::P5;
    else
      throw std::invalid_argument("Pgm typestr is wrong.");
  }

  std::string type_to_str(const PGM::PgmType &t)
  {
    if (t == PGM::P2)
      return "P2";
    else if (t == PGM::P5)
      return "P5";
    else
      throw std::invalid_argument("Pgm type is wrong.");
  }

  void PGM::from_file(const std::string &path)
  {
    std::ifstream infile(path);
    std::stringstream ss;
    std::string inputLine = "";
    // PgmType type;
    // First line : version
    getline(infile, inputLine);
    // type = str_to_type(inputLine);

    // Second line : comment
    getline(infile, comment);

    // Continue with a stringstream
    ss << infile.rdbuf();
    // Third line : size
    ss >> width >> height;
    data.resize(width * height);

    // Following lines : data
    for (size_t row = 0; row < height; ++row)
      for (size_t col = 0; col < width; ++col)
        ss >> data[row * width + col];
    infile.close();
  }

  void PGM::to_file(const std::string &path, PgmType type, bool scale, bool invert) const
  {
    double realScale = 1.;
    double offSet = 0.;
    if (scale)
    {
      auto [minData, maxData] = std::minmax_element(data.cbegin(), data.cend());
      offSet = *minData;
      // special case minData==maxData
      if (!almost_equal(*minData, *maxData))
      {
        realScale = 255 / (*maxData - *minData);
      }
    }
    std::ofstream outfile(path);
    outfile << type_to_str(type) << std::endl;
    outfile << "#" << std::endl;
    outfile << width << " " << height << std::endl;
    outfile << 255 << std::endl;
    for (size_t row = 0; row < height; ++row)
    {
      for (size_t col = 0; col < width; ++col)
      {
        auto val = static_cast<int>((data[row * width + col] - offSet) * realScale);
        if (invert)
          val = 255 - val;
        outfile << val << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }
}
