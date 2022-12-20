#include <boost/test/unit_test.hpp>

#include "../LinAlg.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <memory>
#include <cmath>
#include "../Timer.hpp"
#include "../Eigen/Core"
#include "../Recognizer.hpp"

//bool same(const double &a, const double &b, const double eps = 0.00001) { return std::abs(a - b) < eps; }

BOOST_AUTO_TEST_SUITE(test_linalg)

using namespace linalg;

BOOST_AUTO_TEST_CASE(test_editDistance)
{
  std::vector<WordIdx> ref{0};
  std::vector<WordIdx> rec{0};
  auto dist = Recognizer::editDistance(ref.cbegin(), ref.cend(), rec.cbegin(), rec.cend());
  BOOST_TEST(dist == EDAccumulator(0, 0, 0, 0));

  ref = std::vector<WordIdx>{0};
  rec = std::vector<WordIdx>{1};
  dist = Recognizer::editDistance(ref.cbegin(), ref.cend(), rec.cbegin(), rec.cend());
  BOOST_TEST(dist == EDAccumulator(1, 1, 0, 0));

  ref = std::vector<WordIdx>{8, 5, 3};
  rec = std::vector<WordIdx>{5};
  dist = Recognizer::editDistance(ref.cbegin(), ref.cend(), rec.cbegin(), rec.cend());
  BOOST_TEST(dist == EDAccumulator(2, 0, 2, 0));

  ref = std::vector<WordIdx>{8, 5, 3};
  rec = std::vector<WordIdx>{9, 5, 4, 4};
  dist = Recognizer::editDistance(ref.cbegin(), ref.cend(), rec.cbegin(), rec.cend());
  BOOST_TEST(dist == EDAccumulator(3, 2, 0, 1));

  ref = std::vector<WordIdx>{1};
  rec = std::vector<WordIdx>{2, 3, 1, 9};
  dist = Recognizer::editDistance(ref.cbegin(), ref.cend(), rec.cbegin(), rec.cend());
  BOOST_TEST(dist == EDAccumulator(3, 0, 0, 3));
}

BOOST_AUTO_TEST_SUITE_END()