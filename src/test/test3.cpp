#include <boost/test/unit_test.hpp>

#include "../LinAlg.h"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <memory>
#include <cmath>
#include "../Timer.hpp"
#include "../Eigen/Core"

//bool same(const double &a, const double &b, const double eps = 0.00001) { return std::abs(a - b) < eps; }

BOOST_AUTO_TEST_SUITE(test_linalg)

using namespace linalg;

BOOST_AUTO_TEST_CASE(test_case5)
{
    BaseT d = {5, 1, 2, 3, 4, 5, 6};
    auto tmp = std::make_shared<BaseT>(d);
    Matrix m(tmp, {2, 3}, 1);
    BOOST_TEST(m.at(0, 0) == 1);
    BOOST_TEST(m.at(1, 1) == 5);
    BOOST_TEST(m.at(1, 2) == 6);

    auto tmp1 = std::make_shared<BaseT>(d);
    Matrix m1(tmp1, {3, 2}, 1);
    Matrix dm = m1.dot(m);
    BOOST_TEST(dm.slice.size()[0] == 3);
    BOOST_TEST(dm.slice.size()[1] == 3);
    BOOST_TEST(dm.at(0, 0) == 9);
    BOOST_TEST(dm.at(1, 1) == 26);
    BOOST_TEST(dm.at(1, 2) == 33);
    std::cout << dm[0] << std::endl;
    std::cout << dm[1] << std::endl;
    std::cout << dm[2] << std::endl;

    BaseT vd = {1, 2};
    Vector v(std::make_shared<BaseT>(vd), {2});
    BOOST_TEST(v.at(0) == 1);
    BOOST_TEST(v.at(1) == 2);
    BOOST_TEST(v.dot(v) == 5);
//    Vector dotv = v.dot(m);
//
//    BOOST_TEST(dotv.shape().size() == 1);
//    BOOST_TEST(dotv.shape()[0] == 3);
//    BOOST_TEST(dotv.at(0) == 9);
//    BOOST_TEST(dotv.at(1) == 12);
//    BOOST_TEST(dotv.at(2) == 15);
}

BOOST_AUTO_TEST_CASE(test_transpose) {
    BaseT d = {5, 1, 2, 3, 4, 5, 6};
    auto tmp = std::make_shared<BaseT>(d);
    Matrix m(tmp, {2, 3}, 1);
    BOOST_TEST(m.at(0, 0) == 1);
    BOOST_TEST(m.at(1, 1) == 5);
    BOOST_TEST(m.at(1, 2) == 6);
    Matrix tm = m.transpose();
    BOOST_TEST(tm.at(0, 0) == 1);
    BOOST_TEST(tm.at(0, 1) == 4);
    BOOST_TEST(tm.at(1, 0) == 2);
    BOOST_TEST(tm.at(1, 1) == 5);
    BOOST_TEST(tm.at(2, 0) == 3);
    BOOST_TEST(tm.at(2, 1) == 6);
}

BOOST_AUTO_TEST_CASE(test_tensor)
{
    BaseT d = {5, 1, 2, 3, 4, 5, 6, 7, 8};
    auto tmp = std::make_shared<BaseT>(d);
    Tensor t(tmp, {2, 2, 2}, 1);
    BaseT data = t.get();
    for(size_t i = 0; i < data.size(); i++)
        BOOST_TEST(data[i] == BaseT(d[std::slice(1, 8, 1)])[i]);
    BOOST_TEST(t.slice.stride()[0] == 4);
    BOOST_TEST(t.slice.stride()[1] == 2);
    BOOST_TEST(t.slice.stride()[2] == 1);

    Matrix m = t({0}, {0, 2}, {0, 2}).mat();
    BOOST_TEST(m.at(0, 0) == 1);
    BOOST_TEST(m.at(0, 1) == 2);
    BOOST_TEST(m.at(1, 0) == 3);
    BOOST_TEST(m.at(1, 1) == 4);

    Matrix m1 = t(ALL, {0}, ALL).mat();
    BOOST_TEST(m1.slice.size()[0] == 2);
    BOOST_TEST(m1.slice.size()[1] == 2);

    BOOST_TEST(m1.slice.stride()[0] == 4);
    BOOST_TEST(m1.slice.stride()[1] == 1);

    BOOST_TEST(m1.at(0, 0) == 1);
    BOOST_TEST(m1.at(0, 1) == 2);
    BOOST_TEST(m1.at(1, 0) == 5);
    BOOST_TEST(m1.at(1, 1) == 6);
}
BOOST_AUTO_TEST_CASE(test_addvec)
{
    BaseT d = {5, 1, 2, 3, 4, 5, 6, 7, 8};
    auto tmp = std::make_shared<BaseT>(d);
    Tensor t(tmp, {2, 2, 2}, 1);
    BaseT data = t.get();
    Matrix m = t(ALL, {0}, ALL).mat();
    Vector v = m.col(1);
    std::cout << v << std::endl;
    BOOST_TEST(v.at(0) == 2);
    BOOST_TEST(v.at(1) == 6);
    m = m + v;
    BOOST_TEST(m.at(0, 0) == 3);
    BOOST_TEST(m.at(0, 1) == 4);
    BOOST_TEST(m.at(1, 0) == 11);
    BOOST_TEST(m.at(1, 1) == 12);
}

BOOST_AUTO_TEST_CASE(test_perf) {
    Matrix m({1000, 2000});
    (*m.data) = 2;
    Timer timer;
    timer.tick();
    auto val = m.dot(m.transpose());
    timer.tock();
    std::cout << "m " << timer.secs() << " " << val.at(0, 1) << std::endl;
    Eigen::MatrixXf em(1000, 2000);
    em.fill(2);
    timer.reset();
    timer.tick();
    auto eigval = em * em.transpose();
    timer.tock();
    std::cout << "em " << timer.secs() << " " << eigval(0, 1) << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()