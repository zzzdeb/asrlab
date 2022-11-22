#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN  // in only one cpp file
#include <boost/test/unit_test.hpp>

#include "../Mixtures.hpp"
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>
#include <memory>
#include <cmath>

// const Configuration config
double calc_am_score(std::pair<FeatureIter, FeatureIter> features, Alignment const &alignment, const MixtureModel& model)
{
  double score = 0.;
  size_t num_max_aligns_ = 1;
  for (auto i = 0; i < features.second - features.first; i++)
    for (size_t j = 0; j < num_max_aligns_; j++)
      score += model.score(features.first + i, alignment.at(i * num_max_aligns_ + j).state);
  score /= features.second - features.first;
  return score;
}

bool same(const double &a, const double &b, const double eps = 0.00001) { return std::abs(a - b) < eps; }

BOOST_AUTO_TEST_SUITE(test_mixture)

BOOST_AUTO_TEST_CASE(test_case1)
{
	const char json[] = "{}";
	rapidjson::StringStream s(json);
	const Configuration config(s);
	MixtureModel mmodel(config, 1, 1, MixtureModel::NO_POOLING, true);
	const size_t dim = 1;
	const size_t num_features = 4;
	float features[dim * num_features] = {-1, 0, 4, 5};
	FeatureIter feature_begin(features, dim);
	FeatureIter feature_end(features + dim * num_features, dim);

  	const Alignment aligns{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	
  	ConstAlignmentIter alignment_begin(&*aligns.begin(), 1);
    ConstAlignmentIter alignment_end(&*aligns.begin() + num_features, 1);

	BOOST_TEST(same(1.4189385, mmodel.density_scores(feature_begin, 0)[0]));
	// std::cout << mmodel.density_scores(feature_begin, 0) << std::endl;
	BOOST_TEST(same(0.9189385, mmodel.density_scores(feature_begin+1, 0)[0]));
	BOOST_TEST(same(8.91893853, mmodel.density_scores(feature_begin+2, 0)[0]));
	BOOST_TEST(same(13.4189385, mmodel.density_scores(feature_begin+3, 0)[0]));

	std::cout << calc_am_score({feature_begin, feature_end}, aligns, mmodel) << std::endl;

  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, true, true);
  	mmodel.finalize();

	std::cout << calc_am_score({feature_begin, feature_end}, aligns, mmodel) << std::endl;

	BOOST_TEST(2 == mmodel.get_means()(0, 0));
	double var = 1/6.5;
	BOOST_TEST(var == mmodel.get_vars()(0, 0));
	// std::cout << mmodel.get_means()(0) << std::endl;
	// std::cout << mmodel.density_scores(feature_begin, 0) << std::endl;
	// BOOST_TEST(same(0.9189385, mmodel.density_scores(feature_begin, 0)[0]));

	mmodel.split(3);
	BOOST_TEST(2+std::sqrt(6.5) == mmodel.get_means()(0, 0));
	BOOST_TEST(2-std::sqrt(6.5) == mmodel.get_means()(1, 0));
	BOOST_TEST(var == mmodel.get_vars()(0, 0));
	BOOST_TEST(var == mmodel.get_vars()(1, 0));

	auto scores = mmodel.density_scores(feature_begin, 0);
	BOOST_TEST(same(4.91699, scores[0]));
	BOOST_TEST(same(2.5636, scores[1]));
	scores = mmodel.density_scores(feature_begin+2, 0);
	BOOST_TEST(same(2.57121, scores[0]));
	BOOST_TEST(same(4.14014, scores[1]));
	BOOST_TEST(same((2.5636+ 2.57121)/2, calc_am_score({feature_begin, feature_end}, aligns, mmodel)));

  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, false, true);
	mmodel.finalize();
	BOOST_TEST(4.5 == mmodel.get_means()(0, 0));
	BOOST_TEST(-0.5 == mmodel.get_means()(1, 0));
	BOOST_TEST(4 == mmodel.get_vars()(0, 0));
	BOOST_TEST(4 == mmodel.get_vars()(1, 0));
}
BOOST_AUTO_TEST_CASE(test_case2)
{
	const char json[] = "{}";
	rapidjson::StringStream s(json);
	const Configuration config(s);
	const size_t dim = 2;
	MixtureModel mmodel(config, dim, 1, MixtureModel::NO_POOLING, true);
	const size_t num_features = 4;
	float features[dim * num_features] = {-2, 1, 2, 1, 2, -1, -2, -1};
	FeatureIter feature_begin(features, dim);
	FeatureIter feature_end(features + dim * num_features, dim);

  	const Alignment aligns{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	
  	ConstAlignmentIter alignment_begin(&*aligns.begin(), 1);
    ConstAlignmentIter alignment_end(&*aligns.begin() + num_features, 1);

	BOOST_TEST(same(4.337877, mmodel.density_scores(feature_begin, 0)[0]));
	BOOST_TEST(same(4.337877, mmodel.density_scores(feature_begin+1, 0)[0]));
	BOOST_TEST(same(4.337877, mmodel.density_scores(feature_begin+2, 0)[0]));
	BOOST_TEST(same(4.337877, mmodel.density_scores(feature_begin+3, 0)[0]));

	std::cout << "0" << calc_am_score({feature_begin, feature_end}, aligns, mmodel) << std::endl;
  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, true, true);
  	mmodel.finalize();
	std::cout << "1" << calc_am_score({feature_begin, feature_end}, aligns, mmodel) << std::endl;
	using V = std::vector<double>;
	BOOST_TEST((V{0, 0} == mmodel.get_means()[0]));
	BOOST_TEST((V{0.25, 1} == mmodel.get_vars()[0]));

	mmodel.split(3);
	BOOST_TEST((V{2, 1} == mmodel.get_means()[0]));
	BOOST_TEST((V{-2, -1} == mmodel.get_means()[1]));
	BOOST_TEST((V{0.25, 1} == mmodel.get_vars()[0]));
	BOOST_TEST((V{0.25, 1} == mmodel.get_vars()[1]));

	auto scores = mmodel.density_scores(feature_begin, 0);
	// BOOST_TEST(scores.approx(V{4.551935, 4.551935}));
	scores = mmodel.density_scores(feature_begin+2, 0);
	// BOOST_TEST(scores.approx(V{4.551935, 4.551935}));

  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, false, true);
	mmodel.finalize();
	BOOST_TEST(same(0.666667, mmodel.get_means()(0,0)));
	BOOST_TEST(same(0.333333, mmodel.get_means()(0,1)));
	BOOST_TEST((V{-2, -1} == mmodel.get_means()[1]));
  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, false, true);
	mmodel.finalize();
	BOOST_TEST(same(0.666667, mmodel.get_means()(0,0)));
	BOOST_TEST(same(0.333333, mmodel.get_means()(0,1)));
	BOOST_TEST((V{-2, -1} == mmodel.get_means()[1]));

	BOOST_TEST((V{0.28125,1.125} == mmodel.get_vars()[0]));
	BOOST_TEST((V{10000, 10000} == mmodel.get_vars()[1]));
}

BOOST_AUTO_TEST_CASE(test_case3)
{
	const char json[] = "{}";
	rapidjson::StringStream s(json);
	const Configuration config(s);
	const size_t dim = 2;
	MixtureModel mmodel(config, dim, 1, MixtureModel::NO_POOLING, true);
	mmodel.mean_refs_ = std::vector<size_t>{0, 1, 0};
	auto& mixture = mmodel.mixtures_.front();
	mixture.emplace_back(0, 0);
	mixture.emplace_back(1, 1);
	mixture.emplace_back(2, 2);
	mixture.at(0).mean_idx = 0;
	mixture.at(1).mean_idx = 1;
	mixture.at(2).mean_idx = 2;
	BOOST_TEST(1 == mmodel.get_ith_active(0, 0));

	mmodel.mean_refs_ = std::vector<size_t>{1, 1, 0};
	BOOST_TEST(0 == mmodel.get_ith_active(0, 0));
	BOOST_TEST(1 == mmodel.get_ith_active(0, 1));

	mmodel.mean_refs_ = std::vector<size_t>{1, 0, 1};
	BOOST_TEST(0 == mmodel.get_ith_active(0, 0));
	BOOST_TEST(2 == mmodel.get_ith_active(0, 1));
}

BOOST_AUTO_TEST_CASE(test_sum_scores)
{
	const char json[] = "{}";
	rapidjson::StringStream s(json);
	const Configuration config(s);
	MixtureModel mmodel(config, 1, 1, MixtureModel::NO_POOLING, false);
	const size_t dim = 1;
	const size_t num_features = 4;
	float features[dim * num_features] = {-1, 0, 4, 5};
	FeatureIter feature_begin(features, dim);
	FeatureIter feature_end(features + dim * num_features, dim);

  	const Alignment aligns{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
	
  	ConstAlignmentIter alignment_begin(&*aligns.begin(), 1);
    ConstAlignmentIter alignment_end(&*aligns.begin() + num_features, 1);

	BOOST_TEST(same(1.4189385, mmodel.density_scores(feature_begin, 0)[0]));
	// std::cout << mmodel.density_scores(feature_begin, 0) << std::endl;
	BOOST_TEST(same(0.9189385, mmodel.density_scores(feature_begin+1, 0)[0]));
	BOOST_TEST(same(8.91893853, mmodel.density_scores(feature_begin+2, 0)[0]));
	BOOST_TEST(same(13.4189385, mmodel.density_scores(feature_begin+3, 0)[0]));

  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, true, true);
  	mmodel.finalize();

	BOOST_TEST(2 == mmodel.get_means()(0, 0));
	double var = 1/6.5;
	BOOST_TEST(var == mmodel.get_vars()(0, 0));
	// std::cout << mmodel.get_means()(0) << std::endl;
	// std::cout << mmodel.density_scores(feature_begin, 0) << std::endl;
	// BOOST_TEST(same(0.9189385, mmodel.density_scores(feature_begin, 0)[0]));

	mmodel.split(3);
	BOOST_TEST(2+std::sqrt(6.5) == mmodel.get_means()(0, 0));
	BOOST_TEST(2-std::sqrt(6.5) == mmodel.get_means()(1, 0));
	BOOST_TEST(var == mmodel.get_vars()(0, 0));
	BOOST_TEST(var == mmodel.get_vars()(1, 0));

	auto scores = mmodel.density_scores_normalized(feature_begin, 0);
	BOOST_TEST(same(2.44419, scores[0]));
	BOOST_TEST(same(0.0907964, scores[1]));
	scores = mmodel.density_scores_normalized(feature_begin+2, 0);
	// std::cout << scores << std::endl;
	BOOST_TEST(same(0.189188, scores[0]));
	BOOST_TEST(same(1.75812, scores[1]));
	std::cout << calc_am_score({feature_begin, feature_end}, aligns, mmodel) << std::endl;
	BOOST_TEST(same(2.42741, calc_am_score({feature_begin, feature_end}, aligns, mmodel)));

  	mmodel.accumulate(alignment_begin, alignment_end, feature_begin,  feature_end, false, true);
	mmodel.finalize();
}

BOOST_AUTO_TEST_SUITE_END()
