/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Mixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "emmintrin.h"
#include "pmmintrin.h"

namespace {
  // compute -log(exp(-a) + exp(-b)) using the following equality:
  // -log(exp(-a) + exp(-b)) = -log(exp(-a-c) + exp(-b-c)) - c
  //                         = -log(1 + exp(-abs(a-b))) - c
  // where c = max(-a, -b) = -min(a, b)
  double logsum(double a, double b) {
    const double diff = a - b;
    // if a = b = inf then a - b = nan, but for logsum the result is well-defined (inf)
    if (diff != diff) {
      return std::numeric_limits<double>::infinity();
    }
    return -log1p(std::exp(-std::abs(diff))) + std::min(a, b);
  }

  size_t build_mapping(std::vector<size_t> const& refs, std::vector<size_t>& mapping) {
    mapping.resize(refs.size());

    size_t count = 0ul;
    for (size_t i = 0ul; i < refs.size(); i++) {
      if (refs[i] > 0ul) {
        mapping[i] = count;
        count++;
      }
    }

    return count;
  }

  void test(bool condition, char const* error_msg) {
    if (not condition) {
      std::cerr << error_msg << std::endl;
      abort();
    }
  }

  void read_accumulator(std::istream& in, size_t expected_dimension,
                        std::vector<size_t>& refs,
                        Matrix& features,
                        std::vector<double>& weight) {
    uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    test(in.gcount() == sizeof(size), "Error reading size");

    refs.resize(size);
    features.resize({size, expected_dimension});
    weight.resize(size);

    for (size_t i = 0ul; i < size; i++) {
      uint32_t dimension;

      in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
      test(in.gcount() == sizeof(dimension), "Error reading dimension");
      test(dimension == expected_dimension, "Invalid dimension");

      in.read(reinterpret_cast<char*>(&features(i, 0)), sizeof(double) * dimension);
      test(static_cast<size_t>(in.gcount()) == sizeof(double) * dimension, "Error reading features");

      in.read(reinterpret_cast<char*>(&weight[i]), sizeof(double));
      test(in.gcount() == sizeof(double), "Error reading weight");
    }
  }

  void write_accumulator(std::ostream& out, uint32_t size, uint32_t dimension,
                         std::vector<size_t> const& refs,
                         Matrix const& features,
                         std::vector<double> const& weight) {
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (size_t i = 0ul; i < refs.size(); i++) {
      if (refs[i] == 0ul) {
        continue;
      }
      out.write(reinterpret_cast<const char*>(&dimension),               sizeof(dimension));
      out.write(reinterpret_cast<const char*>(&features(i, 0)), sizeof(double) * dimension);
      out.write(reinterpret_cast<const char*>(&weight[i]),               sizeof(double));
    }
  }

  Vector ToVector(FeatureIter& iter) {
    return {*iter, *iter + iter.size};
  }
}

/*****************************************************************************/

const ParameterString MixtureModel::paramLoadMixturesFrom("load-mixtures-from", "");
const ParameterBool MixtureModel::paramWriteMixtures("write-mixtures", false);

const char     MixtureModel::magic[8] = {'M', 'I', 'X', 'S', 'E', 'T', 0, 0};
const uint32_t MixtureModel::version = 2u;

/*****************************************************************************/

MixtureModel::MixtureModel(Configuration const& config, size_t dimension_p, size_t num_mixtures,
                           VarianceModel var_model, bool max_approx)
            : dimension(dimension_p),
              var_model(var_model),
              max_approx_(max_approx),

              means_(num_mixtures, dimension),
              mean_accumulators_(num_mixtures, dimension),
              mean_weights_(num_mixtures),
              mean_weight_accumulators_(num_mixtures),
              mean_refs_(num_mixtures, 1),
              mixture_accumulators_(num_mixtures),

              vars_(num_mixtures, dimension, 1),
              var_accumulators_(num_mixtures, dimension),
              var_weight_accumulators_(num_mixtures),
              var_refs_(num_mixtures, 1),

              norm_fixed_(dimension * std::log(2 * M_PI) / 2),
              norm_(num_mixtures),

              mixtures_(num_mixtures),
              alignment_begin_(nullptr, 1),
              alignment_end_(nullptr, 1),
              write_mixtures_(paramWriteMixtures(config))
{
  for (size_t i = 0; i < num_mixtures; i++)
  {
    norm_.at(i) = norm_fixed_ - vars_(i).log().sum() / 2;
    mixtures_.at(i).emplace_back(i, i);
  }
}

/*****************************************************************************/

void MixtureModel::reset_accumulators() {
  std::fill(mean_accumulators_.data.begin(), mean_accumulators_.data.end(), 0);
  std::fill(mean_weight_accumulators_.begin(), mean_weight_accumulators_.end(), 0);
  std::fill(var_accumulators_.data.begin(), var_accumulators_.data.end(), 0);
  std::fill(var_weight_accumulators_.begin(), var_weight_accumulators_.end(), 0);
  std::fill(mixture_accumulators_.begin(), mixture_accumulators_.end(), 0);
}

/*****************************************************************************/

void MixtureModel::accumulate(ConstAlignmentIter alignment_begin, ConstAlignmentIter alignment_end,
                              FeatureIter        feature_begin,   FeatureIter        feature_end,
                              bool first_pass, bool max_approx) {
  alignment_begin_ = alignment_begin;
  alignment_end_ = alignment_end;
  std::vector<std::pair<double, DensityIdx>> scores(feature_end - feature_begin);
  reset_accumulators();
  for (size_t i = 0; i < feature_end - feature_begin; i++)
  {
    auto feature = feature_begin + i;
    auto alignment = alignment_begin + i;
    const Mixture& mixture = mixtures_.at((*alignment)->state);
    mixture_accumulators_.at((*alignment)->state)++;
    MixtureDensity density = mixture.at(0);
    if (!first_pass) {
      density.mean_idx = min_score(feature, (*alignment)->state).second;
      density.var_idx = density.mean_idx;
    }

    // scores.at(it - feature_begin) = s;
    // std::cout << mean_accumulators_(density.mean_idx) << std::endl;
    auto fvector = ToVector(feature);
    // if (density.mean_idx == 446)
      // std::cout << fvector << std::endl;
    mean_accumulators_(density.mean_idx) += fvector;
    // std::cout << fvector << std::endl;
    // std::cout << mean_accumulators_(density.mean_idx) << std::endl;
    mean_weight_accumulators_.at(density.mean_idx)++;

    // Section features;
    var_accumulators_(density.var_idx) += fvector.square();
    var_weight_accumulators_.at(density.var_idx)++;;
  }
  // std::vector<size_t> tmp;
  // std::transform(mixtures_.begin(), mixtures_.end(), std::back_inserter(tmp), [](auto& v) {return v.size();});
  // cprint(tmp);
}

void MixtureModel::visualize(std::string header) {
  if (!write_mixtures_)
    return;
  stats_out << "==" << header << std::endl;
  size_t index = 44;
  auto& mixture = mixtures_.at(index);
  for (auto& m: mixture) {
    stats_out << "n ";
    stats_out << m.mean_idx << " ";
    stats_out << mean_refs_.at(m.mean_idx) << " ";
    stats_out << mean_weights_.at(m.mean_idx) << " ";
    stats_out << means_(m.mean_idx) << " ";
    stats_out << var_refs_.at(m.var_idx) << " ";
    stats_out << m.var_idx << " ";
    stats_out << vars_(m.var_idx);

    stats_out << std::endl;
  }
  for(auto iter = alignment_begin_; iter < alignment_end_; iter++) {
    if ((*iter)->state == index)
      stats_out << iter - alignment_begin_ << " ";
  }
  stats_out << std::endl;
}

/*****************************************************************************/

void MixtureModel::finalize() {
  // Implements means_, vars_ and norm_
  for (size_t i = 0; i < mean_weights_.size(); i++) {
    if (mean_weight_accumulators_.at(i) == 0)
      mean_refs_.at(i) = 0;
    if (var_weight_accumulators_.at(i) == 0)
      var_refs_.at(i) = 0;
    if (mean_refs_.at(i) == 0)
      continue;
    means_(i) = mean_accumulators_(i) / mean_weight_accumulators_.at(i);
    vars_(i) = 1 / (var_accumulators_(i) / var_weight_accumulators_.at(i) - means_(i).square()).square().nonzero();
    norm_.at(i) = norm_fixed_ - (vars_(i).log().sum() / 2);
  }
  for (size_t i = 0; i < mixtures_.size(); i++)
    for (const auto& dens: mixtures_.at(i))
      mean_weights_.at(dens.mean_idx) = -std::log(mean_weight_accumulators_.at(dens.mean_idx) / mixture_accumulators_.at(i));
  check_validity();
  visualize("f");
}

void MixtureModel::check_validity() {
  for (size_t i = 0; i < mean_weights_.size(); i++) {
    if (mean_refs_.at(i) == 0)
      continue;
    // for (vars_(i))
  }
  auto check = [](double v) { return v <10000000 && v > -10000000;};
  for (size_t i = 0; i < norm_.size(); i++){
    if(!check(norm_.at(i)))
      throw std::logic_error("Naa");
  }
  for (size_t i = 0; i < mixture_accumulators_.size(); i++)
    if(mixture_accumulators_.at(i) == 0)
      throw std::logic_error("Naa");
  auto check_vars = [](double v) { return v >0;};
  // for (size_t i = 0; i < vars_.size().first; i++)
  //   for (size_t j = 0; j < vars_.size().second; j++)
  //     if(!check_vars(vars_(i, j)))
  //       throw std::logic_error("Naa");
}

/*****************************************************************************/

void MixtureModel::split(size_t min_obs) {
  for (auto& mixture : mixtures_)
  {
    for (size_t i = 0; i < mixture.size(); i++)
    {
      DensityIdx cur_mean_idx= mixture.at(i).mean_idx;
      DensityIdx cur_var_idx= mixture.at(i).var_idx;
      if (mean_refs_.at(cur_mean_idx) == 0)
        continue;
      if (mean_weight_accumulators_.at(cur_mean_idx) >= min_obs) {
        auto cur_mean = means_(cur_mean_idx);
        auto diff = (1 / vars_(cur_var_idx)).sqrt();
        auto p_mean = cur_mean + diff;
        auto n_mean = cur_mean - diff;
        means_(cur_mean_idx) = p_mean;
        mean_weights_.at(cur_mean_idx) += log(2);
        {
          // add mean
          means_.add_row(n_mean);
          mean_accumulators_.add_row();
          mean_weights_.emplace_back(mean_weights_.at(cur_mean_idx));
          mean_weight_accumulators_.emplace_back();
          mean_refs_.emplace_back(1);
        }
        {
          // add var
          vars_.add_row(vars_(cur_var_idx));
          var_accumulators_.add_row();
          var_weight_accumulators_.emplace_back();
          var_refs_.emplace_back(1);
          norm_.emplace_back();
        }
        mixture.emplace_back(mean_weights_.size() - 1, var_weight_accumulators_.size() - 1);
      }
    }
  }
  visualize("s");
}

/*****************************************************************************/

void MixtureModel::eliminate(double min_obs) {
  for (auto& mixture : mixtures_)
  {
    for (size_t i = 0; i < mixture.size(); i++)
    {
      if (mean_weight_accumulators_.at(mixture.at(i).mean_idx) < min_obs)
        mean_refs_.at(i) = 0;
      if (var_weight_accumulators_.at(mixture.at(i).var_idx) < min_obs)
        var_refs_.at(i) = 0;
    }
  }
  visualize("e");
}

/*****************************************************************************/

size_t MixtureModel::num_densities() const {
  return mean_refs_.size() - std::count(mean_refs_.begin(), mean_refs_.end(), 0ul);
}

/*****************************************************************************/

// computes the score (probability in negative log space) for a feature vector
// given a mixture density
double MixtureModel::density_score(FeatureIter const& iter, StateIdx mixture_idx, DensityIdx density_idx) const {
  const Mixture& mixture = mixtures_.at(mixture_idx);
  const MixtureDensity& density = mixture.at(density_idx);
  if (mean_refs_.at(density.mean_idx) == 0)
    return std::numeric_limits<double>::max();
  const double& minus_log_c = mean_weights_.at(density.mean_idx);

  double p = minus_log_c + norm_.at(density.var_idx);
  size_t midx = density.mean_idx;
  size_t vidx = density.var_idx;
  const double add = ((Vector(*iter, *iter+dimension) - means_(midx)).square() * vars_(vidx)).sum()/2;
  p += add;
  if (p == std::numeric_limits<double>::infinity())
    throw std::logic_error("Invalid prob");
  return p;
}

/*****************************************************************************/

// this function returns the density with the lowest score (=highest probability)
// for the given feature vector
std::pair<double, DensityIdx> MixtureModel::min_score(FeatureIter const& iter, StateIdx mixture_idx) const {
  const Mixture& mixture = mixtures_.at(mixture_idx);
  double min_score = std::numeric_limits<double>::max();
  DensityIdx min_index = -1;
  for (size_t i = 0; i < mixture.size(); i++)
  {
    double score = density_score(iter, mixture_idx, i);
    if (score < min_score) {
      min_score = score;
      min_index = mixture.at(i).mean_idx;
    }
  }
  if (min_index == 65535)
    throw std::logic_error("It should be smaller");
  
  return std::make_pair(min_score, min_index);
}

/*****************************************************************************/

// compute the 'full' score of a feature vector for a given mixture. The weights
// of each density are stored in weights and should sum up to 1.0
double MixtureModel::sum_score(FeatureIter const& iter, StateIdx mixture_idx, std::vector<double>* weights) const {
  // TODO: implement
  return 0.0;
}

/*****************************************************************************/

void MixtureModel::prepare_sequence(FeatureIter const& start, FeatureIter const& end) {
}

/*****************************************************************************/

double MixtureModel::score(FeatureIter const& iter, StateIdx mixture_idx) const {
  if (max_approx_) {
    return min_score(iter, mixture_idx).first;
  }
  else {
    return sum_score(iter, mixture_idx, NULL);
  }
}

/*****************************************************************************/

void MixtureModel::read(std::istream& in) {
  char     magic_test[8];
  uint32_t version_test;
  uint32_t dimension_test;

  in.read(magic_test, sizeof(magic_test));
  test(in.gcount() == sizeof(magic_test),             "Error reading magic header");
  test(std::string(magic) == std::string(magic_test), "Invalid magic header");

  in.read(reinterpret_cast<char*>(&version_test), sizeof(version_test));
  test(in.gcount() == sizeof(version_test),           "Error reading version");
  test(version == version_test,                       "Invalid version");

  in.read(reinterpret_cast<char*>(&dimension_test), sizeof(dimension_test));
  test(in.gcount() == sizeof(dimension_test),         "Error reading version");
  test(dimension == dimension_test,                   "Invalid version");

  read_accumulator(in, dimension, mean_refs_, mean_accumulators_, mean_weight_accumulators_);
  test(mean_refs_.size() < (1ul << 16),               "Too many means, mean indices are 16bit ints");
  mean_weights_.resize(mean_weight_accumulators_.size());
  means_.resize(mean_accumulators_.size());

  read_accumulator(in, dimension, var_refs_,  var_accumulators_,  var_weight_accumulators_);
  test(var_refs_.size() < (1ul << 16),                "Too many variances, var indices are 16bit ints");
  vars_.resize(var_accumulators_.size());
  norm_.resize(var_weight_accumulators_.size());

  uint32_t density_count;
  in.read(reinterpret_cast<char*>(&density_count), sizeof(density_count));
  test(in.gcount() == sizeof(density_count),          "Error reading density count");
  std::cerr << "Num densities: " << density_count << std::endl;

  std::vector<MixtureDensity> densities;
  densities.reserve(density_count);

  for (size_t i = 0ul; i < density_count; i++) {
    uint32_t mean_idx;
    in.read(reinterpret_cast<char*>(&mean_idx), sizeof(mean_idx));
    test(in.gcount() == sizeof(mean_idx),             "Error reading mean_idx");
    test(mean_idx < mean_refs_.size(),                "Invalid mean_idx");

    uint32_t var_idx;
    in.read(reinterpret_cast<char*>(&var_idx), sizeof(var_idx));
    test(in.gcount() == sizeof(var_idx),              "Error reading var_idx");
    test(var_idx < var_refs_.size(),                  "Invalid var_idx");

    mean_refs_[mean_idx] += 1ul;
    var_refs_[var_idx]   += 1ul;

    densities.push_back(MixtureDensity(mean_idx, var_idx));
  }

  uint32_t mixture_count;
  in.read(reinterpret_cast<char*>(&mixture_count), sizeof(mixture_count));
  test(in.gcount() == sizeof(mixture_count),          "Error reading mixture count");
  mixtures_.resize(mixture_count);
  std::cerr << "Num mixtures: " << mixture_count << std::endl;

  for (size_t m = 0ul; m < mixture_count; m++) {
    in.read(reinterpret_cast<char*>(&density_count), sizeof(density_count));
    test(in.gcount() == sizeof(density_count),        "Error reading density count for mixture");

    mixtures_[m].clear();
    mixtures_[m].reserve(density_count);
    for (size_t d = 0ul; d < density_count; d++) {
      uint32_t density_idx;
      in.read(reinterpret_cast<char*>(&density_idx), sizeof(density_idx));
      test(in.gcount() == sizeof(density_idx),        "Error reading density idx");
      test(density_idx < densities.size(),            "Invalid density idx");

      mixtures_[m].push_back(densities[density_idx]);

      double density_weight;
      const double expected_density_weight = mean_weight_accumulators_[densities[density_idx].mean_idx];
      in.read(reinterpret_cast<char*>(&density_weight), sizeof(density_weight));
      test(in.gcount() == sizeof(density_weight),     "Error reading density weight");
      test(density_weight == expected_density_weight, "Inconsistent density weight");
    }
  }

  finalize();
}

/*****************************************************************************/

void MixtureModel::write(std::ostream& out) const {
  const uint32_t dimension_u32 = dimension;
  out.write(magic, sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version),       sizeof(version));
  out.write(reinterpret_cast<const char*>(&dimension_u32), sizeof(dimension_u32));

  std::vector<size_t> mean_mapping(mean_refs_.size());
  std::vector<size_t> var_mapping (var_refs_.size());

  const uint32_t mean_count = build_mapping(mean_refs_, mean_mapping);
  const uint32_t var_count  = build_mapping(var_refs_,  var_mapping);

  write_accumulator(out, mean_count, dimension,
                    mean_refs_, mean_accumulators_, mean_weight_accumulators_);
  write_accumulator(out, var_count, dimension,
                    var_refs_,  var_accumulators_,  var_weight_accumulators_);

  uint32_t density_count = 0u;
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    density_count += mixtures_[m].size();
  }
  out.write(reinterpret_cast<const char*>(&density_count), sizeof(density_count));
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    for (size_t d = 0ul; d < mixtures_[m].size(); d++) {
      uint32_t mean_idx = mean_mapping[mixtures_[m][d].mean_idx];
      uint32_t var_idx  = var_mapping[mixtures_[m][d].var_idx];
      out.write(reinterpret_cast<const char*>(&mean_idx), sizeof(mean_idx));
      out.write(reinterpret_cast<const char*>(&var_idx),  sizeof(var_idx));
    }
  }

  const uint32_t mixture_count = mixtures_.size();
  out.write(reinterpret_cast<const char*>(&mixture_count), sizeof(mixture_count));
  uint32_t current_density = 0u;
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    density_count = mixtures_[m].size();
    out.write(reinterpret_cast<const char*>(&density_count), sizeof(density_count));
    for (size_t d = 0ul; d < mixtures_[m].size(); d++) {
      const size_t mean_idx = mixtures_[m][d].mean_idx;
      out.write(reinterpret_cast<const char*>(&current_density), sizeof(current_density));
      out.write(reinterpret_cast<const char*>(&mean_weight_accumulators_[mean_idx]), sizeof(double));
      current_density++;
    }
  }
}

/*****************************************************************************/
