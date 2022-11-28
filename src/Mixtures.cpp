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

  inline void test(bool condition, char const* error_msg) {
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

              var_refs_(var_model == GLOBAL_POOLING ? 1 : num_mixtures, 1),
              vars_(var_refs_.size(), dimension, 1),
              var_accumulators_(var_refs_.size(), dimension),
              var_weight_accumulators_(var_refs_.size()),

              norm_fixed_(dimension * std::log(2 * M_PI) / 2),
              norm_(var_refs_.size()),

              mixtures_(num_mixtures),
              alignment_begin_(nullptr, 1),
              alignment_end_(nullptr, 1),
              write_mixtures_(paramWriteMixtures(config))
{
  for (size_t i = 0; i < num_mixtures; i++)
    mixtures_.at(i).emplace_back(i, var_model == GLOBAL_POOLING ? 0 : i);
  for (size_t i = 0; i < norm_.size(); i++)
    norm_.at(i) = norm_fixed_ - vars_[i].log().sum() / 2;
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
    auto midx = (*alignment)->state;
    const Mixture& mixture = mixtures_.at(midx);
    mixture_accumulators_.at(midx)++;
    auto fvector = ToVector(feature);

    if (max_approx_) {
      auto dens = mixture.front();
      if (!first_pass)
        dens = mixture.at(min_score(feature, midx).second);
      test(mean_refs_.at(dens.mean_idx)!=0, "Naa");

      const double w = 1;
      mean_accumulators_[dens.mean_idx] += w * fvector;
      mean_weight_accumulators_.at(dens.mean_idx) += w;

      // Section features;
      var_accumulators_[dens.var_idx] += w * fvector.square();
      var_weight_accumulators_.at(dens.var_idx) += w;
    } else {
      Vector weights(num_active(mixture), 1);
      if (!first_pass) {
        Vector scores = density_scores_normalized(feature, midx);
        weights = (-scores).exp();
        test(weights.sum() > 0, "Naa");
      }
      for (size_t i = 0; i < weights.size(); i++)
      {
        const auto& dens = mixture.at(get_ith_active(midx, i));
        const double& w = weights[i];
        mean_accumulators_[dens.mean_idx] += w * fvector;
        mean_weight_accumulators_.at(dens.mean_idx) += w;

      // Section features;
        var_accumulators_[dens.var_idx] += w * fvector.square();
        var_weight_accumulators_.at(dens.var_idx) += w;
      }
    }
  }
  // cprint(mean_refs_)
  // std::vector<size_t> tmp;
  // std::transform(mixtures_.begin(), mixtures_.end(), std::back_inserter(tmp), [](auto& v) {return v.size();});
  // cprint(tmp);
}

void MixtureModel::visualize(std::string header) {
  if (!write_mixtures_)
    return;
  stats_out << "==" << header << std::endl;
  size_t index = 0;
  auto& mixture = mixtures_.at(index);
  for (auto& m: mixture) {
    stats_out << "n ";
    stats_out << m.mean_idx << " ";
    stats_out << mean_refs_.at(m.mean_idx) << " ";
    stats_out << mean_weights_.at(m.mean_idx) << " ";
    stats_out << means_[m.mean_idx] << " ";
    stats_out << var_refs_.at(m.var_idx) << " ";
    stats_out << m.var_idx << " ";
    stats_out << vars_[m.var_idx];

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
  vars_ = 0;
  // Implements means_, vars_, norm_ and weights_
  for (size_t midx = 0; midx < mixtures_.size(); midx++) {
    const auto& mixture = mixtures_.at(midx);
    for (const auto& dens : mixture)
    {
      size_t i = dens.mean_idx;
      if (mean_refs_.at(i) == 0)
        continue;
      if (mean_weight_accumulators_.at(i) == 0) {
        if (num_active(mixture) != 1) {
        // test(num_active(mixture) != 1, "It will be empty.");
          mean_refs_.at(i) = 0;
          var_refs_.at(dens.var_idx)--;
        }
        continue;
      }

      means_[i] = mean_accumulators_[i] / mean_weight_accumulators_.at(i);
      mean_weights_.at(i) = -std::log(mean_weight_accumulators_.at(i) / mixture_accumulators_.at(midx));

      if (var_model == NO_POOLING) {
        vars_[i] = 1 / (var_accumulators_[i] / var_weight_accumulators_.at(i) - means_[i].square()).nonzero();
        norm_.at(i) = norm_fixed_ - (vars_[i].log().sum() / 2);
      } else if (var_model == MIXTURE_POOLING)
        vars_[midx] -= mean_weight_accumulators_.at(i) * means_[i].square();
      else if (var_model == GLOBAL_POOLING)
        vars_[0] -= mean_weight_accumulators_.at(i) * means_[i].square();
    }
    if (var_model == MIXTURE_POOLING && var_weight_accumulators_.at(midx) != 0) {
      vars_[midx] += var_accumulators_[midx];
      vars_[midx] = 1 / (vars_[midx]/var_weight_accumulators_.at(midx)).nonzero();
      norm_.at(midx) = norm_fixed_ - (vars_[midx].log().sum() / 2);
    }
  }
  if (var_model == GLOBAL_POOLING) {
      vars_[0] += var_accumulators_[0];
      vars_[0] = 1 / (vars_[0]/var_weight_accumulators_.at(0)).nonzero();
      norm_.at(0) = norm_fixed_ - (vars_[0].log().sum() / 2);
  }
  check_validity();
  visualize("f");
}

void MixtureModel::check_validity() {
  for (const auto& w : mean_weights_)
    test(w>=0, "Na");
  
  for (size_t i = 0; i < mixtures_.size(); i++) {
    size_t active_mxts = 0;
    for (const auto& mixture: mixtures_.at(i))
    {
      if (mean_refs_.at(mixture.mean_idx) != 0)
        active_mxts++;
    }
    test(active_mxts > 0, "All densities disabled");
  }
  
  for (size_t i = 0; i < mean_weights_.size(); i++) {
    if (mean_refs_.at(i) == 0)
      continue;
    // for (vars_(i))
  }
  // auto check_norm = [](double v) { return v <10000000 && v > -10000000;};
  // for (size_t i = 0; i < norm_.size(); i++){
  //   test(check_norm(norm_.at(i)), "Naa");
  // }
  // for (size_t i = 0; i < mixture_accumulators_.size(); i++)
  //   test(mixture_accumulators_.at(i) > 0, "Naa");
  auto check_vars = [](double v) { return v >0;};
  // for (size_t i = 0; i < vars_.size().first; i++)
  //   for (size_t j = 0; j < vars_.size().second; j++)
  //     if(!check_vars(vars_(i, j)))
  //       throw std::logic_error("Naa");
}

/*****************************************************************************/

void MixtureModel::split(size_t min_obs) {
  for (auto& mixture : mixtures_) {
    for (size_t i = 0; i < mixture.size(); i++)
    {
      DensityIdx midx = mixture.at(i).mean_idx;
      DensityIdx vidx = mixture.at(i).var_idx;
      if (mean_refs_.at(midx) == 0)
        continue;
      if (mean_weight_accumulators_.at(midx) >= min_obs) {
        auto cur_mean = means_[midx];
        auto diff = (1 / vars_[vidx]).sqrt();
        auto p_mean = cur_mean + diff;
        auto n_mean = cur_mean - diff;
        means_[midx] = p_mean;
        mean_weights_.at(midx) += log(2);
        {
          // add mean
          means_.add_row(n_mean);
          mean_accumulators_.add_row();
          mean_weights_.emplace_back(mean_weights_.at(midx));
          mean_weight_accumulators_.emplace_back();
          mean_refs_.emplace_back(1);
        }
        size_t add_vidx = 0;
        switch(var_model)
        {
          case GLOBAL_POOLING:
            add_vidx = 0;
          break;
          case MIXTURE_POOLING:
            add_vidx = vidx;
          break;
          case NO_POOLING:
            add_vidx = mean_weights_.size() - 1;
          vars_.add_row(vars_[vidx]);
          var_accumulators_.add_row();
          var_weight_accumulators_.emplace_back();
            var_refs_.emplace_back();
          norm_.emplace_back(norm_.at(vidx));
          break;
        }
        var_refs_.at(add_vidx)++;
        mixture.emplace_back(mean_weights_.size() - 1, add_vidx);
      }
    }
  }
  visualize("s");
}

/*****************************************************************************/

void MixtureModel::eliminate(double min_obs) {
  for (const auto& mixture : mixtures_)
    for (const auto& dens : mixture) {
      if (mean_refs_.at(dens.mean_idx) == 0)
        continue;
      if (mean_weight_accumulators_.at(dens.mean_idx) < min_obs && num_active(mixture) != 1) {
        test(num_active(mixture) != 1, "It will be empty.");
        mean_refs_.at(dens.mean_idx) = 0;
        var_refs_.at(dens.var_idx)--;
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
double MixtureModel::density_score(FeatureIter const& iter, const MixtureDensity& density) const {
  if (mean_refs_.at(density.mean_idx) == 0)
    return std::numeric_limits<double>::infinity();
  const double& minus_log_c = mean_weights_.at(density.mean_idx);

  double p = minus_log_c + norm_.at(density.var_idx);
  size_t midx = density.mean_idx;
  size_t vidx = density.var_idx;
  const double add = ((Vector(*iter, *iter+dimension) - means_[midx]).square() * vars_[vidx]).sum()/2;
  p += add;
  if (p < 0) // single point mixtures
    p = 0;
  test(p>=0 && p < std::numeric_limits<double>::infinity(), "Invalid prob");
  return p;
}

/*****************************************************************************/

Vector MixtureModel::density_scores(FeatureIter const& iter, StateIdx mixture_idx) const {
  const Mixture& mixture = mixtures_.at(mixture_idx);
  std::vector<double> scores;
  for (const auto& dens: mixture)
  {
    if (mean_refs_.at(dens.mean_idx) == 0)
      continue;
    scores.emplace_back(density_score(iter, dens));
  }
  test(scores.size() == num_active(mixture), "Naa");
  return {std::move(scores)};
}

Vector MixtureModel::density_scores_normalized(FeatureIter const& iter, StateIdx mixture_idx) const {
  if (num_active(mixtures_.at(mixture_idx)) == 1)
    return {1};
  Vector scores = density_scores(iter, mixture_idx);
  double nsum = sum_score(iter, mixture_idx, nullptr);
  return scores - nsum;
}
  
// this function returns the density with the lowest score (=highest probability)
// for the given feature vector
std::pair<double, DensityIdx> MixtureModel::min_score(FeatureIter const& iter, StateIdx mixture_idx) const {
  const Mixture& mixture = mixtures_.at(mixture_idx);
  Vector scores = density_scores(iter, mixture_idx);
  auto min_score = scores.min();
  test(*min_score != 65535 && *min_score != std::numeric_limits<double>::infinity(), "It should be smaller");
  test(min_score - scores.begin < num_active(mixture), "Naa");
  return std::make_pair(*min_score, get_ith_active(mixture_idx, min_score - scores.begin));
}

/*****************************************************************************/

// compute the 'full' score of a feature vector for a given mixture. The weights
// of each density are stored in weights and should sum up to 1.0
double MixtureModel::sum_score(FeatureIter const& iter, StateIdx mixture_idx, std::vector<double>* weights) const {//here the weights is not used?
  Vector scores = density_scores(iter, mixture_idx);
  if (scores.size() == 1)
    return *scores.begin;
  double nsum = *scores.begin;
  for(auto it = scores.begin+1; it < scores.end; it++)
    nsum = logsum(nsum, *it);
  return nsum;
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

size_t MixtureModel::get_ith_active(StateIdx mixture_idx, size_t i) const {
  const auto& mixture = mixtures_.at(mixture_idx);
  int ret = -1;
  int ret_idx = -1;
  for (size_t j = 0; j < mixture.size(); j++)
  {
    if (mean_refs_.at(mixture.at(j).mean_idx) != 0)
      ret++;
    if (ret == i) {
      ret_idx = j;
      break;
    }
  }
  test(ret_idx < mixture.size() && ret_idx >= i, "Naa");
  return ret_idx;
}
size_t MixtureModel::num_active(const Mixture& m) const {
  size_t ret = 0;
  for (const auto& dens: m)
    if (mean_refs_.at(dens.mean_idx) != 0)
      ret++;
  return ret;
}

/*****************************************************************************/
