/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __MIXTURES_HPP__
#define __MIXTURES_HPP__

#include <algorithm>
#include <vector>

#include "Config.hpp"
#include "Corpus.hpp"
#include "FeatureScorer.hpp"
#include "Types.hpp"
#include "pgm/Matrix.hpp"

class MixtureModel : public FeatureScorer {
public:
  enum VarianceModel {
    GLOBAL_POOLING,
    MIXTURE_POOLING,
    NO_POOLING
  };

  static const ParameterString paramLoadMixturesFrom;

  const size_t dimension;
  const VarianceModel var_model;

  MixtureModel(Configuration const& config, size_t dimension, size_t num_mixtures,
               VarianceModel var_model, bool max_approx);

  void reset_accumulators();
  void accumulate(ConstAlignmentIter alignment_begin, ConstAlignmentIter alignment_end,
                  FeatureIter        feature_begin,   FeatureIter        feature_end,
                  bool first_pass, bool max_approx=true);
  void finalize();
  void split(size_t min_obs);
  void eliminate(double min_obs);
  void check_validity();

  size_t num_densities() const;

  double                        density_score  (FeatureIter const& iter, StateIdx mixture_idx, DensityIdx density_idx) const;
  std::pair<double, DensityIdx> min_score      (FeatureIter const& iter, StateIdx mixture_idx) const;
  double                        sum_score      (FeatureIter const& iter, StateIdx mixture_idx, std::vector<double>* weights) const;

  virtual void prepare_sequence(FeatureIter const& start, FeatureIter const& end);
  virtual double score(FeatureIter const& iter, StateIdx mixture_idx) const;

  void read(std::istream& in);
  void write(std::ostream& out) const;

private:
  static const char     magic[8];
  static const uint32_t version;

  bool max_approx_;

  Matrix means_;                                 // current mean
  Matrix mean_accumulators_;                     // temp. accumulator (first order stat.) used in reestimation
  std::vector<double> mean_weights_;             // weight of the density
  std::vector<double> mean_weight_accumulators_; // temp. accumulator used in reestimation
  std::vector<size_t> mean_refs_;                // reference counter

  Matrix vars_;                                  // current variance
  Matrix var_accumulators_;                      // temp. accumulator (second order stat.) used in reestimation
  std::vector<double> var_weight_accumulators_;  // temp. accumulator used in reestimation
  std::vector<size_t> var_refs_;                 // reference counter

  const double norm_fixed_;                          //
  std::vector<double> norm_;                     // normalization factor for gaussian distribution

  std::vector<Mixture> mixtures_;
};

#endif /* __MIXTURES_HPP__ */
