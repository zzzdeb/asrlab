/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __TRAINING_HPP__
#define __TRAINING_HPP__

#include <utility>

#include "Alignment.hpp"
#include "Corpus.hpp"
#include "Lexicon.hpp"
#include "Mixtures.hpp"
#include "Types.hpp"

class Trainer {
public:
  static const ParameterUInt paramMinObs;
  static const ParameterUInt paramNumSplits;
  static const ParameterUInt paramNumAligns;
  static const ParameterUInt paramNumEstimates;
  static const ParameterUInt paramNumMaxAligns;

  static const ParameterDouble paramPruningThreshold;
  static const ParameterDouble paramBwPruningThreshold;
  static const ParameterDouble paramBwPruningThreshold2;

  static const ParameterString paramMixturePath;
  static const ParameterString paramAlignmentPath;
  static const ParameterString paramTrainingStatsPath;

  static const ParameterBool paramWriteLinearSegmentation;
  static const ParameterBool paramRealign;
  static const ParameterBool paramAlignmentPruning;
  static const ParameterBool paramBWTraining;

  Trainer(Configuration const& config, std::shared_ptr<const Lexicon> lexicon, MixtureModel& mixtures, TdpModel const& tdp_model, bool max_approx)
         : min_obs_(paramMinObs(config)), num_splits_(paramNumSplits(config)), num_aligns_(paramNumAligns(config)),
           num_estimates_(paramNumEstimates(config)), num_max_aligns_(paramNumMaxAligns(config)),
           pruning_threshold_(paramPruningThreshold(config)), mixture_path_(paramMixturePath(config)),
           alignment_path_(paramAlignmentPath(config)), training_stats_path_(paramTrainingStatsPath(config)),
           max_approx_(max_approx), write_linear_segmentation_(paramWriteLinearSegmentation(config)),
           realign_(paramRealign(config)), alignment_pruning_(paramAlignmentPruning(config)),
           lexicon_(lexicon), mixtures_(mixtures), aligner_(mixtures_, tdp_model, num_max_aligns_) {}

  void train(Corpus const& corpus);

  MarkovAutomaton           build_segment_automaton(WordIter segment_begin, WordIter segment_end) const;
  std::pair<size_t, size_t> linear_segmentation(MarkovAutomaton const& automaton,
                                                FeatureIter   feature_begin, FeatureIter   feature_end,
                                                AlignmentIter align_begin,   AlignmentIter align_end) const;

  MixtureModel const& get_mixtures() const {
    return mixtures_;
  }

  void set_pthreshold(double v) { pruning_threshold_ = v; }
private:
  const unsigned min_obs_;
  const unsigned num_splits_;
  const unsigned num_aligns_;
  const unsigned num_estimates_;
  const unsigned num_max_aligns_;

  double pruning_threshold_;

  const std::string mixture_path_;
  const std::string alignment_path_;
  const std::string training_stats_path_;

  const bool max_approx_;
  const bool write_linear_segmentation_;
  const bool realign_;
  const bool alignment_pruning_;

  std::shared_ptr<const Lexicon> lexicon_;
  MixtureModel& mixtures_;
  Aligner aligner_;
  struct Count {
      size_t min{std::numeric_limits<size_t>::max()};
      size_t count;
      size_t sum;
  };
  std::unordered_map<WordIdx, Count> count;

  void write_linear_segmentation(std::string const& feature_path, size_t speech_begin, size_t speech_end,
                                 FeatureIter feature_begin, FeatureIter feature_end) const;

  double calc_am_score(Corpus const& corpus, Alignment const& alignment) const;
};

#endif /* __TRAINING_HPP__ */
