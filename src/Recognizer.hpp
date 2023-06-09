/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __RECOGNIZER_HPP__
#define __RECOGNIZER_HPP__

#include "Corpus.hpp"
#include "Lexicon.hpp"
#include "Mixtures.hpp"
#include "TdpModel.hpp"

struct EDAccumulator {
  uint16_t total_count;
  uint16_t substitute_count;
  uint16_t insert_count;
  uint16_t delete_count;

  EDAccumulator() : total_count(0u), substitute_count(0u), insert_count(0u), delete_count(0u) {}
  EDAccumulator(uint16_t total_count, uint16_t substitute_count, uint16_t insert_count, uint16_t delete_count)
               : total_count(total_count), substitute_count(substitute_count), insert_count(insert_count), delete_count(delete_count) {}

  EDAccumulator operator+(EDAccumulator const& other) const {
    return EDAccumulator(total_count      + other.total_count,
                         substitute_count + other.substitute_count,
                         insert_count     + other.insert_count,
                         delete_count     + other.delete_count);
  }

  EDAccumulator& operator+=(EDAccumulator const& other) {
    total_count      += other.total_count;
    substitute_count += other.substitute_count;
    insert_count     += other.insert_count;
    delete_count     += other.delete_count;
    return *this;
  }

  void reset() { total_count = 0u; substitute_count = 0u; insert_count = 0u; delete_count = 0u; }
  void substitution_error() { total_count++; substitute_count++; }
  void insertion_error()    { total_count++; insert_count++;     }
  void deletion_error()     { total_count++; delete_count++;     }
};

bool operator==(EDAccumulator const& a, EDAccumulator const& b);
bool operator<(EDAccumulator const& a, EDAccumulator const& b);
std::ostream& operator<<(std::ostream& os, EDAccumulator const& b);

class Recognizer {
public:
  static const ParameterBool   paramLookahead;
  static const ParameterDouble paramAmThreshold;
  static const ParameterDouble paramLookaheadScale;
  static const ParameterDouble paramWordPenalty;

  Recognizer(Configuration const& config, std::shared_ptr<const Lexicon> lexicon, FeatureScorer& scorer, TdpModel const& tdp_model)
            : am_threshold_(paramAmThreshold(config)),
              word_penalty_(paramWordPenalty(config)), lexicon_(lexicon), scorer_(scorer), tdp_model_(tdp_model) {}

  void recognize(Corpus const& corpus);
  void recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output);

  static EDAccumulator editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end);
  auto& threshold() { return am_threshold_; }
private:
  double am_threshold_;
  const double word_penalty_;

  std::shared_ptr<const Lexicon>& lexicon_;
  FeatureScorer&  scorer_;
  TdpModel const& tdp_model_;
};

#endif /* __RECOGNIZER_HPP__ */
