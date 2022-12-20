/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>

#include "Recognizer.hpp"
#include "TdpModel.hpp"
#include "Timer.hpp"

/*****************************************************************************/

namespace {
    static const double INF = std::numeric_limits<double>::infinity();//define the positiv infinity of Double in the namespace
}

/*****************************************************************************/

const ParameterDouble Recognizer::paramAmThreshold   ("am-threshold",   20.0);
const ParameterDouble Recognizer::paramWordPenalty   ("word-penalty",   10.0);

/*****************************************************************************/

void Recognizer::recognize(Corpus const& corpus) {
  EDAccumulator acc;
  size_t ref_total = 0ul;
  size_t sentence_errors = 0ul;
  std::vector<WordIdx> recognized_words;

  for (SegmentIdx s = 0ul; s < corpus.get_corpus_size(); s++) {
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    std::pair<WordIter, WordIter> ref_seq = corpus.get_word_sequence(s);

    recognizeSequence(features.first, features.second, recognized_words);

    EDAccumulator ed = editDistance(ref_seq.first, ref_seq.second, recognized_words.begin(), recognized_words.end());
    acc += ed;
    ref_total += std::distance(ref_seq.first, ref_seq.second);
    if (ed.total_count > 0u) {
      sentence_errors++;
    }

    const double wer = (static_cast<double>(acc.total_count) / static_cast<double>(ref_total)) * 100.0;
    std::cerr << (s + 1ul) << "/" << corpus.get_corpus_size()
              << " WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
              << "% (S/I/D) " << ed.substitute_count << "/" << ed.insert_count << "/" << ed.delete_count << " | ";

    std::copy(recognized_words.begin(), recognized_words.end(), std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << "| ";
    std::copy(ref_seq.first,            ref_seq.second,         std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << std::endl;
  }

  const double wer = (static_cast<double>(acc.total_count) / static_cast<double>(ref_total)) * 100.0;
  const double ser = (static_cast<double>(sentence_errors) / static_cast<double>(corpus.get_corpus_size())) * 100;
  const double time = 0.0; // TODO: compute
  const double rtf = 0.0; // TODO: compute

  std::cerr << "WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
            << "% (S/I/D) " << acc.substitute_count << "/" << acc.insert_count << "/" << acc.delete_count << std::endl;
  std::cerr << "SER: " << std::setw(6) << std::fixed << ser << "%" << std::endl;
  std::cerr << "Time: " << time << " seconds" << std::endl;
  std::cerr << "RTF: " << rtf << std::endl;
}

/*****************************************************************************/

void Recognizer::recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {
  // TODO: implement
  // scorer_.prepare_sequence(feature_begin, feature_end);
  // for(auto feat = feature_begin; feat < feature_end; feat++) {
  //   scorer_.score(feat);
  // }
}

/*****************************************************************************/

EDAccumulator Recognizer::editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end) {
  const auto EDINS = EDAccumulator(1, 0, 1, 0);
  const auto EDDEL = EDAccumulator(1, 0, 0, 1);
  const auto EDSUB = EDAccumulator(1, 1, 0, 0);

  size_t m = ref_end - ref_begin;
  size_t n = rec_end - rec_begin;

  std::vector<EDAccumulator> prev(m + 1);
  for (size_t i = 1; i <= m; i++) {
    prev.at(i) = prev.at(i-1) + EDINS;
  }
  std::vector<EDAccumulator> cur(m + 1);

  for(size_t i = 1; i <= n; i++) {
    for (size_t j = 0; j <= m; j++) {
      if (j == 0) {
        cur.at(j) = prev.at(j) + EDDEL;
        continue;
      }
      std::vector<EDAccumulator> hyps{
          prev.at(j) + EDDEL,
          cur.at(j - 1) + EDINS,
      };
      if (*(ref_begin + j - 1) == *(rec_begin + i - 1))
        hyps.emplace_back(prev.at(j - 1));
      else
        hyps.emplace_back(prev.at(j - 1) + EDSUB);
      cur.at(j) = *std::min_element(hyps.begin(), hyps.end());
    }
    std::swap(cur, prev);
  }
  return prev.back();
}

bool operator==(EDAccumulator const& a, EDAccumulator const& b) {
  return a.total_count == b.total_count && a.delete_count == b.delete_count && a.insert_count == b.insert_count && a.substitute_count == b.substitute_count;
}
bool operator<(EDAccumulator const& a, EDAccumulator const& b) {
  return a.total_count < b.total_count;
}
std::ostream& operator<<(std::ostream& os, EDAccumulator const& b){
  return os << b.total_count << '|' << b.substitute_count << '|' << b.insert_count << '|' << b.delete_count << '|';
}

/*****************************************************************************/
