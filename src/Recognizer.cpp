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
    template <class T>
    void printv(const std::vector<T>& w) {
        for (const auto& a : w) {
          std::cout << a << " ";
        }
        std::cout << std::endl;
    };
    template<class T>
    void printdv(std::vector<std::vector<T>> v) {
        for (auto& a : v) {
          for (auto &b: a) {
            std::cout << b << " ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
    };
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
  scorer_.prepare_sequence(feature_begin, feature_end);
  size_t T = feature_end - feature_begin;
  std::vector<std::vector<double>> cur;
  std::vector<std::vector<double>> prev;
  auto resize = [&](std::vector<std::vector<double>>& v) {
    v.resize(lexicon_->num_words());
    for (WordIdx w = 0; w < lexicon_->num_words(); w++) {
      v.at(w).resize(lexicon_->get_automaton_for_word(w).num_states() + 1);
      std::fill(v.at(w).begin() + 1, v.at(w).end(), INF);
    }
  };
  resize(cur);
  resize(prev);
  prev.front().back() = 0;
  std::vector<std::vector<int>> B(lexicon_->num_words());
  size_t B_width = T + 1;
  for (WordIdx w = 0; w < lexicon_->num_words(); w++)
    B.at(w).resize(B_width * (lexicon_->get_automaton_for_word(w).num_states() + 1));
  std::vector<int> B1(T + 1);

  std::vector<WordIdx> W(T + 1);
  auto d = [&](size_t t, size_t s, WordIdx& w) -> double {
      auto feat = feature_begin + t - 1;
      auto state = lexicon_->get_automaton_for_word(w)[s - 1];
      return scorer_.score(feat, state);
  };

  auto S = [&](const WordIdx& w) -> size_t { return lexicon_->get_automaton_for_word(w).num_states(); };
  // scorer_.prepare_sequence(feature_begin, feature_end);
  for(size_t t = 1; t <= T; t++) {
    for (WordIdx w = 0; w < lexicon_->num_words(); w++) {
      prev.at(w).at(0) = prev.at(W.at(t-1)).at(S(W.at(t-1))) + word_penalty_;
      B.at(w).at(t - 1 + 0 * B_width) = t - 1;
      size_t s = 1;
      for (; s <= S(w); s++) {
        std::vector<double> to_scores;
        for (size_t i = 0; i <= std::min(2ul, s); i++)
          to_scores.emplace_back(prev.at(w).at(s - i) + tdp_model_.score(lexicon_->get_automaton_for_word(w)[s - 1], i));
        auto min = std::min_element(to_scores.begin(), to_scores.end());
        size_t argmin = min - to_scores.begin();
        if (*min == INF) {
          cur.at(w).at(s) = INF;
          continue;
        }
        cur.at(w).at(s) = d(t, s, w) + *min;
        B.at(w).at(t + s * T) = B.at(w).at(t - 1 + T * (s - argmin));
      }
    }
    WordIdx w_min = 0;
    double min_score = INF;
    for (WordIdx w = 0; w < lexicon_->num_words(); w++) {
      if (cur.at(w).back() < min_score) {
        min_score = cur.at(w).back();
        w_min = w;
      }
    }
    B1.at(t) = B.at(w_min).at(t + T * S(w_min));
    W.at(t) = w_min;
    std::swap(prev, cur);
  }
  size_t n = 0;
  size_t t = T;
  output.clear();
  while (t > 0) {
    n++;
    if (W.at(t) != lexicon_->silence_idx())
      output.insert(output.cbegin(), W.at(t));
    t = B1.at(t);
  }
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
