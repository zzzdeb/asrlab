/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Alignment.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

/*****************************************************************************/

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
  static const double INF = std::numeric_limits<double>::infinity();
}

/*****************************************************************************/

Aligner::Aligner(MixtureModel const& mixtures, TdpModel const& tdp_model, size_t max_aligns)
                 : mixtures_(mixtures), tdp_model_(tdp_model), max_aligns_(max_aligns) {}

/*****************************************************************************/

double Aligner::align_sequence_full(FeatureIter feature_begin, FeatureIter feature_end,
                                    MarkovAutomaton const& reference,
                                    AlignmentIter align_begin, AlignmentIter align_end) {
  
  size_t T = feature_end - feature_begin;
  size_t S = reference.num_states();
  std::vector<double> prev(S);
  std::fill(prev.begin() + 1, prev.end(), INF);
  std::vector<double> cur(S);
  std::vector<uint8_t> B(T*S);

  auto d = [&](size_t t, size_t s) { 
    auto feat = feature_begin + t;
    auto midx = reference[s];
    return mixtures_.score(feat, midx);
  };

  for (size_t t = 0; t < T; t++)
  {
    size_t s = 0;
    {
      // unreachable
      if (S - 1 > (T - t - 1) * 2)
        s = S - 1 - (T - t - 1) * 2;
      std::fill(cur.begin(), cur.begin() + s, INF);
    }

    for (; s < S; s++)
    {
      std::vector<double> to_scores;
      for (size_t i = 0; i <= std::min(2ul, s); i++)
        to_scores.emplace_back(prev[s - i] + tdp_model_.score(s, i));
      auto min = std::min_element(to_scores.begin(), to_scores.end());
      auto argmin = min - to_scores.begin();
      if (*min == INF) {
        cur.at(s) = INF; 
        continue;
      }
      cur.at(s) = d(t, s) + *min;
      B[t*S + s] = argmin;
    }
    // for (size_t i = 0; i < cur.size(); i++)
    //   std::cout << cur.at(i) <<":" << static_cast<int>(B[t*S + i]) << ",";
    // std::cout << std::endl;
    std::swap(prev, cur);
  }
  size_t s = S - 1;
  for(auto it = align_end-1; it != align_begin; it--) {
    size_t t = it - align_begin;
    (*it)->state = reference[s];
    s -= B[t * S + s];
    // std::cout << s << ",";
  }
  (*align_begin)->state = reference[s];
  // std::cout << s << std::endl;
  return prev.at(S-1);
}

/*****************************************************************************/

double Aligner::align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                                      MarkovAutomaton const& reference,
                                      AlignmentIter align_begin, AlignmentIter align_end,
                                      double pruning_threshold) {
  size_t T = feature_end - feature_begin;
  size_t S = reference.num_states();
  std::vector<double> prev(S);
  std::fill(prev.begin() + 1, prev.end(), INF);
  std::vector<double> cur(S);
  std::vector<uint8_t> B(T*S);

  auto d = [&](size_t t, size_t s) { 
    auto feat = feature_begin + t;
    auto midx = reference[s];
    return mixtures_.score(feat, midx);
  };

  for (size_t t = 0; t < T; t++)
  {
    size_t s = 0;
    {
      // unreachable
      if (S - 1 > (T - t - 1) * 2)
        s = S - 1 - (T - t - 1) * 2;
      std::fill(cur.begin(), cur.begin() + s, INF);
    }

    for (; s < S; s++)
    {
      std::vector<double> to_scores;
      for (size_t i = 0; i <= std::min(2ul, s); i++)
        to_scores.emplace_back(prev[s - i] + tdp_model_.score(s, i));
      auto min = std::min_element(to_scores.begin(), to_scores.end());
      auto argmin = min - to_scores.begin();
      if (*min == INF) {
        cur.at(s) = INF; 
        continue;
      }
      cur.at(s) = d(t, s) + *min;
      B[t*S + s] = argmin;
    }
    double threshold = *std::min_element(cur.begin(), cur.end()) + pruning_threshold;
    for(auto& v : cur)
      if (v > threshold)
        v = INF;
    std::swap(prev, cur);
  }
  size_t s = S - 1;
  for(auto it = align_end-1; it != align_begin; it--) {
    size_t t = it - align_begin;
    (*it)->state = reference[s];
    s -= B[t * S + s];
    // std::cout << s << ",";
  }
  (*align_begin)->state = reference[s];
  // std::cout << s << std::endl;
  return prev.at(S-1);
}

/*****************************************************************************/

void dump_alignment(std::ostream& out, Alignment const& alignment, size_t max_aligns) {
  for (size_t f = 0ul; f < (alignment.size() / max_aligns); f++) {
    for (size_t a = 0ul; a < alignment[f * max_aligns].count; a++) {
      const size_t idx = f * max_aligns + a;
      out << f << " " << alignment[idx].state << " " << alignment[idx].weight << std::endl;
    }
  }
}

/*****************************************************************************/

void write_alignment(std::ostream& out, Alignment const& alignment, size_t max_aligns) {
  out.write(reinterpret_cast<char const*>(&max_aligns), sizeof(max_aligns));
  size_t num_frames = alignment.size() / max_aligns;
  out.write(reinterpret_cast<char const*>(&num_frames), sizeof(num_frames));
  out.write(reinterpret_cast<char const*>(alignment.data()), max_aligns * num_frames * sizeof(Alignment::value_type));
}

/*****************************************************************************/

void read_alignment(std::istream& in, Alignment& alignment, size_t& max_aligns) {
  in.read(reinterpret_cast<char*>(&max_aligns), sizeof(max_aligns));
  size_t num_frames = 0ul;
  in.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
  alignment.resize(num_frames * max_aligns);
  in.read(reinterpret_cast<char*>(alignment.data()), max_aligns * num_frames * sizeof(Alignment::value_type));
}

/*****************************************************************************/
