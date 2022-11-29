/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __MARKOV_AUTOMATON_HPP__ 
#define __MARKOV_AUTOMATON_HPP__

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "Types.hpp"

/*
 * In C++, struct and class are basically common, the only difference is that if you use the class keyword,
 * the member variables or member functions defined in the class are all private by default, while with the struct keyword,
 * the member variables or member functions defined in the structure are all public by default.
 */
struct MarkovAutomaton {
  std::vector<StateIdx> states;

  MarkovAutomaton() {}

  MarkovAutomaton(StateIdx start, uint16_t num, uint16_t repetitions) : states(num*repetitions) {
    std::vector<StateIdx>::iterator iter = states.begin();
    for (StateIdx s = start; s < start + num; s++) {//ids: start,start+1,...,start+num-1
      std::fill_n(iter, repetitions, s);//every s repeat "repetitions" times
      iter += repetitions;
    }
  }

  //https://blog.csdn.net/SMF0504/article/details/52311207
  StateIdx first_state() const {
    return states[0u];
  }

  StateIdx last_state() const {
    return states[states.size() - 1u];
  }

  size_t num_states() const {
    return states.size();
  }

  StateIdx max_state() const {
    return std::accumulate(states.begin(), states.end(), std::numeric_limits<StateIdx>::min(),//numeric_limits: the min from <current> type(uint16_t)
                           static_cast<StateIdx const&(*)(StateIdx const&, StateIdx const&)>(std::max<StateIdx>));//?
  }

  StateIdx& operator[](size_t idx) {//define the MarkovAutomaton[] ?
    return states[idx];
  }
  //what's the difference?
  StateIdx operator[](size_t idx) const {
    return states[idx];
  }

  static MarkovAutomaton concat(std::vector<MarkovAutomaton const*> automata) {
    MarkovAutomaton result;
    result.states.resize(std::accumulate(automata.cbegin(), automata.cend(), 0, [](int &sum, const auto *a)
                                         { return sum + a->states.size(); }));
    auto current_pos = result.states.begin();
    for (const auto automat : automata)
      current_pos = std::copy(automat->states.cbegin(), automat->states.cend(), current_pos);
    return result;
  }
};

#endif /* __MARKOV_AUTOMATON_HPP__ */
