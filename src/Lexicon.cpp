/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Lexicon.hpp"

#include <cassert>
#include <iostream>

WordIdx Lexicon::add_word(std::string const& orth, uint16_t num_states, uint16_t state_repetitions, bool silence) {
  WordIdx word_idx = automata_.size();
  if (silence) {
    silence_ = word_idx;
  }
  StateIdx start_state = automata_.empty() ? 0u : (automata_[automata_.size()-1u].last_state() + 1u);

  orth_.push_back(orth);
  automata_.push_back(MarkovAutomaton(start_state, num_states, state_repetitions));
  num_states_ = automata_[automata_.size() - 1u].last_state() + 1u;

  return word_idx;
}

/*****************************************************************************/

MarkovAutomaton const& Lexicon::get_silence_automaton() const {
  return automata_[silence_];
}

/*****************************************************************************/

MarkovAutomaton const& Lexicon::get_automaton_for_word(WordIdx word_idx) const {
  return automata_[word_idx];
}

/*****************************************************************************/

StateIdx Lexicon::num_states() const {
  return num_states_;
}

/*****************************************************************************/

WordIdx Lexicon::num_words() const {
  return automata_.size();
}

/*****************************************************************************/

WordIdx Lexicon::silence_idx() const {
  return silence_;
}

/*****************************************************************************/

WordIdx Lexicon::operator[](std::string const& orth) const {
  // TODO: faster search
  auto iter = std::find(orth_.begin(), orth_.end(), orth);
  if (iter == orth_.end()) {
    std::cerr << "unknown word: '" << orth << "'" << std::endl; 
    return -1; // TODO: add unknown word
  }
  else {
    return iter - orth_.begin();
  }
}

/*****************************************************************************/

namespace {
  static const ParameterString paramWord          ("orthography", "<unk>");
  static const ParameterUInt   paramNumStates     ("num-states", 1);
  static const ParameterUInt   paramNumRepetitions("num-repetitions", 1);
  static const ParameterBool   paramIsSilence     ("is-silence", false);
}

std::shared_ptr<Lexicon> build_lexicon_from_config(Configuration const& config) {
  std::shared_ptr<Lexicon> result = std::make_shared<Lexicon>();
  assert(config.is_array("lexicon"));
  auto words = config.get_array<Configuration>("lexicon");
  for (auto& w : words) {
    std::string word            = paramWord(w);
    uint32_t    num_states      = paramNumStates(w);
    uint32_t    num_repetitions = paramNumRepetitions(w);
    bool        is_silence      = paramIsSilence(w);
    result->add_word(word, num_states, num_repetitions, is_silence);
  }
  return result;
}

/*****************************************************************************/
