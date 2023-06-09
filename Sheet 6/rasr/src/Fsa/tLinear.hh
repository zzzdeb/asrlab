// Copyright 2011 RWTH Aachen University. All rights reserved.
//
// Licensed under the RWTH ASR License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef _T_FSA_LINEAR_HH
#define _T_FSA_LINEAR_HH

#include "Types.hh"
#include <vector>

namespace Ftl {

/**
 * Test if automaton is linear explicitly.
 * You should use hasProperty() to avoid redundant checks.
 */
template <class _Automaton> bool isLinear(typename _Automaton::ConstRef);

/**
 * Convert linear automaton to vector of input symbols.
 * The automaton must be linear.  Epsilon labels are suppressed.
 */
template <class _Automaton>
void getLinearInput(typename _Automaton::ConstRef,
                    std::vector<Fsa::LabelId> &result);

/**
 * Convert linear automaton to vector of output symbols.
 * The automaton must be linear.  Epsilon labels are suppressed.
 */
template <class _Automaton>
void getLinearOutput(typename _Automaton::ConstRef,
                     std::vector<Fsa::LabelId> &result);

/**
 * Compute total weight of linear automaton.
 * The automaton must be linear.
 */
template <class _Automaton>
typename _Automaton::Weight getLinearWeight(typename _Automaton::ConstRef);

} // namespace Ftl

#include "tLinear.cc"

#endif // _T_FSA_LINEAR_HH
