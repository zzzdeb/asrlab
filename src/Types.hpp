/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <stdint.h>
#include <vector>
#include <utility>

typedef size_t   SegmentIdx;
typedef size_t   WordIdx;
typedef uint16_t StateIdx;
typedef uint16_t DensityIdx;

struct MixtureDensity {
  DensityIdx mean_idx;
  DensityIdx var_idx;

  MixtureDensity(DensityIdx mean_idx, DensityIdx var_idx)
                : mean_idx(mean_idx), var_idx(var_idx) {}
};

typedef std::vector<MixtureDensity> Mixture;

struct AlignmentItem {
  uint16_t count;
  StateIdx state;
  float    weight;

  AlignmentItem() : count(0u), state(0u), weight(0.0) {}//constructor without parameters, then set variables as defaulted values

  AlignmentItem(uint16_t count, StateIdx state, float weight)
               : count(count), state(state), weight(weight) {}//equal to this.count=count; count outside is the variable, inside is the value
};

typedef std::vector<AlignmentItem> Alignment;

typedef std::vector<WordIdx>::const_iterator WordIter;
/*
 const_iterator: C++ defines a type called const_iterator for each container type,
 which can only be used to read the elements inside the container,
 but cannot change its value. Dereferencing the const_iterator type yields a reference to a const object.
 */

#endif /* TYPES_HPP */
