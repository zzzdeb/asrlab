/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __OUTPUT_LAYER_HPP__
#define __OUTPUT_LAYER_HPP__

#include "FeedForwardLayer.hpp"

class OutputLayer : public FeedForwardLayer {
public:
  OutputLayer(Configuration const& config);
  virtual ~OutputLayer();

  void forward(std::shared_ptr<linalg::BaseT> output, std::gslice const& slice, std::vector<unsigned> const& mask) const override;
private:
};

// -------------------- inline functions --------------------

inline OutputLayer::OutputLayer(Configuration const& config) : FeedForwardLayer(config) {
}

inline OutputLayer::~OutputLayer() {
}

#endif /* __OUTPUT_LAYER_HPP__ */
