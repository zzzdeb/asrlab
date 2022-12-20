/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __FEED_FORWARD_LAYER_HPP__
#define __FEED_FORWARD_LAYER_HPP__

#include "NetworkLayer.hpp"

class FeedForwardLayer : public NetworkLayer {
public:
  enum struct Nonlinearity {
    None,
    Sigmoid,
    Tanh,
    ReLU,
  };

  static const ParameterString paramNonlinearity;

  FeedForwardLayer(Configuration const& config);
  virtual ~FeedForwardLayer();

  void init(bool input_error_needed) override;

  void init_parameters(std::function<float()> const& generator) override;
  void forward(BufferT& outputs_, std::vector<unsigned> const& mask) const override;
  void backward_start() override;
  void backward(BufferT& output, BufferT& error,
                        std::vector<unsigned> const& mask) override;
  void nonlinear_backward(BufferT& output, BufferT& error, std::vector<unsigned> const& mask);
protected:
  FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity);
private:
  const Nonlinearity nonlinearity_;
};

#endif /* __FEED_FORWARD_LAYER_HPP__ */
