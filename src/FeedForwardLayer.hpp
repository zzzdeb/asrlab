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

  virtual void init(bool input_error_needed);

  virtual void init_parameters(std::function<float()> const& generator);
  void forward (std::shared_ptr<std::valarray<float>> output,
                        std::gslice const& slice, std::vector<unsigned> const& mask) const override;
  virtual void backward_start();
  void backward(std::shared_ptr<std::valarray<float>> output, std::shared_ptr<std::valarray<float>> error,
                        std::gslice const& slice, std::vector<unsigned> const& mask) override;
protected:
  FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity);
private:
  const Nonlinearity nonlinearity_;
};

#endif /* __FEED_FORWARD_LAYER_HPP__ */
