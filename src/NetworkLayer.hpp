/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __NETWORK_LAYER_HPP__
#define __NETWORK_LAYER_HPP__

#include <valarray>
#include <functional>

#include "Types.hpp"
#include "Config.hpp"
#include "LinAlg.h"

class NetworkLayer {
public:
  static const ParameterString paramLayerName;
  static const ParameterUInt   paramOutputSize;

  NetworkLayer(Configuration const& config);
  virtual ~NetworkLayer();

  virtual void init(bool input_error_needed);

  std::string const& get_layer_name() const;
  void               set_input_sizes(size_t feature_size, size_t batch_size, size_t max_seq_length);
  size_t             get_output_size() const;

  std::vector<std::string> const& get_input_layer_names() const;

  BufferT& get_input_buffer();
  BufferT& get_error_buffer();
  std::shared_ptr<std::valarray<float>> get_params();
  std::shared_ptr<std::valarray<float>> get_gradient();

  virtual void init_parameters(std::function<float()> const& generator) = 0;
  virtual void forward (std::vector<Matr>& outputs_,
                        std::vector<unsigned> const& mask) const = 0;
  virtual void backward_start() = 0;
  virtual void backward(BufferT& output, BufferT& error, std::vector<unsigned> const& mask) = 0;
  virtual void save(std::string const& path) const;
  virtual void load(std::string const& path);
  void update();
protected:
  const std::string layer_name_;

  size_t       feature_size_;
  size_t       batch_size_;
  size_t       max_seq_length_;
  const size_t output_size_;
  bool         input_error_needed_;

  std::vector<std::string> input_layer_names_;

  BufferT input_buffer_;
  BufferT error_buffer_;

  std::shared_ptr<std::valarray<float>> params_; // feature_size * output_size + output_size
  linalg::Matrix W_;
  Matr eW_;
  linalg::Vector b_;
  Vect eb_;
  std::shared_ptr<std::valarray<float>> gradient_;
  linalg::Matrix dW_;
  Matr edW_;
  linalg::Vector db_;
  Vect edb_;
};

// Some methods are defined in the header to allow inlining

inline NetworkLayer::NetworkLayer(Configuration const& config)
                                 : layer_name_(paramLayerName(config)),
                                   feature_size_(0ul), // these 3 are set later via set_input_sizes
                                   batch_size_(0ul),
                                   max_seq_length_(0ul),
                                   output_size_(paramOutputSize(config)),
                                   input_layer_names_(config.get_array<std::string>("input")) ,
                                   input_buffer_(batch_size_),
                                   error_buffer_(batch_size_),
                                   params_(std::make_shared<linalg::BaseT>()),
                                   W_(params_, {feature_size_, output_size_}),
                                   b_(params_, {output_size_}, feature_size_*output_size_),
                                   gradient_(std::make_shared<linalg::BaseT>()),
                                   dW_(gradient_, {feature_size_, output_size_}),
                                   db_(gradient_, {output_size_}, feature_size_*output_size_)
{
}
 
inline NetworkLayer::~NetworkLayer() {
}

inline std::string const& NetworkLayer::get_layer_name() const {
  return layer_name_;
}

inline void NetworkLayer::set_input_sizes(size_t feature_size, size_t batch_size, size_t max_seq_length) {
  feature_size_   = feature_size;
  batch_size_     = batch_size;
  max_seq_length_ = max_seq_length;
  input_buffer_.resize(batch_size);
  error_buffer_.resize(batch_size);
}

inline size_t NetworkLayer::get_output_size() const {
  return output_size_;
}

inline std::vector<std::string> const& NetworkLayer::get_input_layer_names() const {
  return input_layer_names_;
}

inline BufferT& NetworkLayer::get_input_buffer() {
  return input_buffer_;
}

inline BufferT& NetworkLayer::get_error_buffer() {
  return error_buffer_;
}

inline std::shared_ptr<std::valarray<float>> NetworkLayer::get_params() {
  return params_;
}

inline std::shared_ptr<std::valarray<float>> NetworkLayer::get_gradient() {
  return gradient_;
}

#endif /* __NETWORK_LAYER_HPP__ */
