/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "FeedForwardLayer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <assert.h>
#include <cblas.h>

#include "Util.hpp"

namespace {
  FeedForwardLayer::Nonlinearity nonlinearity_from_string(std::string const& str) {
    if (str == "sigmoid") {
      return FeedForwardLayer::Nonlinearity::Sigmoid;
    }
    else if (str == "tanh") {
      return FeedForwardLayer::Nonlinearity::Tanh;
    }
    else if (str == "relu") {
      return FeedForwardLayer::Nonlinearity::ReLU;
    }
    return FeedForwardLayer::Nonlinearity::None;
  }
}
using namespace linalg;

const ParameterString FeedForwardLayer::paramNonlinearity("nonlinearity", "");

FeedForwardLayer::FeedForwardLayer(Configuration const& config) : NetworkLayer(config),
                                                                  nonlinearity_(nonlinearity_from_string(paramNonlinearity(config))) {
}

FeedForwardLayer::~FeedForwardLayer() {
}

void FeedForwardLayer::init(bool input_error_needed) {
  NetworkLayer::init(input_error_needed);
  params_->resize(feature_size_ * output_size_ + output_size_);
  W_.reset(params_, {feature_size_, output_size_});
  eW_.resize(feature_size_, output_size_);
  b_.reset(params_, {output_size_}, feature_size_*output_size_);
  eb_.resize(output_size_);
  gradient_->resize(params_->size());
  dW_.reset(gradient_, {feature_size_, output_size_});
  db_.reset(gradient_, {output_size_}, feature_size_*output_size_);
}

void FeedForwardLayer::init_parameters(std::function<float()> const& generator) {
  for (size_t i = 0ul; i < params_->size(); i++) {
      (*params_)[i] = generator();
  }
}


void FeedForwardLayer::forward(std::shared_ptr<std::valarray<float>> output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  // slice = max_seq, 7000 x batch_size, 200 x output_size, 1
  Tensor output_tensor(output, slice);
  #pragma omp parallel for
  for (size_t i = 0; i < mask.size(); i++) {
      Matr eseq_matrix = input_tensor_({0, mask.at(i)}, {i}, ALL).mat();
      output_tensor({0, mask.at(i)}, i, ALL).mat() = (eseq_matrix * eW_).rowwise() + eb_.transpose();
  }
  switch(nonlinearity_) {
    case Nonlinearity::None:
    break;
    case Nonlinearity::ReLU:
      output_tensor.relu();
    break;
    case Nonlinearity::Sigmoid:
      output_tensor.sigmoid();
    break;
    case Nonlinearity::Tanh:
      output_tensor.tanh();
    break;
    default:
      throw std::invalid_argument("No type");
  }
}

void FeedForwardLayer::backward_start() {
  *error_buffer_ = 0.0f;
  *gradient_     = 0.0f;
}

void FeedForwardLayer::backward(std::shared_ptr<std::valarray<float>> output, std::shared_ptr<std::valarray<float>> error,
                                std::gslice const& slice, std::vector<unsigned> const& mask) {
    nonlinear_backward(output, error, slice, mask);
    // dENL = dE * dNL
    Tensor dE_tensor(error, slice);
    float num_features = std::accumulate(mask.cbegin(), mask.cend(), 0);
    // dW = input * dENL
    #pragma omp parallel for
    for(size_t j = 0; j < batch_size_; j++)
        for(size_t i = 0; i < mask.at(j); i++)
            dW_ += input_tensor_.at(i, j).outer(dE_tensor.at(i, j));
    db_ = dE_tensor.sumx();
    for(auto& v : *gradient_)
        v /= num_features;

    // error = dENL * W (dENL * dL)
    if (input_error_needed_) {
        #pragma omp parallel for
        for(size_t i = 0; i < mask.size(); i++)
        {
            Matr eseq_matrix = dE_tensor({0, mask.at(i)}, {i}, ALL).mat();
            error_tensor_({0, mask.at(i)}, i, ALL).mat() = eseq_matrix * eW_.transpose();
        }
    }
}

void FeedForwardLayer::nonlinear_backward(std::shared_ptr<std::valarray<float>> output, std::shared_ptr<std::valarray<float>> error,
        std::gslice const& slice, std::vector<unsigned> const& mask) {
    Tensor output_tensor(output, slice);
    Tensor dE_tensor(error, slice);
    // Out = output; dE = error
    switch(nonlinearity_) {
        case Nonlinearity::None:
            // dENL = dE
            break;
        case Nonlinearity::ReLU:
            // dENL = dE * dOut(out <= 0 -> 0 , out > 0 -> 1)
            output_tensor.drelu();
            dE_tensor *= output_tensor;
            break;
        case Nonlinearity::Sigmoid:
            // dENL = dE * dOut(out* (1-out))
            output_tensor.dsigmoid();
            dE_tensor *= output_tensor;
            break;
        case Nonlinearity::Tanh:
            // dENL = dE * dOut(1-tanh^2(out))
            output_tensor.dtanh();
            dE_tensor *= output_tensor;
            break;
        default:
            throw std::invalid_argument("No type");
    }
}

FeedForwardLayer::FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity)
                                  : NetworkLayer(config), nonlinearity_(nonlinearity) {
}

