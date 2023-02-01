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
#include "Timer.hpp"

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


void FeedForwardLayer::forward(BufferT& outputs_, std::vector<unsigned> const& mask) const {
  size_t true_batch_size = 0;
  for (;true_batch_size<outputs_.size();true_batch_size++)
      if (outputs_.at(true_batch_size).rows() == 0)
          break;
  #pragma omp parallel for
  for (size_t i = 0; i < mask.size(); i++) {
      const Matr& eseq_matrix = input_buffer_.at(i);
      outputs_[i] = (eseq_matrix * eW_).rowwise() + eb_.transpose();
  }
  auto sigmoid = [](const float& v) -> float { return 1 / (1 - std::exp(-v));};
  auto relu = [](const float& v) { return v>0 ? v : 0; };
  auto tanh = [](const float& v) { return std::tanh(v); };
  switch(nonlinearity_) {
    case Nonlinearity::None:
    break;
    case Nonlinearity::ReLU:
        for(size_t i = 0; i< outputs_.size(); i++) {
            outputs_.at(i) = outputs_.at(i).unaryExpr(relu);
        }
    break;
    case Nonlinearity::Sigmoid:
        for(size_t i = 0; i< outputs_.size(); i++) {
            outputs_.at(i) = outputs_.at(i).unaryExpr(sigmoid);
        }
    break;
    case Nonlinearity::Tanh:
        for(size_t i = 0; i< outputs_.size(); i++) {
            outputs_.at(i) = outputs_.at(i).unaryExpr(tanh);
        }
    break;
    default:
      throw std::invalid_argument("No type");
  }
}

void FeedForwardLayer::backward_start() {
  std::for_each(error_buffer_.begin(), error_buffer_.end(), [](auto& v) {v.fill(0.0f);});
  *gradient_     = 0.0f;
  edW_.resize(eW_.rows(), eW_.cols());
  edW_.fill(0);
  edb_.resize(eb_.size());
  edb_.fill(0);
}

void FeedForwardLayer::backward(BufferT& output, BufferT& error, std::vector<unsigned> const& mask) {
    nonlinear_backward(output, error, mask);
    // dENL = dE * dNL
    float num_features = std::accumulate(mask.cbegin(), mask.cend(), 0);
    // dW = input * dENL
    #pragma omp parallel for
    for(size_t j = 0; j < output.size(); j++) {
        edW_ += input_buffer_.at(j).transpose() * error.at(j);
    }
    edW_.array() /= num_features;
    for(auto& out: error)
        edb_ += out.colwise().sum();
    edb_.array() /= num_features;

    // error = dENL * W (dENL * dL)
    if (input_error_needed_) {
        #pragma omp parallel for
        for(size_t i = 0; i < mask.size(); i++)
            error_buffer_.at(i) = error.at(i) * eW_.transpose();
    }
    dW_ = edW_;
    for (size_t i = 0; i < db_.shape()[0]; i++)
        db_.at(i) = edb_(i);
}

void FeedForwardLayer::nonlinear_backward(BufferT& output, BufferT& error, std::vector<unsigned> const& mask) {
    // Out = output; dE = error
    auto dsigmoid = [](const float& v) -> float { return v * (1 - v); };
    auto drelu = [](const float& v) -> float { return v > 0 ? 1 : 0; };
    auto dtanh = [](const float& v) -> float { return 1 - std::pow(v, 2); };
    switch(nonlinearity_) {
        case Nonlinearity::None:
            // dENL = dE
            break;
        case Nonlinearity::ReLU:
            // dENL = dE * dOut(out <= 0 -> 0 , out > 0 -> 1)
            for(size_t i = 0; i< output.size(); i++) {
                error.at(i) = error.at(i).cwiseProduct(output.at(i).unaryExpr(drelu));
            }
            break;
        case Nonlinearity::Sigmoid:
            // dENL = dE * dOut(out* (1-out))
            for(size_t i = 0; i< output.size(); i++) {
                error.at(i) = error.at(i).cwiseProduct(output.at(i).unaryExpr(dsigmoid));
            }
            break;
        case Nonlinearity::Tanh:
            // dENL = dE * dOut(1-tanh^2(out))
            for(size_t i = 0; i< output.size(); i++) {
                error.at(i) = error.at(i).cwiseProduct(output.at(i).unaryExpr(dtanh));
            }
            break;
        default:
            throw std::invalid_argument("No type");
    }
}

FeedForwardLayer::FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity)
                                  : NetworkLayer(config), nonlinearity_(nonlinearity) {
}

