/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __NEURAL_NETWORK_HPP__
#define __NEURAL_NETWORK_HPP__

#include <map>
#include <memory>
#include <valarray>
#include <vector>

#include "Config.hpp"
#include "FeatureScorer.hpp"
#include "NetworkLayer.hpp"
#include "OutputLayer.hpp"
#include <fstream>

struct FeatureBuffer {
  std::valarray<float>& output;
  std::gslice const& slice;

  FeatureBuffer(std::valarray<float>& output, std::gslice const& slice)
               : output(output), slice(slice) {
  }

  ~FeatureBuffer() {
  }
};

struct OutputBuffer {
  std::shared_ptr<std::valarray<float>> fwd_buffer;
  std::shared_ptr<std::valarray<float>> bwd_buffer;
  const std::gslice     slice;

  OutputBuffer(std::shared_ptr<std::valarray<float>> fwd_buffer, std::shared_ptr<std::valarray<float>> bwd_buffer, std::gslice const& slice)
              : fwd_buffer(fwd_buffer), bwd_buffer(bwd_buffer), slice(slice) {
  }

  ~OutputBuffer() {
  }
};

class NeuralNetwork : public FeatureScorer {
public:
  static const ParameterString paramLoadNeuralNetworkFrom;
  static const ParameterString paramPriorFile;
  static const ParameterFloat  paramPriorScale;
  static const ParameterUInt   paramContextFrames;
  static const ParameterString  paramNNOutFile;

  NeuralNetwork(Configuration const& config, size_t feature_size, size_t batch_size, size_t max_seq_length, size_t num_classes);
  ~NeuralNetwork();

  virtual void   prepare_sequence(FeatureIter const& start, FeatureIter const& end);
  virtual double score           (FeatureIter const& iter, StateIdx state_idx) const;

  std::valarray<float>&       get_feature_buffer();
  std::gslice                 get_feature_buffer_slice() const;
  std::vector<unsigned>&      get_batch_mask();
  std::shared_ptr<std::valarray<float>> const get_score_buffer() const;
  NetworkLayer*               get_network_layer(std::string const& name);

  std::map<std::string, std::shared_ptr<std::valarray<float>>> const& get_parameters() const;
  std::map<std::string, std::shared_ptr<std::valarray<float>>> const& get_gradients()  const;

  void init_parameters(std::function<float()> const& generator);
  void forward();
  void forward_visualize();
  void backward(std::valarray<float> const& targets);
  void save(std::string const& folder) const;
  void load(std::string const& folder);
  void load_prior(); // only needed when the NN is used as a feature-scorer
private:
  const size_t feature_size_;
  const size_t batch_size_;
  const size_t max_seq_length_;
  const size_t num_classes_;
  const size_t context_frames_; // only needed if the NN is used as a feature-scorer

  bool initialized_;

  std::vector<std::unique_ptr<NetworkLayer>>   layers_;
  std::vector<std::vector<OutputBuffer>>       output_infos_;
  std::map<std::string, std::shared_ptr<std::valarray<float>>> parameters_;
  std::map<std::string, std::shared_ptr<std::valarray<float>>> gradients_;

  std::valarray<float>  feature_buffer_; // feature_size * batch_size * max_seq_length
  FeatureIter           feature_buffer_start_;
  std::vector<unsigned> batch_mask_;

  std::shared_ptr<std::valarray<float>> score_buffer_;
  std::shared_ptr<std::valarray<float>> error_buffer_;

  const std::string    prior_path_;
  const float          prior_scale_;
  std::valarray<float> log_prior_;

  void load_prior(std::string const& path);
  std::ofstream out_file;
};

#endif /* __NEURAL_NETWORK_HPP__ */
