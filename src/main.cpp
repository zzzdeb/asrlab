/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Config.hpp"
#include "Corpus.hpp"
#include "IO.hpp"
#include "NeuralNetwork.hpp"
#include "NNTraining.hpp"
#include "Recognizer.hpp"
#include "SignalAnalysis.hpp"
#include "Training.hpp"

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config-file>" << std::endl;
    return EXIT_FAILURE;
  }

  const Configuration config(argv[1]);

  const ParameterString paramAction           ("action", "");
  const ParameterString paramFeaturePath      ("feature-path", "");
  const ParameterString paramNormalizationPath("normalization-path", "");
  const ParameterBool   paramMaxApprox        ("max-approx", true);
  const ParameterString paramFeatureScorer    ("feature-scorer", "gmm");
  const ParameterString paramAlignmentPath    ("alignment-path", "");
  const ParameterString paramLexiconPath      ("lexicon-path", "");

  std::string action(paramAction(config));
  std::string feature_path(paramFeaturePath(config));
  std::string normalization_path(paramNormalizationPath(config));
  bool        max_approx(paramMaxApprox(config));

  if (argc >= 3) {
    action = std::string(argv[2]);
  }

  std::string lexicon_path = paramLexiconPath(config);
  if (lexicon_path.empty()) {
    std::cerr << "Error: Must provide lexicon path" << std::endl;
    return EXIT_FAILURE;
  }

  const Configuration lex_config(lexicon_path);
  std::shared_ptr<Lexicon> lexicon = build_lexicon_from_config(lex_config);

  CorpusDescription corpus_description(config);
  corpus_description.read(lexicon);
  SignalAnalysis analyzer(config);

  if (action != "extract-features") {
    if (normalization_path.size() > 0) {
      std::ifstream normalization_stream(normalization_path.c_str(), std::ios_base::in);
      if (not normalization_stream.good()) {
        std::cerr << "Error: could not open normalization file" << std::endl;
        exit(EXIT_FAILURE);
      }
      analyzer.read_normalization_file(normalization_stream);
    }
  }
/*****************************************************************************/
  if (action == "extract-features") {
    const ParameterString paramAudioPath  ("audio-path",   "");
    const ParameterString paramAudioFormat("audio-format", "sph");

    std::string audio_path  (paramAudioPath  (config));
    std::string audio_format(paramAudioFormat(config));

    /* proceed over training/test samples and perform signal analysis */
    size_t i = 0ul;
    for (auto seg_iter = corpus_description.begin(); seg_iter != corpus_description.end(); ++seg_iter) {
      i++;
      std::cerr << "Processing (" << i << "): " << seg_iter->name << std::endl;
      analyzer.process(audio_path   + seg_iter->name + std::string(".") + audio_format,
                       feature_path + seg_iter->name + std::string(".mm2"));
    }
    if (normalization_path.size() > 0) {
      std::ofstream normalization_stream(normalization_path.c_str(), std::ios_base::out | std::ios_base::trunc);
      if (not normalization_stream.good()) {
        std::cerr << "Error: could not open normalization file" << std::endl;
        exit(EXIT_FAILURE);
      }
      analyzer.compute_normalization();
      analyzer.write_normalization_file(normalization_stream);
    }
  }
/*****************************************************************************/
  else if (action == "train" or action == "recognize" or action == "pruning-stat") {
    Corpus corpus;
    corpus.read(corpus_description, feature_path, analyzer);

    TdpModel tdp_model(config, lexicon->get_silence_automaton()[0ul]);

    if (action == "train") {
      MixtureModel mixtures(config, analyzer.n_features_total, lexicon->num_states(), MixtureModel::MIXTURE_POOLING, max_approx);

      Trainer trainer(config, lexicon, mixtures, tdp_model, max_approx);
      trainer.train(corpus);
    }
    else if (action == "pruning-stat") {
      std::vector<double> prunings(50);
      for (size_t i = 0; i < prunings.size(); i++) {
        prunings.at(i) = 10 * (i+1);
      
        MixtureModel mixtures(config, analyzer.n_features_total, lexicon->num_states(), MixtureModel::MIXTURE_POOLING, max_approx);

        Trainer trainer(config, lexicon, mixtures, tdp_model, max_approx);
        trainer.set_pthreshold(prunings.at(i));
        trainer.train(corpus);
      }
    }
    else { // action == "recognize"
      std::string feature_scorer = paramFeatureScorer(config);
      FeatureScorer* fs = nullptr;

      if (feature_scorer == "gmm") {
        fs = new MixtureModel(config, analyzer.n_features_total, lexicon->num_states(), MixtureModel::MIXTURE_POOLING, max_approx);
      }
      else if (feature_scorer == "nn") {
        const ParameterUInt paramContextFrames("context-frames", 0);
        size_t context_frames = paramContextFrames(config);
        fs = new NeuralNetwork(config, analyzer.n_features_total * (2ul * context_frames + 1ul), 1ul, corpus.get_max_seq_length(), lexicon->num_states());
        dynamic_cast<NeuralNetwork*>(fs)->load_prior();
      }
      else {
        std::cerr << "unknown feature scorer: " << feature_scorer << std::endl;
        exit(EXIT_FAILURE);
      }

      if (true) {
        Recognizer recognizer(config, lexicon, *fs, tdp_model);
        recognizer.recognize(corpus);
      } else { // many threshold
        std::vector<double> thresholds{1000000, 500, 250, 100, 25};
        for (const auto& threshold : thresholds) {
          Recognizer recognizer(config, lexicon, *fs, tdp_model);
          recognizer.threshold() = threshold;
          recognizer.recognize(corpus);
          std::cout << "AM_THRESHOLD = " << threshold << std::endl << std::endl;
        }
      }
    }
  }
/*****************************************************************************/
  else if (action == "train-nn" or action == "compute-prior" or action == "visualize-nn") {
    const ParameterUInt paramBatchSize("batch-size", 35u);
    const unsigned batch_size(paramBatchSize(config));

    Corpus corpus;
    corpus.read(corpus_description, feature_path, analyzer);

    MiniBatchBuilder mini_batch_builder(config, corpus, batch_size, lexicon->num_states(), lexicon->get_silence_automaton()[0]);
    NeuralNetwork    nn(config, mini_batch_builder.feature_size(), batch_size, corpus.get_max_seq_length(), lexicon->num_states());

    if (action == "train-nn" or action == "visualize-nn") {
      NnTrainer nn_trainer(config, mini_batch_builder, nn);
      nn_trainer.train();
      if (action == "visualize-nn") {}
        nn.forward_visualize();
    } 
    else { // action == "compute-prior"
      const ParameterString paramPriorFile("prior-file", "");
      const std::string prior_file = paramPriorFile(config);

      std::ofstream out(prior_file, std::ios::out | std::ios::trunc);
      if (not out.good()) {
        std::cerr << "Could not open prior-file: " << prior_file << std::endl;
        std::abort();
      }
      const size_t num_classes = lexicon->num_states();
      std::valarray<float> prior(num_classes);
      std::valarray<float> targets;

      for (size_t batch = 0ul; batch < mini_batch_builder.num_train_batches(); batch++) {
          nn.get_feature_buffer() = 0.0f;
          mini_batch_builder.build_batch(batch, false,
                                         nn.get_feature_buffer(),
                                         nn.get_feature_buffer_slice(),
                                         targets,
                                         nn.get_batch_mask());
          nn.forward();
          const std::valarray<float>& score_buffer = *nn.get_score_buffer();
          for (size_t i = 0; i < nn.get_batch_mask().size(); i++)
            for (size_t j = 0; j < nn.get_batch_mask()[i]; j++) {
                prior += score_buffer[std::slice((i * mini_batch_builder.max_seq_length() + j) * num_classes, num_classes, 1)];
          }
      }
      prior /= prior.sum(); // todo(ze) some priors are zero.
      for (size_t i = 0; i < prior.size(); i++)
          out << prior[i] << " ";
    }
  }
/*****************************************************************************/
  else {
    std::cerr << "Error: unknown action " << action << std::endl;
    exit(EXIT_FAILURE);
  }
/*****************************************************************************/

  return EXIT_SUCCESS;
}

