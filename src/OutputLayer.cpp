#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

using namespace linalg;

void OutputLayer::forward(std::shared_ptr<BaseT> output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  // softmax
    FeedForwardLayer::forward(output, slice, mask);
    Tensor output_tensor(output, slice);
    for(size_t i = 0; i < mask.size(); i++)
    {
        const Matrix seq_matrix = input_tensor_({0, mask.at(i)}, {i}, ALL).mat(); // {max_seq, feature_size}
        output_tensor({0, mask.at(i)}, i, ALL).mat() = seq_matrix.dot(W_).radd(b_);
        output_tensor({0, mask.at(i)}, i, ALL).mat().softmax();
    }
}