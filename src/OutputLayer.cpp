#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

using namespace linalg;

void OutputLayer::forward(std::shared_ptr<BaseT> output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  // softmax
    Tensor output_tensor(output, slice);
    for(size_t i = 0; i < mask.size(); i++)
    {
        const auto Wsize = feature_size_ * output_size_;
        const Matrix W(params_, {feature_size_, output_size_});
        const Vector b(params_, {output_size_}, Wsize);

        const Matrix seq_matrix = input_tensor_({0, mask.at(i)}, {i}, ALL); // {max_seq, feature_size}
        static_cast<Matrix>(output_tensor({0, mask.at(i)}, i, ALL)) = seq_matrix.dot(W) + b;
        static_cast<Matrix>(output_tensor({0, mask.at(i)}, i, ALL)).softmax();
    }
}

