#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

using namespace linalg;

void OutputLayer::forward(BufferT& outputs_, std::vector<unsigned> const& mask) const {
  // softmax
    FeedForwardLayer::forward(outputs_, mask);
    #pragma omp parallel for
    for(size_t i = 0; i < outputs_.size(); i++)
    {
        if (outputs_.at(i).rows() == 0)
            continue;
        auto& out = outputs_.at(i);
        out.colwise()-=out.rowwise().maxCoeff();
        out = out.array().exp();
        Vect norm = out.rowwise().sum();
        out.array().colwise() /= norm.array();
    }
}