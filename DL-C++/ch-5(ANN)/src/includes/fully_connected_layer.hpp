#ifndef __MY_FC_LAYERS__
#define __MY_FC_LAYERS__

#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <functional>
#include <stdexcept>

using Scalar = float;
template <int Rank> using Tensor = Eigen::Tensor<Scalar, Rank>;
using Tensor1D = Tensor<1>;
using Tensor2D = Tensor<2>;
template <int Rank> using DSizes = Eigen::DSizes<Eigen::Index, Rank>;

template <int Rank>
Tensor<Rank> sigmoid_activation(const Tensor<Rank>& Z) {
    auto sigmoid = [](Scalar z) {
        if (z >= 45.f) return 1.f;
        if (z <= -45.f) return 0.f;
        return 1.f / (1.f + std::exp(-z));
    };
    return Z.unaryExpr(sigmoid);
}

class Dense {
public:
    Dense(Tensor2D weights_, Tensor1D bias_, std::function<Tensor2D(const Tensor2D&)> activation_)
        : weights(std::move(weights_)), bias(std::move(bias_)), activation(activation_) {}

    Tensor2D operator()(const Tensor2D& input) {
        auto input_dims = input.dimensions();
        auto weight_dims = weights.dimensions();
        int batch = input_dims[0];
        int in_size = input_dims[1];
        int out_size = weight_dims[1];

        if (in_size != weight_dims[0]) throw std::invalid_argument("Input size mismatch");
        if (bias.dimension(0) != out_size) throw std::invalid_argument("Bias size mismatch");

        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
        Tensor2D prod = input.contract(weights, contract_dims);

        DSizes<2> bias_shape{1, out_size};
        auto bias_reshaped = bias.reshape(bias_shape);
        DSizes<2> bcast{batch, 1};
        Tensor2D bias_bcast = bias_reshaped.broadcast(bcast);

        Tensor2D Z = prod + bias_bcast;
        return activation(Z);
    }

    int size() { return bias.size() + weights.size(); }

private:
    Tensor2D weights;
    Tensor1D bias;
    std::function<Tensor2D(const Tensor2D&)> activation;
};

template <int Rank>
Tensor2D flatten(const Tensor<Rank>& input) {
    auto dims = input.dimensions();
    int batch = dims[0];
    int flat_size = input.size() / batch;
    DSizes<2> new_shape{batch, flat_size};
    return input.reshape(new_shape);
}

#endif