#include <iostream>
#include <random>
#include "includes/fully_connected_layer.hpp"

// Change from double to float to match the Dense class
using Tensor_1D = Eigen::Tensor<float, 1>;
using Tensor_2D = Eigen::Tensor<float, 2>;
using Tensor_3D = Eigen::Tensor<float, 3>;

auto bias_initializer = [](const int size) {
    Tensor_1D result(size);
    result.setZero();
    return result;
};

int main(int, char**)
{
    srand((unsigned int) time(0));

    auto weight_initializer = [](const int rows, const int cols, float range = 0.05f) {
        Tensor_2D _random(rows, cols);
        _random.setRandom();
        Tensor_2D result = (_random - _random.constant(.5f)) * _random.constant(range);
        return result;
    };

    Dense layer1(weight_initializer(4, 6), bias_initializer(6), sigmoid_activation<2>);
    Dense layer2(weight_initializer(6, 4), bias_initializer(4), sigmoid_activation<2>);
    Dense output_layer(weight_initializer(4, 2), bias_initializer(2), sigmoid_activation<2>);

    auto model = [&](const Tensor_2D& X) {
        auto l_1 = layer1(X);
        auto l_2 = layer2(l_1);
        auto result = output_layer(l_2);
        return result;
    };

    Tensor_2D input(1, 4);
    input.setValues({{-1.5f, 0.4f, 2.1f, -1.2f}});  // Add 'f' suffix for float literals

    auto output = model(input);

    std::cout << "The output is\n\n" << output << "\n\n";

    std::cout << "layer1 size is " << layer1.size() << "\n";
    std::cout << "layer2 size is " << layer2.size() << "\n";
    std::cout << "output_layer size is " << output_layer.size() << "\n";

    return 0;
}

