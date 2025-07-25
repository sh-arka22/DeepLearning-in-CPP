#include <iostream>
#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>

double sigmoid(float x) {
    if( x >= 45.0f) return 1.0f; // Avoid overflow
    if( x <= -45.0f) return 0.0f; // Avoid underflow
    return 1.0f / (1.0f + std::exp(-x));
}

// Method 1: Using unaryExpr (more efficient)
template<typename T, int _RANK>
auto sigmoid_activation_fast(Eigen::Tensor<T, _RANK> &input) {
    auto result = input.unaryExpr(std::ref(sigmoid));
    return result;
}
using Tensor_1D = Eigen::Tensor<double, 1>;
using Tensor_2D = Eigen::Tensor<double, 2>;
using Tensor_3D = Eigen::Tensor<double, 3>;
  
Tensor_1D calc_layer(const Tensor_1D &input, const Tensor_2D &weights, const Tensor_1D &bias) {
    Eigen::array<Eigen::IndexPair<int>, 1> output_dims = {Eigen::IndexPair<int>(0, 0)};

    auto prod = input.contract(weights, output_dims);
    Tensor_1D result = prod + bias;
    auto result_sigmoid = sigmoid_activation_fast(result);
    return result_sigmoid;
}

int main(){
    Tensor_1D X(2);
    X.setValues({-1.5, 0.4});

    Tensor_2D W(2, 3);
    W.setValues({{1, 2, 3}, {4, 5, 6}});
    
    Tensor_1D B(3);
    B.setValues({-1, 1, 2});

    std::cout << "X:\n\n"
              << X << "\n\n";
    
    std::cout << "W:\n\n"
              << W << "\n\n";
    
    std::cout << "B:\n\n"
              << B << "\n\n";

    auto R = calc_layer(X, W, B);

    std::cout << "R:\n\n"
              << R << "\n\n";

    return 0;
}