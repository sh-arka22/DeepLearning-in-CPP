#include<iostream>
#include<cmath>
#include<unsupported/Eigen/CXX11/Tensor>

float sigmoid(float x) {
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

// Method 2: Using manual loop (for educational purposes)
template<typename T, int _RANK>
auto sigmoid_activation_manual(Eigen::Tensor<T, _RANK> &input) {
    Eigen::Tensor<T, _RANK> output = input;
    
    // Note: Eigen tensors don't support range-based for loops directly
    // We need to iterate using indices
    auto dimensions = output.dimensions();
    int total_size = 1;
    for (int i = 0; i < _RANK; ++i) {
        total_size *= dimensions[i];
    }
    
    // Access elements using data() pointer
    T* data = output.data();
    for (int i = 0; i < total_size; ++i) {
        data[i] = sigmoid(data[i]);
    }
    
    return output;
}

// Use the fast method as the main function
template<typename T, int _RANK>
auto sigmoid_activation(Eigen::Tensor<T, _RANK> &input) {
    return sigmoid_activation_fast(input);
}

int main() {
    Eigen::Tensor<float, 2> input(2, 3);
    input.setValues({{0.0f, 1.0f, -1.0f}, {2.0f, -2.0f, 45.0f}});
    
    std::cout << "Input tensor:\n" << input << std::endl;
    std::cout << "\n";
    
    // Test the fast method
    auto output_fast = sigmoid_activation_fast(input);
    std::cout << "Sigmoid Activation Output (fast method):\n" << output_fast << std::endl;
    std::cout << "\n";
    
    // Test the manual method
    auto output_manual = sigmoid_activation_manual(input);
    std::cout << "Sigmoid Activation Output (manual method):\n" << output_manual << std::endl;
    std::cout << "\n";
    
    // Test the main function
    auto output = sigmoid_activation(input);
    std::cout << "Sigmoid Activation Output (main function):\n" << output << std::endl;
    
    return 0;
}