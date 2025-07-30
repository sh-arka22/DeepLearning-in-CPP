#include <string>
#include <exception>
#include <iomanip>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using TYPE = float;

using Tensor_0D = Eigen::Tensor<TYPE, 0>;
using Tensor_1D = Eigen::Tensor<TYPE, 1>;
using Tensor_2D = Eigen::Tensor<TYPE, 2>;
using Tensor_3D = Eigen::Tensor<TYPE, 3>;
using Tensor_4D = Eigen::Tensor<TYPE, 4>;

template<int _RANK>
using Tensor = Eigen::Tensor<TYPE, _RANK>;

template<int _RANK>
using DimArray = Eigen::array<Eigen::DenseIndex, _RANK>;

Eigen::Tensor<float, 2> softmax_2D(const Eigen::Tensor<float, 2> &z){

    auto dimensions = z.dimensions();

    int batch_size = dimensions.at(0);
    int instances_size = dimensions.at(1);

/*
Input:  [[2.0, 1.0, 3.0, 0.5],     z_max: [3.0,   <- max of row 0
         [1.5, 4.0, 2.0, 1.0],              4.0,   <- max of row 1
         [0.5, 1.0, 2.5, 3.0]]              3.0]   <- max of row 2
*/
    // Getting the maximum for each instance.
    // Note that this operation reduces 1 dimension
    Eigen::array<int,1> depth_dim({1}); // Reduce along dimension 1
    auto z_max = z.maximum(depth_dim); // Shape: [3] (one max per sample)

// broadcasting
/*
max_reshaped:  [[3.0],        max_values:  [[3.0, 3.0, 3.0, 3.0],
                [4.0],                      [4.0, 4.0, 4.0, 4.0],
                [3.0]]                      [3.0, 3.0, 3.0, 3.0]]
*/
    // Getting the max array as an 2-rank tensor
    auto max_reshaped = z_max.reshape(Eigen::array<Eigen::DenseIndex, 2>({batch_size, 1}));  // Shape: [3, 1]
    auto max_values = max_reshaped.broadcast(Eigen::array<Eigen::DenseIndex, 2>({1, instances_size})); // Shape: [3, 4]

/*
Original:     [[2.0, 1.0, 3.0, 0.5],     Normalized:  [[-1.0, -2.0,  0.0, -2.5],
               [1.5, 4.0, 2.0, 1.0],                  [-2.5,  0.0, -2.0, -3.0],
               [0.5, 1.0, 2.5, 3.0]]                  [-2.5, -2.0, -0.5,  0.0]]
*/
    // Normalizing the input
    auto normalized = z - max_values;

    /* INPUT
    [
        [-1.0, -2.0,  0.0, -2.5],
        [-2.5,  0.0, -2.0, -3.0],
        [-2.5, -2.0, -0.5,  0.0]
    ]
    */

    /* OUTPUT
    [
        [0.368, 0.135, 1.000, 0.082],    ← e^(-1.0), e^(-2.0), e^(0.0), e^(-2.5)
        [0.082, 1.000, 0.135, 0.050],    ← e^(-2.5), e^(0.0), e^(-2.0), e^(-3.0)
        [0.082, 0.135, 0.607, 1.000]]    ← e^(-2.5), e^(-2.0), e^(-0.5), e^(0.0)
    */
    auto expo = normalized.exp();

    /* INPUT
    [
        [0.368, 0.135, 1.000, 0.082],     expo_sums:  [1.585]  ← 0.368+0.135+1.000+0.082
        [0.082, 1.000, 0.135, 0.050],                 [1.267]  ← 0.082+1.000+0.135+0.050
        [0.082, 0.135, 0.607, 1.000]]                 [1.824]  ← 0.082+0.135+0.607+1.000
    */
    // FIX: Use proper Eigen::array type for sum operation
    auto expo_sums = expo.sum(Eigen::array<int, 1>({1}));

    /* INPUT
    expo_sums: [1.585, 1.267, 1.824]     sums_reshaped:  [[1.585],
                                                           [1.267],
                                                           [1.824]]
    */
    // FIX: Use proper Eigen::array type for reshape operation
    auto sums_reshaped = expo_sums.reshape(Eigen::array<Eigen::DenseIndex, 2>({batch_size, 1}));

    /* INPUT
    sums_reshaped:  [[1.585],     sums:  [[1.585, 1.585, 1.585, 1.585],
                     [1.267],             [1.267, 1.267, 1.267, 1.267],
                     [1.824]]             [1.824, 1.824, 1.824, 1.824]]
    */
    // FIX: Use proper Eigen::array type for broadcast operation
    auto sums = sums_reshaped.broadcast(Eigen::array<Eigen::DenseIndex, 2>({1, instances_size}));

    /* INPUT
    expo:           [[0.368, 0.135, 1.000, 0.082],     result:  [[0.232, 0.085, 0.631, 0.052],
                     [0.082, 1.000, 0.135, 0.050],              [0.065, 0.788, 0.107, 0.039],
                     [0.082, 0.135, 0.607, 1.000]]              [0.045, 0.074, 0.333, 0.549]]
    */
    Tensor_2D result = expo / sums;

    return result;
}

int main(int, char**) {
    // Create batch of logits (3 samples, 4 classes each)
    Tensor_2D logits(3, 4);
    logits.setValues({
        {2.0f, 1.0f, 3.0f, 0.5f},   // Sample 0
        {1.5f, 4.0f, 2.0f, 1.0f},   // Sample 1
        {0.5f, 1.0f, 2.5f, 3.0f}    // Sample 2
    });

    auto probabilities = softmax_2D(logits);

    std::cout << "=== Batch Softmax Demo ===" << std::endl;
    std::cout << "Input logits (3 samples, 4 classes):" << std::endl;
    
    for (int i = 0; i < 3; i++) {
        std::cout << "Sample " << i << ": [";
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << logits(i, j);
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nOutput probabilities:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "Sample " << i << ": [";
        float sum = 0.0f;
        for (int j = 0; j < 4; j++) {
            float prob = probabilities(i, j);
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << prob;
            if (j < 3) std::cout << ", ";
            sum += prob;
        }
        std::cout << "] (sum: " << std::fixed << std::setprecision(6) << sum << ")" << std::endl;
    }

    // Add analysis section
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "Batch processing benefits:" << std::endl;
    std::cout << "- Processes all " << logits.dimension(0) << " samples simultaneously" << std::endl;
    std::cout << "- Uses vectorized operations for efficiency" << std::endl;
    std::cout << "- Broadcasting avoids explicit loops" << std::endl;
    std::cout << "- Numerical stability through max subtraction" << std::endl;
    
    // Show which class has highest probability for each sample
    std::cout << "\nPredicted classes:" << std::endl;
    for (int i = 0; i < 3; i++) {
        float max_prob = 0.0f;
        int max_idx = 0;
        for (int j = 0; j < 4; j++) {
            if (probabilities(i, j) > max_prob) {
                max_prob = probabilities(i, j);
                max_idx = j;
            }
        }
        std::cout << "Sample " << i << ": Class " << max_idx 
                  << " (probability: " << std::fixed << std::setprecision(4) << max_prob << ")" << std::endl;
    }

    return 0;
}