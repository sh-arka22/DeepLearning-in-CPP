#include <string>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <cmath>
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

// Softmax activation function
template <int _RANK>
auto softmax(const Tensor<_RANK> &input) {
    // Find max for numerical stability
    auto max_val = input.maximum();
    
    // Subtract max and compute exp
    auto shifted = input - input.constant(max_val);
    auto exp_vals = shifted.exp();
    
    // Compute sum of exponentials
    auto sum_exp = exp_vals.sum();
    TYPE sum_val = ((Tensor_0D)sum_exp)(0);
    
    // Normalize
    auto result = exp_vals / exp_vals.constant(sum_val);
    return result;
}

// Categorical Cross-Entropy Loss
template <int _RANK>
auto categorical_cross_entropy(const Tensor<_RANK> &predictions, const Tensor<_RANK> &true_labels) {
    // Add small epsilon to prevent log(0)
    const TYPE epsilon = 1e-15f;
    auto clipped_preds = predictions.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);
    
    // Compute -sum(true_labels * log(predictions))
    auto log_preds = clipped_preds.log();
    auto cross_entropy = true_labels * log_preds;
    auto neg_sum = -cross_entropy.sum();
    
    TYPE loss = ((Tensor_0D)neg_sum)(0);
    
    // Average over batch size (assuming first dimension is batch)
    if (_RANK > 1) {
        auto dims = predictions.dimensions();
        int batch_size = dims[0];
        loss = loss / static_cast<TYPE>(batch_size);
    }
    
    return loss;
}

int main(int, char**)
{
    std::cout << std::fixed << std::setprecision(6);
    
    // Example 1: Binary classification (2 classes)
    std::cout << "=== Binary Classification Example ===" << std::endl;
    
    // True labels (one-hot encoded)
    Tensor_2D true_binary(2, 2);
    true_binary.setValues({{1.0f, 0.0f},   // Sample 1: Class 0
                          {0.0f, 1.0f}});  // Sample 2: Class 1
    
    // Good predictions (close to true labels)
    Tensor_2D good_pred_binary(2, 2);
    good_pred_binary.setValues({{0.9f, 0.1f},   // Close to [1,0]
                               {0.2f, 0.8f}});  // Close to [0,1]
    
    // Poor predictions (far from true labels)
    Tensor_2D poor_pred_binary(2, 2);
    poor_pred_binary.setValues({{0.4f, 0.6f},   // Far from [1,0]
                               {0.7f, 0.3f}});  // Far from [0,1]
    
    float good_loss_binary = categorical_cross_entropy(good_pred_binary, true_binary);
    float poor_loss_binary = categorical_cross_entropy(poor_pred_binary, true_binary);
    
    std::cout << "True labels:" << std::endl;
    std::cout << true_binary << std::endl << std::endl;
    
    std::cout << "Good predictions:" << std::endl;
    std::cout << good_pred_binary << std::endl;
    std::cout << "CCE Loss (good): " << good_loss_binary << std::endl << std::endl;
    
    std::cout << "Poor predictions:" << std::endl;
    std::cout << poor_pred_binary << std::endl;
    std::cout << "CCE Loss (poor): " << poor_loss_binary << std::endl;
    std::cout << "Loss ratio (poor/good): " << poor_loss_binary / good_loss_binary << std::endl << std::endl;
    
    // Example 2: Multi-class classification (3 classes)
    std::cout << "=== Multi-class Classification Example ===" << std::endl;
    
    // True labels for 3 samples, 3 classes
    Tensor_2D true_multi(3, 3);
    true_multi.setValues({{1.0f, 0.0f, 0.0f},   // Sample 1: Class 0
                         {0.0f, 1.0f, 0.0f},    // Sample 2: Class 1  
                         {0.0f, 0.0f, 1.0f}});  // Sample 3: Class 2
    
    // Raw logits (before softmax)
    Tensor_2D logits(3, 3);
    logits.setValues({{2.0f, 1.0f, 0.1f},     // Should predict class 0
                     {0.5f, 3.0f, 0.2f},      // Should predict class 1
                     {0.1f, 0.3f, 2.5f}});    // Should predict class 2
    
    // Apply softmax to get probabilities
    auto predictions_multi = softmax(logits);
    
    float loss_multi = categorical_cross_entropy(predictions_multi, true_multi);
    
    std::cout << "True labels:" << std::endl;
    std::cout << true_multi << std::endl << std::endl;
    
    std::cout << "Raw logits:" << std::endl;
    std::cout << logits << std::endl << std::endl;
    
    std::cout << "Softmax probabilities:" << std::endl;
    std::cout << predictions_multi << std::endl;
    std::cout << "CCE Loss: " << loss_multi << std::endl << std::endl;
    
    // Example 3: Effect of confidence
    std::cout << "=== Confidence Effect Example ===" << std::endl;
    
    Tensor_2D true_conf(1, 3);
    true_conf.setValues({{0.0f, 1.0f, 0.0f}});  // True class is 1
    
    // High confidence correct prediction
    Tensor_2D high_conf(1, 3);
    high_conf.setValues({{0.05f, 0.9f, 0.05f}});
    
    // Low confidence correct prediction
    Tensor_2D low_conf(1, 3);
    low_conf.setValues({{0.3f, 0.4f, 0.3f}});
    
    // Wrong prediction with high confidence
    Tensor_2D wrong_conf(1, 3);
    wrong_conf.setValues({{0.9f, 0.05f, 0.05f}});
    
    float loss_high = categorical_cross_entropy(high_conf, true_conf);
    float loss_low = categorical_cross_entropy(low_conf, true_conf);
    float loss_wrong = categorical_cross_entropy(wrong_conf, true_conf);
    
    std::cout << "True label: [0, 1, 0]" << std::endl;
    std::cout << "High confidence correct [0.05, 0.9, 0.05]: " << loss_high << std::endl;
    std::cout << "Low confidence correct [0.3, 0.4, 0.3]: " << loss_low << std::endl;
    std::cout << "High confidence wrong [0.9, 0.05, 0.05]: " << loss_wrong << std::endl;
    
    std::cout << "\nObservations:" << std::endl;
    std::cout << "- Higher confidence in correct prediction = lower loss" << std::endl;
    std::cout << "- Higher confidence in wrong prediction = higher loss" << std::endl;
    std::cout << "- CCE heavily penalizes confident wrong predictions" << std::endl;

    return 0;
}