#include <unsupported/Eigen/CXX11/Tensor>

template <typename T>
auto convolution2D = [](const Eigen::Tensor<T, 3>& input, const Eigen::Tensor<T, 2>& kernel) -> Eigen::Tensor<T, 3> {
    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(0, 0);
    padding[2] = std::make_pair(0, 0);
    
    Eigen::Tensor<T, 3> padded = input.pad(padding);
    Eigen::array<int, 2> dims({1, 2});
    Eigen::Tensor<T, 3> result = padded.convolve(kernel, dims);
    return result;
};