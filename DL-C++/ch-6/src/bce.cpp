#include <iostream>
#include <string>
#include <exception>
#include <iomanip>

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

template <int _RANK>
TYPE bce(const Tensor<_RANK> &PRED, const Tensor<_RANK> &TRUE) {
    auto COMP_TRUE = TRUE.constant(1.) - TRUE;
    auto COMP_PRED = PRED.constant(1.) - PRED;
    auto part1 = TRUE * (PRED + PRED.constant(1e-7)).log();
    auto part2 = COMP_TRUE * (COMP_PRED + COMP_PRED.constant(1e-7)).log();
    auto parts = part1 + part2;
    float sum = ((Tensor_0D)(parts.sum()))(0);
    float result = -sum / PRED.size();
    return result;
}