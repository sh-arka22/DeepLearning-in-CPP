#include<iostream>
#include<cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include<iomanip>  // For std::setprecision


Eigen::Tensor<float, 1> softmax(const Eigen::Tensor<float, 1>& z) {
    const Eigen::Tensor<float, 0> m = z.maximum();

    Eigen::Tensor<float, 1> normalised = z - z.constant(m(0));
    Eigen::Tensor<float, 1> exp_vals = normalised.exp();

    const Eigen::Tensor<float, 0> sum_tensor = exp_vals.sum();
    const float sum_exp = sum_tensor(0);
    Eigen::Tensor<float, 1> result = exp_vals / exp_vals.constant(sum_exp);
    return result;
}

int main(int, char **)
{
    Eigen::Tensor<float, 2> input(8, 3);
    input.setValues({
        {0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.},
        {100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}
    });

    const int batch_size = input.dimension(0);
    const int output_size = input.dimension(1);

    for (int i = 0; i < batch_size; i++) {
        Eigen::Tensor<float, 1> row = input.chip(i, 0);

        Eigen::Tensor<float, 1> output = softmax(row); // Changed Tensor_1D to Eigen::Tensor<float, 1>
        

        Eigen::Tensor<float, 2> output_matrix(batch_size, output_size);
        output_matrix.chip(i, 0) = output;

        std::cout << "softmax([" << row << "]): [" << output << "]\n\n";
    }

    return 0;
}
