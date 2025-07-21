#include <iostream>
#include "utils/Eigen/Core"


using Matrix = Eigen::MatrixXd;

int main(){
    Matrix kernel(3, 3);

    kernel << 
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1;

    std::cout << "Kernel:\n" << kernel << "\n\n";

    Matrix input(6, 6);
    input << 3, 1, 0, 2, 5, 6,
        4, 2, 1, 1, 4, 7,
        5, 4, 0, 0, 1, 2,
        1, 2, 2, 1, 3, 4,
        6, 3, 1, 0, 5, 2,
        3, 1, 0, 1, 3, 3;

    std::cout << "Input:\n" << input << "\n\n";


    auto Conv_2D = [](const Matrix& input, const Matrix& kernel) {
        int kRows = kernel.rows();
        int kCols = kernel.cols();
        int iRows = input.rows();
        int iCols = input.cols();
        Matrix output(iRows - kRows + 1, iCols - kCols + 1);

        for (int i = 0; i <= iRows - kRows; ++i) {
            for (int j = 0; j <= iCols - kCols; ++j) {
                output(i, j) = (input.block(i, j, kRows, kCols).array() * kernel.array()).sum();
            }
        }
        return output;
    };

    auto output = Conv_2D(input, kernel);
    std::cout << "Convolution:\n" << output << "\n";

    return 0;
}