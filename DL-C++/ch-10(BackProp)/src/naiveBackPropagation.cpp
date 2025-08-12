#include <unsupported/Eigen/CXX11/Tensor>
template <typename T>


// forward pass
template <typename T>
auto forward(const Tensor<T, 2> &X, const Tensor<T, 2> &W0, const Tensor<T, 2> &W1, const Tensor<T, 2> &W2){

    const array<IndexPair<T>, 1> contract_dims = {IndexPair<T>(1, 0)};
    

    // first hidden layer
    RELU relu;
    Tensor<T, 2> Z0 = X.contract(W0, contract_dims); // X.W0
    Tensor<T, 2> Y0 = relu.evaluate(Z0); // ReLU(X.W0)

    // second hidden layer
    Tensor<T, 2> Z1 = Y0.contract(W1, contract_dims); // Y0.W1
    Tensor<T, 2> Y1 = relu.evaluate(Z1); // ReLU(Y0.W1)

    // output layer
    Tensor<T, 2> Z2 = Y1.contract(W2, contract_dims); // Y1.W2
    Tensor<T, 2> Y2 = softmax.evaluate(Z2); // softmax(Y1.W2)

    return std::make_tuple(Z0, Z1, Z2, Y0, Y1, Y2);
}

auto gradient(const Tensor<T, 2> &dc_dy, cont Tensor<T, 2> &input, const Tensor<T, 2> &z, const Tensor<T, 2> &y, const Tensor<T, 2> &w, const Activation &activation, const bool propagate = true){
    const int batch_size = input.dimension(0);
    //calculating dy_dz
    Tensor<float, 3> dy_dz = activation.jacobian(z);

    //reshaping dc_dy to 3d to meet bmm
    const array<Index, 3> dc_dy_3d_dim = {batch_size, 1, Y.dimension(1)};
    Tensor<float, 3> dc_dy_3d = dc_dy.reshape(dc_dy_3d_dim);

    //calculating dc_dz using batched matrix multiplication
    Tensor<float, 3> dc_dz = batched_matrix_multiplication(dc_dy_3d, dy_dz);

    //calculating dc_dw
    const array<IndexPair<T>, 1> product_dims_0_0 = {IndexPair<T>(0, 0)};
    const array<Index, 2> dc_dw_dim = {w.dimension(0), w.dimension(1)};
    Tensor<T, 2> dc_dw = input.contract(dc_dz, product_dims_0_0).reshape(dc_dw_dim);
    Tensor<float, 2> grad = dc_dw/dc_dw.constant(batch_size);

    Tensor<flaot, 2> downstream;
    if(propagate){
        //flase only for the first hidden layer
        //calculating the error propagation dc_dy for the previous layer
        const array<IndexPair<T>, 2> error_propagation_dims = {batch_size, input.dimension(1)};
        const array<IndexPair<T>, 1> product_dims_2_1 = {IndexPair<T>(2, 1)};
        downstream = dc_dz.contract(dy_dz, error_propagation_dims).contract(product_dims_2_1);
    }
    return std::make_tuple(grad, downstream);

}

template <typename T>
auto backward(cosnt Tensor<T, 2> &TRUE, const Tensor<T, 2> &x, const Tensor<T, 2> &z0, const Tensor<T, 2> &z1, const Tensor<T, 2> &z2, const Tensor<T, 2> &y0, const Tensor<T, 2> &y1, const Tensor<T, 2> &y2, const Tensor<T, 2> &w0, const Tensor<T, 2> &w1, const Tensor<T, 2> &w2){
    
}















// simple four-layer neuron
T loop(const Tensor<T, 2>& &TRUE, const Tensor<T, 2>& &X, const Tensor<T, 2> &W0, const Tensor<T, 2> &W1, const Tensor<T, 2> &W2, const T learning_rate){
    
    //forward pass
    auto [Z0, Z1, Z2, Y0, Y1, Y2] = forward(X, W0, W1, W2);

    //Output Cost
    CategoricalCrossEntropy cost_fn;
    T LOSS = cost_fn.evaluate( TRUE, Y2);

    //backward pass
    auto [grad0, grad1, grad2] = backward(TRUE, X,Z0,Z1,Z2,Y0,Y1,Y2,W0,W1,W2);


    //UPDATE PASS
    update(W0, W1, W2, grad0, grad1, grad2, learning_rate);

    return LOSS;
}

