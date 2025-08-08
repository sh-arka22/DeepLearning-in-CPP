const Eigen::Tensor<T, 2> pooling(const Eigen::Tensor<T, 2>&x, int pool_size, int stride = 1){
    
    //reshaping the input to a 4-Rank shape (depth, rows, columns, batch)
    Eigen::array<Eigen::DenseIndex, 4> reshaped_dims{{1,x.dimension(0),x.dimension(1),1}};
    Eigen::Tensor<T, 4> reshaped_input = x.reshape(reshaped_dims);

    //getting the patches
    Eigen::Tensor<T, 5> patches =
    reshaped_input.extract_image_patches(
        pool_size, pool_size,   // kernel height & width
        stride, stride,         // stride height & width
        Eigen::PADDING_VALID    // no zero-padding
    );
    
    //Getting the max of each patch (reduce over kernel dimensions)
    Eigen::array<Eigen::DenseIndex, 2> dims({1, 2});
    Eigen::Tensor<T, 3> maxed_patches = patches.maximum(dims);
    
    //Reshape back to 2D by removing depth and batch dimensions
    //Reshape back to 2D using correct pooling output dimensions
    int output_rows = (x.dimension(0) - pool_size) / stride + 1;
    int output_cols = (x.dimension(1) - pool_size) / stride + 1;
    Eigen::array<Eigen::DenseIndex, 2> output_dims{{output_rows, output_cols}};
    auto result = maxed_patches.reshape(output_dims);

    return result;
}

/*


    Start:      x                          (R, C)

    Reshape:    reshaped_input              (1, R, C, 1)             // (C, H, W, N)

    Patches:    patches                     (1, pool, pool, P, 1)    // P = out_rows*out_cols

    Reduce:     maxed_patches               (1, P, 1)

    Reshape:    result                      (out_rows, out_cols)


*/


*