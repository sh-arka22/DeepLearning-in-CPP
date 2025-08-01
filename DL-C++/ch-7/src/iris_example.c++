#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<string>
#include<random>
#include<algorithm>
#include<stdexcept>
#include<tuple>
#include<unsupported/Eigen/CXX11/Tensor>


auto load_iris_dataset = [](std::string file_path, bool shuffle = true, float split_percentage = .8) {
    std::ifstream file;
    file.open(file_path);
    const int N_REGISTERS = 150;

    if (!file.is_open())
        throw std::invalid_argument("File " + file_path + " not found.");

    std::vector<std::string> lines;
    lines.reserve(N_REGISTERS);
    std::string line;

    while (getline(file, line)) {
        lines.push_back(line);
    }

    if (shuffle) {
        auto rd = std::random_device {}; 
        auto rng = std::default_random_engine { rd() };
        std::shuffle(lines.begin(), lines.end(), rng);
    }

    std::vector<float> data;
    const int EXPECTED_SIZE = N_REGISTERS * (4 + 3); // 4 attributes + 3 hot-encoding for the 3 classes
    data.reserve(EXPECTED_SIZE);
    std::string element;
    const std::string class_setosa = "Iris-setosa";
    const std::string class_versicolor = "Iris-versicolor";
    const std::string class_virginica = "Iris-virginica";

    for (const auto &_line : lines) {
        std::stringstream ss(_line);
        while (getline(ss, element, ',')){ 
            if (class_setosa.compare(element) == 0) {
                data.push_back(1.);
                data.push_back(0.);
                data.push_back(0.);
            }
            else if (class_versicolor.compare(element) == 0) {
                data.push_back(0.);
                data.push_back(1.);
                data.push_back(0.);
            }
            else if (class_virginica.compare(element) == 0) {
                data.push_back(0.);
                data.push_back(0.);
                data.push_back(1.);
            }
            else {
                double value = std::stof(element);
                data.push_back(value);
            }
        }
    }

    if (data.size() != EXPECTED_SIZE){
        throw std::invalid_argument("Wrong dataset size: " + std::to_string(data.size()));
    }

    Eigen::array<int, 2> dims({1, 0});
    auto tensor_map = Eigen::TensorMap<Eigen::Tensor<float, 2>>(data.data(), 7, 150).shuffle(dims);

    const int split_at = static_cast<int>(N_REGISTERS * split_percentage);

    Eigen::array<Eigen::Index, 2> training_x_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> training_x_extents = {split_at, 4};
    Eigen::Tensor<float, 2> training_X_ds = tensor_map.slice(training_x_offsets, training_x_extents);

    Eigen::array<Eigen::Index, 2> training_y_offsets = {0, 4};
    Eigen::array<Eigen::Index, 2> training_y_extents = {split_at, 3};
    Eigen::Tensor<float, 2> training_Y_ds = tensor_map.slice(training_y_offsets, training_y_extents);

    Eigen::array<Eigen::Index, 2> validation_x_offsets = {split_at, 0};
    Eigen::array<Eigen::Index, 2> validation_x_extents = {N_REGISTERS - split_at, 4};
    Eigen::Tensor<float, 2> validation_X_ds = tensor_map.slice(validation_x_offsets, validation_x_extents);

    Eigen::array<Eigen::Index, 2> validation_y_offsets = {split_at, 4};
    Eigen::array<Eigen::Index, 2> validation_y_extents = {N_REGISTERS - split_at, 3};
    Eigen::Tensor<float, 2> validation_Y_ds = tensor_map.slice(validation_y_offsets, validation_y_extents);

    auto result = std::make_tuple(training_X_ds, training_Y_ds, validation_X_ds, validation_Y_ds);

    return result;
};