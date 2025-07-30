cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch5_ann
        VERSION 1.0
        DESCRIPTION "Chapter 5 Artificial Neural Networks"
        LANGUAGES CXX)
        
# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# C++17 is mandatory for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 package (required for tensor operations)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Create executable only for sigmoid (the file that exists)
add_executable(sigmoid_demo "${CMAKE_CURRENT_LIST_DIR}/src/sigmoid.cpp")

# Check if other files exist and add them conditionally
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/fc_layer.cpp")
    add_executable(fc_layer_demo "${CMAKE_CURRENT_LIST_DIR}/src/fc_layer.cpp")
    set(HAS_FC_LAYER TRUE)
else()
    message(WARNING "fc_layer.cpp not found - skipping fc_layer_demo")
    set(HAS_FC_LAYER FALSE)
endif()

if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/mlp_example.cpp")
    add_executable(mlp_demo "${CMAKE_CURRENT_LIST_DIR}/src/mlp_example.cpp")
    set(HAS_MLP_EXAMPLE TRUE)
else()
    message(WARNING "mlp_example.cpp not found - skipping mlp_demo")
    set(HAS_MLP_EXAMPLE FALSE)
endif()

# Apply compiler options to sigmoid_demo (always exists)
target_compile_options(sigmoid_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(sigmoid_demo Eigen3::Eigen)
target_include_directories(sigmoid_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(sigmoid_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

# Apply settings to fc_layer_demo if it exists
if(HAS_FC_LAYER)
    target_compile_options(fc_layer_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(fc_layer_demo Eigen3::Eigen)
    target_include_directories(fc_layer_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
    set_target_properties(fc_layer_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
endif()

# Apply settings to mlp_demo if it exists
if(HAS_MLP_EXAMPLE)
    target_compile_options(mlp_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(mlp_demo Eigen3::Eigen)
    target_include_directories(mlp_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
    set_target_properties(mlp_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
endif()

# Create dependencies list based on what exists
set(ALL_TARGETS sigmoid_demo)
if(HAS_FC_LAYER)
    list(APPEND ALL_TARGETS fc_layer_demo)
endif()
if(HAS_MLP_EXAMPLE)
    list(APPEND ALL_TARGETS mlp_demo)
endif()

# Add custom target to run all available programs
add_custom_target(run_all
    COMMAND echo "=== Running Sigmoid Activation Demo ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/sigmoid_demo
    $<$<BOOL:${HAS_FC_LAYER}>:COMMAND echo ""> 
    $<$<BOOL:${HAS_FC_LAYER}>:COMMAND echo "=== Running Fully Connected Layer Demo ==="> 
    $<$<BOOL:${HAS_FC_LAYER}>:COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/fc_layer_demo> 
    $<$<BOOL:${HAS_MLP_EXAMPLE}>:COMMAND echo "">
    $<$<BOOL:${HAS_MLP_EXAMPLE}>:COMMAND echo "=== Running Multi-Layer Perceptron Demo ===">
    $<$<BOOL:${HAS_MLP_EXAMPLE}>:COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mlp_demo>
    DEPENDS ${ALL_TARGETS}
    COMMENT "Running all available ANN programs"
)

# Individual run targets
add_custom_target(run_sigmoid
    COMMAND echo "=== Running Sigmoid Activation Demo ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/sigmoid_demo
    DEPENDS sigmoid_demo
    COMMENT "Running sigmoid activation function demo"
)

# Conditional targets for other programs
if(HAS_FC_LAYER)
    add_custom_target(run_fc
        COMMAND echo "=== Running Fully Connected Layer Demo ==="
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/fc_layer_demo
        DEPENDS fc_layer_demo
        COMMENT "Running fully connected layer demo"
    )
endif()

if(HAS_MLP_EXAMPLE)
    add_custom_target(run_mlp
        COMMAND echo "=== Running Multi-Layer Perceptron Demo ==="
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mlp_demo
        DEPENDS mlp_demo
        COMMENT "Running multi-layer perceptron demo"
    )
endif()

# Legacy run target (for backward compatibility)
add_custom_target(run
    COMMAND echo "=== Running Sigmoid Activation Demo ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/sigmoid_demo
    DEPENDS sigmoid_demo
    COMMENT "Running sigmoid activation function demo"
)

# Build only target
add_custom_target(build_all
    DEPENDS ${ALL_TARGETS}
    COMMENT "Building all available ANN programs"
)