cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch6_loss_functions
        VERSION 1.0
        DESCRIPTION "Chapter 6 Loss Functions and Cost Optimization"
        LANGUAGES CXX)
        
# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# C++17 is mandatory for modern features (structured bindings, etc.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3 package (required for tensor operations)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Create executable for MSE program
add_executable(mse_demo "${CMAKE_CURRENT_LIST_DIR}/src/mse.cpp")

# Apply compiler options
target_compile_options(mse_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Link Eigen3 libraries
target_link_libraries(mse_demo Eigen3::Eigen)

# Include Eigen headers
target_include_directories(mse_demo PRIVATE ${EIGEN3_INCLUDE_DIR})

# Set output directory
set_target_properties(mse_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

# Add custom target to run the MSE demo
add_custom_target(run
    COMMAND echo "=== Running Mean Squared Error Demo ==="
    COMMAND echo "This demo generates synthetic data and compares two hypothesis functions"
    COMMAND echo "Data: Y = 2*X + 3 + noise"
    COMMAND echo "H0: Y = -1*X + 4"
    COMMAND echo "H1: Y = 1.5*X + 1"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mse_demo
    DEPENDS mse_demo
    COMMENT "Running MSE cost function demo"
)

# Alternative target names for convenience
add_custom_target(run_mse
    COMMAND echo "=== Running Mean Squared Error Demo ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mse_demo
    DEPENDS mse_demo
    COMMENT "Running MSE demo"
)

add_custom_target(run_all
    COMMAND echo "=== Running All Chapter 6 Demos ==="
    COMMAND echo "1. Mean Squared Error Demo"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mse_demo
    COMMAND echo ""
    COMMAND echo "2. Categorical Cross-Entropy Demo"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/cce_demo
    DEPENDS mse_demo cce_demo
    COMMENT "Running all loss function demos"
)

# Build only target
add_custom_target(build_all
    DEPENDS mse_demo
    COMMENT "Building all loss function programs"
)

# Add custom target with verbose output (fixed shell escaping)
add_custom_target(run_verbose
    COMMAND echo "=== Mean Squared Error Cost Function Demo ==="
    COMMAND echo ""
    COMMAND echo "This demo demonstrates:"
    COMMAND echo "1. Synthetic data generation with noise"
    COMMAND echo "2. Linear regression hypothesis functions" 
    COMMAND echo "3. MSE cost calculation and comparison"
    COMMAND echo ""
    COMMAND echo "Generated data follows: Y = 2*X + 3 + noise"
    COMMAND echo "Testing two hypothesis functions:"
    COMMAND echo "  H0: Y = -1*X + 4"
    COMMAND echo "  H1: Y = 1.5*X + 1"
    COMMAND echo ""
    COMMAND echo "Expected: H1 should have lower cost than H0"
    COMMAND echo "=========================================="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mse_demo
    DEPENDS mse_demo
    COMMENT "Running MSE demo with detailed explanations"
)

# Add debugging target (with less strict compiler options)
add_executable(mse_debug "${CMAKE_CURRENT_LIST_DIR}/src/mse.cpp")
target_compile_options(mse_debug PRIVATE -Wall -Wextra -g -O0)
target_link_libraries(mse_debug Eigen3::Eigen)
target_include_directories(mse_debug PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(mse_debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

add_custom_target(run_debug
    COMMAND echo "=== Running MSE Demo - Debug Mode ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/mse_debug
    DEPENDS mse_debug
    COMMENT "Running MSE demo in debug mode"
)

# Add executable for CCE program
add_executable(cce_demo "${CMAKE_CURRENT_LIST_DIR}/src/cce.cpp")
target_compile_options(cce_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(cce_demo Eigen3::Eigen)
target_include_directories(cce_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
set_target_properties(cce_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

add_custom_target(run_cce
    COMMAND echo "=== Running Categorical Cross-Entropy Demo ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/cce_demo
    DEPENDS cce_demo
    COMMENT "Running CCE loss function demo"
)