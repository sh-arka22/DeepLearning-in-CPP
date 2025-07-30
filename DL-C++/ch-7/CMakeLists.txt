cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch7_softmax_demos
        VERSION 1.0
        DESCRIPTION "Chapter 7 Softmax Activation Function Demos"
        LANGUAGES CXX)

# C++17 support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Check for source files and create executables conditionally
set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

# Regular Softmax Demo
if(EXISTS "${SOURCE_DIR}/softmax.cpp")
    add_executable(softmax_demo "${SOURCE_DIR}/softmax.cpp")
    target_compile_options(softmax_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(softmax_demo Eigen3::Eigen)
    target_include_directories(softmax_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
    set(HAS_SOFTMAX TRUE)
    message(STATUS "Found softmax.cpp - adding target")
else()
    message(WARNING "softmax.cpp not found - skipping")
    set(HAS_SOFTMAX FALSE)
endif()

# Batch Softmax Demo
if(EXISTS "${SOURCE_DIR}/batch_softmax.cpp")
    add_executable(batch_softmax_demo "${SOURCE_DIR}/batch_softmax.cpp")
    target_compile_options(batch_softmax_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(batch_softmax_demo Eigen3::Eigen)
    target_include_directories(batch_softmax_demo PRIVATE ${EIGEN3_INCLUDE_DIR})
    set(HAS_BATCH_SOFTMAX TRUE)
    message(STATUS "Found batch_softmax.cpp - adding target")
else()
    message(WARNING "batch_softmax.cpp not found - skipping")
    set(HAS_BATCH_SOFTMAX FALSE)
endif()

# Create list of available targets
set(ALL_TARGETS)
if(HAS_SOFTMAX)
    list(APPEND ALL_TARGETS softmax_demo)
endif()
if(HAS_BATCH_SOFTMAX)
    list(APPEND ALL_TARGETS batch_softmax_demo)
endif()

# Build all target
add_custom_target(build_all
    DEPENDS ${ALL_TARGETS}
    COMMENT "Building all available softmax demos"
)

# Individual run targets
if(HAS_SOFTMAX)
    add_custom_target(run_softmax
        COMMAND echo "=== Regular Softmax Demo ==="
        COMMAND echo "Processes samples one by one"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        DEPENDS softmax_demo
        COMMENT "Running regular softmax demo"
    )
endif()

if(HAS_BATCH_SOFTMAX)
    add_custom_target(run_batch_softmax
        COMMAND echo "=== Batch Softmax Demo ==="
        COMMAND echo "Processes multiple samples simultaneously"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        DEPENDS batch_softmax_demo
        COMMENT "Running batch softmax demo"
    )
endif()

# Run all available demos
add_custom_target(run_all
    COMMAND echo "=== Chapter 7: Softmax Activation Functions ==="
    COMMAND echo "Demonstrating different softmax implementations"
    COMMAND echo ""
    DEPENDS ${ALL_TARGETS}
    COMMENT "Running all available softmax demos"
)

# Create conditional run_all implementations
if(HAS_SOFTMAX AND HAS_BATCH_SOFTMAX)
    # Both demos available
    add_custom_target(run_all_impl
        COMMAND echo "1. REGULAR SOFTMAX (Row-by-row processing)"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        COMMAND echo ""
        COMMAND echo "2. BATCH SOFTMAX (Vectorized processing)"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        DEPENDS ${ALL_TARGETS}
    )
    add_dependencies(run_all run_all_impl)
elseif(HAS_SOFTMAX)
    add_custom_target(run_all_impl
        COMMAND echo "1. REGULAR SOFTMAX (Row-by-row processing)"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        DEPENDS softmax_demo
    )
    add_dependencies(run_all run_all_impl)
elseif(HAS_BATCH_SOFTMAX)
    add_custom_target(run_all_impl
        COMMAND echo "1. BATCH SOFTMAX (Vectorized processing)"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        DEPENDS batch_softmax_demo
    )
    add_dependencies(run_all run_all_impl)
endif()

# Verbose run target with detailed explanations
add_custom_target(run_verbose
    COMMAND echo "=== Chapter 7: Softmax Activation Functions ==="
    COMMAND echo ""
    COMMAND echo "This chapter demonstrates two softmax implementations:"
    COMMAND echo "1. Regular Softmax - Row-by-row processing"
    COMMAND echo "2. Batch Softmax - Vectorized batch processing"
    COMMAND echo ""
    COMMAND echo "Both convert logits to probability distributions"
    COMMAND echo "Batch version is more efficient for multiple samples"
    COMMAND echo ""
    COMMAND echo "======================================================"
    DEPENDS ${ALL_TARGETS}
    COMMENT "Running all demos with detailed explanations"
)

# Create verbose implementation based on available targets
if(HAS_SOFTMAX)
    add_custom_target(run_verbose_softmax
        COMMAND echo ""
        COMMAND echo "1. REGULAR SOFTMAX DEMO"
        COMMAND echo "Processes 8 samples individually using chip() operations"
        COMMAND echo "Shows: Input logits -> Softmax probabilities"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        DEPENDS softmax_demo
    )
    add_dependencies(run_verbose run_verbose_softmax)
endif()

if(HAS_BATCH_SOFTMAX)
    add_custom_target(run_verbose_batch
        COMMAND echo ""
        COMMAND echo "2. BATCH SOFTMAX DEMO"
        COMMAND echo "Processes 3 samples simultaneously using broadcasting"
        COMMAND echo "More efficient: uses vectorized operations"
        COMMAND echo "Shows: Batch logits -> Batch probabilities"
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        DEPENDS batch_softmax_demo
    )
    add_dependencies(run_verbose run_verbose_batch)
endif()

# Simple run target (runs first available)
if(HAS_BATCH_SOFTMAX)
    add_custom_target(run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        DEPENDS batch_softmax_demo
        COMMENT "Running batch softmax demo"
    )
elseif(HAS_SOFTMAX)
    add_custom_target(run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        DEPENDS softmax_demo
        COMMENT "Running regular softmax demo"
    )
endif()

# Comparison target - runs both demos back to back
if(HAS_SOFTMAX AND HAS_BATCH_SOFTMAX)
    add_custom_target(run_comparison
        COMMAND echo "=== Softmax Implementation Comparison ==="
        COMMAND echo ""
        COMMAND echo "PART 1: Regular Softmax (Individual Processing)"
        COMMAND echo "Processing each sample separately..."
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
        COMMAND echo ""
        COMMAND echo "================================================"
        COMMAND echo ""
        COMMAND echo "PART 2: Batch Softmax (Vectorized Processing)"
        COMMAND echo "Processing all samples simultaneously..."
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
        COMMAND echo ""
        COMMAND echo "Comparison Summary:"
        COMMAND echo "- Regular: Good for understanding, easier to debug"
        COMMAND echo "- Batch: More efficient, better for production"
        COMMAND echo "- Both produce identical mathematical results"
        DEPENDS ${ALL_TARGETS}
        COMMENT "Running comparison between softmax implementations"
    )
endif()

# Print build summary
message(STATUS "=== Chapter 7 Build Configuration ===")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Available demos:")
if(HAS_SOFTMAX)
    message(STATUS "  - Regular Softmax (row-by-row)")
endif()
if(HAS_BATCH_SOFTMAX)
    message(STATUS "  - Batch Softmax (vectorized)")
endif()
message(STATUS "=====================================")