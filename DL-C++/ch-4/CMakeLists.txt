cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch4_opencv
        VERSION 1.0
        DESCRIPTION "Chapter 4 OpenCV Programs"
        LANGUAGES CXX)
        
# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV package
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Create executables for both OpenCV programs
add_executable(opencv_demo "${CMAKE_CURRENT_LIST_DIR}/src/using_openCV.cpp")
add_executable(convolution_demo "${CMAKE_CURRENT_LIST_DIR}/src/convolution_2d_example.cpp")

# Apply compiler options to both executables
target_compile_options(opencv_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_options(convolution_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Link OpenCV libraries to both executables
target_link_libraries(opencv_demo ${OpenCV_LIBS})
target_link_libraries(convolution_demo ${OpenCV_LIBS})

# Set output directory for both executables
set_target_properties(opencv_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
set_target_properties(convolution_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

# Add custom target to run all programs
add_custom_target(run_all
    COMMAND echo "=== Running OpenCV Image Processing Demo ==="
    COMMAND echo "Copying photo.png to working directory..."
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/photo.png ${CMAKE_CURRENT_BINARY_DIR}/photo.png
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/opencv_demo
    COMMAND echo ""
    COMMAND echo "=== Running Convolution 2D Example ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/convolution_demo
    DEPENDS opencv_demo convolution_demo
    COMMENT "Running all OpenCV programs"
)

# Individual run targets for convenience
add_custom_target(run_opencv
    COMMAND echo "=== Running OpenCV Image Processing Demo ==="
    COMMAND echo "Copying photo.png to working directory..."
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/src/photo.png ${CMAKE_CURRENT_BINARY_DIR}/photo.png
    COMMAND echo "Press any key to close windows"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/opencv_demo
    DEPENDS opencv_demo
    COMMENT "Running OpenCV image processing demo"
)

add_custom_target(run_convolution
    COMMAND echo "=== Running Convolution 2D Example ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/convolution_demo
    DEPENDS convolution_demo
    COMMENT "Running convolution 2D example"
)

# Build only target
add_custom_target(build_all
    DEPENDS opencv_demo convolution_demo
    COMMENT "Building all OpenCV programs"
)