cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch2_programs
        VERSION 1.0
        DESCRIPTION "Chapter 2 Programs"
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

# Create executables for all three programs
add_executable(hello_world "${CMAKE_CURRENT_LIST_DIR}/example.cpp")
add_executable(functional_demo "${CMAKE_CURRENT_LIST_DIR}/funtional/src/using_functional.cpp")
add_executable(demo "${CMAKE_CURRENT_LIST_DIR}/demo.cpp")

# Apply compiler options to all executables
target_compile_options(hello_world PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_options(functional_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_compile_options(demo PRIVATE -Wall -Wextra -Wpedantic -Werror)

# Set output directory for all executables
set_target_properties(hello_world PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
set_target_properties(functional_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
set_target_properties(demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")

# Add a custom target to run all programs
add_custom_target(run_all
    COMMAND echo "=== Running example.cpp ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/hello_world
    COMMAND echo ""
    COMMAND echo "=== Running using_functional.cpp ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/functional_demo
    COMMAND echo ""
    COMMAND echo "=== Running demo.cpp ==="
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/demo
    DEPENDS hello_world functional_demo demo
    COMMENT "Running all programs in ch-2"
)

# Individual run targets for convenience
add_custom_target(run_hello
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/hello_world
    DEPENDS hello_world
    COMMENT "Running example.cpp"
)

add_custom_target(run_functional
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/functional_demo
    DEPENDS functional_demo
    COMMENT "Running using_functional.cpp"
)

add_custom_target(run_demo
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/demo
    DEPENDS demo
    COMMENT "Running demo.cpp"
)