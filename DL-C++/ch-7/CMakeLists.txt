cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(ch7_softmax_demos
        VERSION 1.0
        DESCRIPTION "Chapter 7 Softmax Activation Function Demos"
        LANGUAGES CXX)

# ————— C++17 —————
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ————— Find Eigen3 —————
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# ————— Output dir —————
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ————— Sources —————
set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

# ——— softmax_demo ———
if(EXISTS "${SOURCE_DIR}/softmax.cpp")
  add_executable(softmax_demo "${SOURCE_DIR}/softmax.cpp")
  target_compile_options(softmax_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(softmax_demo PRIVATE Eigen3::Eigen)
  set(HAS_SOFTMAX TRUE)
  message(STATUS "Adding target: softmax_demo")
else()
  message(WARNING "softmax.cpp not found – skipping softmax_demo")
  set(HAS_SOFTMAX FALSE)
endif()

# ——— batch_softmax_demo ———
if(EXISTS "${SOURCE_DIR}/batch_softmax.cpp")
  add_executable(batch_softmax_demo "${SOURCE_DIR}/batch_softmax.cpp")
  target_compile_options(batch_softmax_demo PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(batch_softmax_demo PRIVATE Eigen3::Eigen)
  set(HAS_BATCH_SOFTMAX TRUE)
  message(STATUS "Adding target: batch_softmax_demo")
else()
  message(WARNING "batch_softmax.cpp not found – skipping batch_softmax_demo")
  set(HAS_BATCH_SOFTMAX FALSE)
endif()

# ——— iris_example ———
if(EXISTS "${SOURCE_DIR}/iris_example.c++")
  add_executable(iris_example "${SOURCE_DIR}/iris_example.c++")
  target_compile_options(iris_example PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(iris_example PRIVATE Eigen3::Eigen)
  set(HAS_IRIS TRUE)
  message(STATUS "Adding target: iris_example")

  # copy data/ → build/bin/data/
  add_custom_command(TARGET iris_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/data
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data
    COMMENT "Copying data/ → bin/data/"
  )
else()
  message(WARNING "iris_example.c++ not found – skipping iris_example")
  set(HAS_IRIS FALSE)
endif()

# ——— Collect all targets ———
# initialize to empty string
set(ALL_TARGETS "")

if(HAS_SOFTMAX)
  list(APPEND ALL_TARGETS softmax_demo)
endif()

if(HAS_BATCH_SOFTMAX)
  list(APPEND ALL_TARGETS batch_softmax_demo)
endif()

if(HAS_IRIS)
  list(APPEND ALL_TARGETS iris_example)
endif()

# ——— build_all ———
add_custom_target(build_all
  DEPENDS ${ALL_TARGETS}
  COMMENT "Building all available demos"
)

# ——— run_* convenience targets ———
if(HAS_SOFTMAX)
  add_custom_target(run_softmax
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
    DEPENDS softmax_demo
    COMMENT "Running softmax_demo"
  )
endif()

if(HAS_BATCH_SOFTMAX)
  add_custom_target(run_batch_softmax
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
    DEPENDS batch_softmax_demo
    COMMENT "Running batch_softmax_demo"
  )
endif()

if(HAS_IRIS)
  add_custom_target(run_iris
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ./iris_example data/Iris.csv
    DEPENDS iris_example
    COMMENT "Running iris_example"
  )
endif()

# ——— run_all ———
add_custom_target(run_all
  DEPENDS ${ALL_TARGETS}
  COMMENT "Running all demos"
)

if(HAS_SOFTMAX)
  add_custom_target(run_all_part1
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/softmax_demo
    DEPENDS softmax_demo
  )
  add_dependencies(run_all run_all_part1)
endif()

if(HAS_BATCH_SOFTMAX)
  add_custom_target(run_all_part2
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/batch_softmax_demo
    DEPENDS batch_softmax_demo
  )
  add_dependencies(run_all run_all_part2)
endif()

if(HAS_IRIS)
  add_custom_target(run_all_part3
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMAND ./iris_example data/Iris.csv
    DEPENDS iris_example
  )
  add_dependencies(run_all run_all_part3)
endif()

# ——— Summary ———
message(STATUS "----------------------------------------")
message(STATUS "  Output dir: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

if(HAS_SOFTMAX)
  message(STATUS "  - softmax_demo")
endif()

if(HAS_BATCH_SOFTMAX)
  message(STATUS "  - batch_softmax_demo")
endif()

if(HAS_IRIS)
  message(STATUS "  - iris_example")
endif()

message(STATUS "----------------------------------------")
