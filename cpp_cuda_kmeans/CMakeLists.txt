cmake_minimum_required(VERSION 3.22)
project(kmeans_benchmark_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KMEANS_ENABLE_OPENMP "Enable OpenMP for CPU parallel implementation" ON)

if(KMEANS_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
    else()
        message(WARNING "OpenMP not found, building without OpenMP")
    endif()
endif()

set(SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SOURCES
    ${SRC_ROOT}/main.cpp
    ${SRC_ROOT}/core/kmeans_base.cpp
    ${SRC_ROOT}/core/kmeans_cpu.cpp
    ${SRC_ROOT}/core/kmeans_cpu_omp.cpp
    ${SRC_ROOT}/core/kmeans_gpu.cu
    ${SRC_ROOT}/data/dataset.cpp
    ${SRC_ROOT}/data/registry.cpp
    ${SRC_ROOT}/data/validation.cpp
    ${SRC_ROOT}/experiments/config.cpp
    ${SRC_ROOT}/experiments/runner.cpp
    ${SRC_ROOT}/experiments/suite.cpp
    ${SRC_ROOT}/metrics/metrics.cpp
    ${SRC_ROOT}/metrics/timers.cpp
    ${SRC_ROOT}/utils/logging.cpp
)

add_executable(kmeans_benchmark ${SOURCES})

target_include_directories(kmeans_benchmark PRIVATE ${SRC_ROOT})

target_compile_definitions(kmeans_benchmark PRIVATE KMEANS_CUDA_ENABLED=1)

set_target_properties(kmeans_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_benchmark PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(kmeans_benchmark PRIVATE KMEANS_OPENMP_ENABLED=1)
endif()

