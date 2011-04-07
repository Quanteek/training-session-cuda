cmake_minimum_required(VERSION 2.8)
project(test_cuda)
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_executable(test main.cu)

