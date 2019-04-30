cmake_minimum_required(VERSION 3.14)
project(ACC CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES hw1.cu)

add_executable(HW_1 ${SOURCE_FILES})