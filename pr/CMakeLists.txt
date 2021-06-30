cmake_minimum_required(VERSION 3.19)
project(pr)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-O3 ")


include_directories(.)

add_executable(pr SLIC.cpp SLIC.h)
