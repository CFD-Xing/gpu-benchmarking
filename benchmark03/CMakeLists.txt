##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(Benchmark03)
ADD_EXECUTABLE(benchmark03 benchmark03.cc)

FIND_PACKAGE(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark03 -lcublas Kokkos::kokkos)
