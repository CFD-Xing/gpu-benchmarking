##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(Benchmark05)
ADD_EXECUTABLE(benchmark05 benchmark05.cc)

FIND_PACKAGE(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark05 -lcublas Kokkos::kokkos)
