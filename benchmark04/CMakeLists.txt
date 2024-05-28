##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(Benchmark04)
ADD_EXECUTABLE(benchmark04 benchmark04.cc)

FIND_PACKAGE(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark04 -lcublas Kokkos::kokkos)
