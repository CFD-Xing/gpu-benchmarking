##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

PROJECT(Benchmark02)
ADD_EXECUTABLE(benchmark02 benchmark02.cc)

FIND_PACKAGE(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark02 Kokkos::kokkos)
