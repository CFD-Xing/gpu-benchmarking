##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

PROJECT(Benchmark01)
ADD_EXECUTABLE(benchmark01 benchmark01.cc)

FIND_PACKAGE(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark01 Kokkos::kokkos)
