##
#  CMake script for the benchmark01 program:
##

CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

PROJECT(Benchmark03)
ADD_EXECUTABLE(benchmark03 benchmark03.cc)

SET_TARGET_PROPERTIES(benchmark03 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
TARGET_COMPILE_OPTIONS(benchmark03 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
                       --cudart=shared
                       --relocatable-device-code=true
                       >)
FIND_PACKAGE(Kokkos REQUIRED)
#TARGET_LINK_LIBRARIES(benchmark03 -lcublas Kokkos::kokkos)
TARGET_LINK_LIBRARIES(benchmark03 -lcublas -lcudadevrt Kokkos::kokkos)



