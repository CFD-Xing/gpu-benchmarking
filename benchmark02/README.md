# Step 1
------------------
Clone the Kokkos library from GitHub:

    $ git clone https://github.com/kokkos/kokkos.git

# Step 2
------------------
Compile the Kokkos library:

    $ cd kokkos
    $ mkdir build && cd build 
    $ cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=~/kokkos-install -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON

# Step 3
------------------
Clone the benchmarking code from GitHub:

    $ git clone https://github.com/CFD-Xing/gpu-benchmarking.git

# Step 4
------------------
Compile the benchmakring code:

    $ cd gpu-benchmarking/benchmark02
    $ mkdir build && cd build
    $ cmake ../. -DKokkos_ROOT=~/kokkos-install/
