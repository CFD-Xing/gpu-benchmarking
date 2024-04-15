# Step 1
git clone https://github.com/kokkos/kokkos.git

# Step 2
cd kokkos\
mkdir build && cd build \
cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_INSTALL_PREFIX=~/kokkos-install -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=ON\
make install

# Step 3
git clone https://github.com/CFD-Xing/gpu-benchmarking.git

# Step 4
cd gpu-benchmarking/benchmark02\
mkdir build && cd build\
cmake ../. -DKokkos_ROOT=~/kokkos-install/\
make
