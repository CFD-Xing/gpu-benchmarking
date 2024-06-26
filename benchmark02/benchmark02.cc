#include <algorithm>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

#include "../utils/cuda_vectors.h"
#include "../utils/timer.h"

template <typename T, bool vl = false>
__global__ void add_vector(T *x, T *y, unsigned int n)
{
    constexpr unsigned int vecsize = (16u / sizeof(T));
    unsigned int tid               = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int stride            = blockDim.x * gridDim.x;

    if constexpr (vl && std::is_same_v<T, float>)
    {
        for (unsigned int id = tid; id < n / vecsize; id += stride)
        {
            float4 &x4      = reinterpret_cast<float4 *>(x)[id];
            const float4 y4 = reinterpret_cast<const float4 *>(y)[id];
            x4 += y4;
        }
    }
    else if constexpr (vl && std::is_same_v<T, double>)
    {
        for (unsigned int id = tid; id < n / vecsize; id += stride)
        {
            double2 &x2      = reinterpret_cast<double2 *>(x)[id];
            const double2 y2 = reinterpret_cast<const double2 *>(y)[id];
            x2 += y2;
        }
    }
    else
    {
        for (unsigned int id = tid; id < n; id += stride)
        {
            x[id] += y[id];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (tid < n % vecsize)
        {
            unsigned int id = n - 1u - tid;
            x[id] += y[id];
        }
    }
}

template <typename Functor>
__global__ void vector_kernel(const unsigned int begin, const unsigned int end,
                              Functor functor)
{
    unsigned int i = begin + blockDim.x * blockIdx.x + threadIdx.x;

    while (i < end)
    {
        functor(i);
        i += blockDim.x * gridDim.x;
    }
}

template <typename T> void run_test(const unsigned int size)
{
    Timer time;
    const unsigned int n_tests = 40;

    // Kokkos results
    double time_kokkos = std::numeric_limits<double>::max();
    T result_kokkos;
    {
        Kokkos::View<T *> data1("data1", size);
        Kokkos::View<T *> data2("data2", size);
        Kokkos::parallel_for(
            size, KOKKOS_LAMBDA(unsigned int i) {
                data1(i) = i % 13 + (0.2 + 0.00001 * (i % 100191));
                data2(i) = i % 8 + (0.4 + 0.00003 * (i % 100721));
            });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                size, KOKKOS_LAMBDA(unsigned int i) { data1(i) += data2(i); });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos      = std::min(time_kokkos, t_w);
        }
        Kokkos::parallel_reduce(
            size,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += data1(i) * data1(i);
            },
            result_kokkos);
    }

    // Thrust kernels
    double time_thrust = std::numeric_limits<double>::max();
    T result_thrust;
    {
        thrust::device_vector<T> ddata1(size);
        thrust::device_vector<T> ddata2(size);
        thrust::tabulate(ddata1.begin(), ddata1.end(),
                         [] __device__(unsigned int i)
                         { return i % 13 + (0.2 + 0.00001 * (i % 100191)); });
        thrust::tabulate(ddata2.begin(), ddata2.end(),
                         [] __device__(unsigned int i)
                         { return i % 8 + (0.4 + 0.00003 * (i % 100721)); });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            thrust::transform(
                ddata1.begin(), ddata1.end(), ddata2.begin(), ddata1.begin(),
                [] __device__(const T &x, const T &y) { return x + y; });
            time.stop();
            time_thrust = std::min(time_thrust, time.elapsedSeconds());
        }
        result_thrust = thrust::transform_reduce(
            ddata1.begin(), ddata1.end(), [] __device__(const T &x)
            { return x * x; }, (T)0.0, thrust::plus<T>());
    }

    // CUDA kernels 1 - No vector loading
    double time_cuda1 = std::numeric_limits<double>::max();
    T result_cuda1;
    {
        const int threads = 1024;
        const int blocks  = ((size / 8 + threads - 1) / threads);
        std::vector<T> host_vector1(size), host_vector2(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            host_vector1[i] = i % 13 + (0.2 + 0.00001 * (i % 100191));
            host_vector2[i] = i % 8 + (0.4 + 0.00003 * (i % 100721));
        }
        T *cuda_vector1, *cuda_vector2;
        cudaMalloc(&cuda_vector1, size * sizeof(T));
        cudaMalloc(&cuda_vector2, size * sizeof(T));
        cudaMemcpy(cuda_vector1, &host_vector1[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_vector2, &host_vector2[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            add_vector<T, false>
                <<<blocks, threads>>>(cuda_vector1, cuda_vector2, size);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda1 = std::min(time_cuda1, time.elapsedSeconds());
        }
        result_cuda1 = thrust::transform_reduce(
            thrust::device, cuda_vector1, cuda_vector1 + size,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(cuda_vector1);
        cudaFree(cuda_vector2);
    }

    // CUDA kernels 2 - Vector loading
    double time_cuda2 = std::numeric_limits<double>::max();
    T result_cuda2;
    {
        const int threads = 1024;
        const int blocks  = ((size / 8 + threads - 1) / threads);
        std::vector<T> host_vector1(size), host_vector2(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            host_vector1[i] = i % 13 + (0.2 + 0.00001 * (i % 100191));
            host_vector2[i] = i % 8 + (0.4 + 0.00003 * (i % 100721));
        }
        T *cuda_vector1, *cuda_vector2;
        cudaMalloc(&cuda_vector1, size * sizeof(T));
        cudaMalloc(&cuda_vector2, size * sizeof(T));
        cudaMemcpy(cuda_vector1, &host_vector1[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_vector2, &host_vector2[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            add_vector<T, true>
                <<<blocks, threads>>>(cuda_vector1, cuda_vector2, size);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda2 = std::min(time_cuda2, time.elapsedSeconds());
        }
        result_cuda2 = thrust::transform_reduce(
            thrust::device, cuda_vector1, cuda_vector1 + size,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(cuda_vector1);
        cudaFree(cuda_vector2);
    }

    // CUDA kernels 3 - functor
    double time_cuda3 = std::numeric_limits<double>::max();
    T result_cuda3;
    {
        const int threads = 1024;
        const int blocks  = ((size / 8 + threads - 1) / threads);
        std::vector<T> host_vector1(size), host_vector2(size);
        for (unsigned int i = 0; i < size; ++i)
        {
            host_vector1[i] = i % 13 + (0.2 + 0.00001 * (i % 100191));
            host_vector2[i] = i % 8 + (0.4 + 0.00003 * (i % 100721));
        }
        T *cuda_vector1, *cuda_vector2;
        cudaMalloc(&cuda_vector1, size * sizeof(T));
        cudaMalloc(&cuda_vector2, size * sizeof(T));
        cudaMemcpy(cuda_vector1, &host_vector1[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_vector2, &host_vector2[0], size * sizeof(T),
                   cudaMemcpyHostToDevice);
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            vector_kernel<<<blocks, threads>>>(
                0u, size, [=] __device__(unsigned int i)
                { cuda_vector1[i] += cuda_vector2[i]; });
            cudaDeviceSynchronize();
            time.stop();
            time_cuda3 = std::min(time_cuda3, time.elapsedSeconds());
        }
        result_cuda3 = thrust::transform_reduce(
            thrust::device, cuda_vector1, cuda_vector1 + size,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(cuda_vector1);
        cudaFree(cuda_vector2);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout << "Size " << size
              << " Case:     Kokkos      Thrust      Cuda        Cuda (vl)     "
                 "   Cuda (functor)"
              << std::endl;
    std::cout << "Size " << size << " norm: " << std::sqrt(result_kokkos) << " "
              << std::sqrt(result_thrust) << " "
              << " " << std::sqrt(result_cuda1) << " "
              << std::sqrt(result_cuda2) << " " << std::sqrt(result_cuda3)
              << std::endl;

    std::cout << "Size " << size
              << " GB/s: " << sizeof(T) * 3e-9 * size / time_kokkos << " "
              << sizeof(T) * 3e-9 * size / time_thrust << " "
              << sizeof(T) * 3e-9 * size / time_cuda1 << " "
              << sizeof(T) * 3e-9 * size / time_cuda2 << " "
              << sizeof(T) * 3e-9 * size / time_cuda3 << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Benchmark02 : Vector Addition   " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 1024; size < 1000000000u; size *= 2)
    {
        run_test<double>(size);
    }
    Kokkos::finalize();
}
