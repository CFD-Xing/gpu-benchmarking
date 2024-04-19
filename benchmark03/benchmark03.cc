#include <algorithm>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include <cublas.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "../utils/cuda_vectors.h"
#include "../utils/timer.h"

template <typename T, bool vl = false>
__device__ void dot_vl(T *sums, const T *x, const T *y, const unsigned int n)
{
    constexpr unsigned int vecsize = (16u / sizeof(T));
    auto block                     = cg::this_thread_block();
    auto warp                      = cg::tiled_partition<32>(block);
    T v                            = 0;

    if constexpr (vl && std::is_same_v<T, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = block.thread_rank(); tid < n / vecsize;
             tid += block.size())
        {
            const float4 x4 = reinterpret_cast<const float4 *>(x)[tid];
            const float4 y4 = reinterpret_cast<const float4 *>(y)[tid];
            v4 += x4 * y4;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<T, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = block.thread_rank(); tid < n / vecsize;
             tid += block.size())
        {
            const double2 x2 = reinterpret_cast<const double2 *>(x)[tid];
            const double2 y2 = reinterpret_cast<const double2 *>(y)[tid];
            v2 += x2 * y2;
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = block.thread_rank(); tid < n;
             tid += block.size())
        {
            v += x[tid] * y[tid];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (block.thread_rank() < n % vecsize)
        {
            unsigned int tid = n - 1u - block.thread_rank();
            v += x[tid] * y[tid];
        }
    }

    warp.sync();
    v += warp.shfl_down(v, 16); // |
    v += warp.shfl_down(v, 8);  // | warp level
    v += warp.shfl_down(v, 4);  // | reduce here
    v += warp.shfl_down(v, 2);  // |
    v += warp.shfl_down(v, 1);  // |

    // use atomicAdd to sum over warps
    if (warp.thread_rank() == 0)
    {
        atomicAdd(sums, v);
    }
}

template <typename T, bool vl = false>
__global__ void compute_matvec(const unsigned int N, const unsigned int M,
                               const T *A, const T *x, T *y)
{
    __shared__ T sum[1];

    for (unsigned int i = blockIdx.x; i < M; i += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            sum[0] = 0.0;
        }

        __syncthreads();

        dot_vl<T, vl>(sum, A + i * N, x, N);

        __syncthreads();

        if (threadIdx.x == 0)
        {
            y[i] = sum[0];
        }
    }
}

template <typename T> void run_test(const unsigned int size)
{
    Timer time;
    const unsigned int M       = size;
    const unsigned int N       = size;
    const unsigned int n_tests = 40;

    // Kokkos results
    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    double time_kokkos = std::numeric_limits<double>::max();
    std::vector<T> result_kokkos(1);
    {
        Kokkos::View<T *> d_A("d_A", M * N);
        Kokkos::View<T *> d_x("d_x", N);
        Kokkos::View<T *> d_y("d_y", M);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_handle &team) {
                unsigned int j = team.league_rank();
                d_x[j]         = j;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, M), [&](unsigned int i)
                    { d_A[i * N + j] = sin((T)(i * N + j + 1)); });
            });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                Kokkos::TeamPolicy<>(M, Kokkos::AUTO),
                KOKKOS_LAMBDA(const team_handle &team) {
                    T result;
                    unsigned int i = team.league_rank();
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(team, N),
                        [&](const unsigned int &j, T &sum)
                        { sum += d_A(i * N + j) * d_x(j); },
                        result);
                    d_y(i) = result;
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos      = std::min(time_kokkos, t_w);
        }
        Kokkos::parallel_reduce(
            M,
            KOKKOS_LAMBDA(unsigned int i, T &val) { val += d_y(i) * d_y(i); },
            result_kokkos[0]);
    }

    // cuBLAS kernels 1 - row major
    double time_cublas1 = std::numeric_limits<double>::max();
    std::vector<T> result_cublas1(1);
    {
        std::vector<T> h_A(M * N), h_x(N);
        for (unsigned int j = 0; j < N; j++)
        {
            h_x[j] = j;
            for (unsigned int i = 0; i < M; i++)
            {
                h_A[i * N + j] = std::sin((T)(i * N + j + 1));
            }
        }
        T *d_A, *d_x, *d_y;
        cudaMalloc(&d_A, M * N * sizeof(T));
        cudaMalloc(&d_x, N * sizeof(T));
        cudaMalloc(&d_y, M * sizeof(T));
        cudaMemcpy(d_A, &h_A[0], M * N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, &h_x[0], N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, M * sizeof(T));
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            if constexpr (std::is_same_v<T, float>)
            {
                cublasSgemv('T', M, N, 1.0f, d_A, M, d_x, 1, 0.0f, d_y, 1);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                cublasDgemv('T', M, N, 1.0, d_A, M, d_x, 1, 0.0, d_y, 1);
            }
            cudaDeviceSynchronize();
            time.stop();
            time_cublas1 = std::min(time_cublas1, time.elapsedSeconds());
        }
        result_cublas1[0] = thrust::transform_reduce(
            thrust::device, d_y, d_y + M,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // cuBLAS kernels 2 - column major
    double time_cublas2 = std::numeric_limits<double>::max();
    std::vector<T> result_cublas2(1);
    {
        std::vector<T> h_A(M * N), h_x(N);
        for (unsigned int j = 0; j < N; j++)
        {
            h_x[j] = j;
            for (unsigned int i = 0; i < M; i++)
            {
                h_A[j * M + i] = std::sin((T)(i * N + j + 1));
            }
        }
        T *d_A, *d_x, *d_y;
        cudaMalloc(&d_A, M * N * sizeof(T));
        cudaMalloc(&d_x, N * sizeof(T));
        cudaMalloc(&d_y, M * sizeof(T));
        cudaMemcpy(d_A, &h_A[0], M * N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, &h_x[0], N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, M * sizeof(T));
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            if constexpr (std::is_same_v<T, float>)
            {
                cublasSgemv('N', M, N, 1.0f, d_A, M, d_x, 1, 0.0f, d_y, 1);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                cublasDgemv('N', M, N, 1.0, d_A, M, d_x, 1, 0.0, d_y, 1);
            }
            cudaDeviceSynchronize();
            time.stop();
            time_cublas2 = std::min(time_cublas2, time.elapsedSeconds());
        }
        result_cublas2[0] = thrust::transform_reduce(
            thrust::device, d_y, d_y + M,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // CUDA 1 kernels - No vector loading
    double time_cuda1 = std::numeric_limits<double>::max();
    std::vector<T> result_cuda1(1);
    {
        const int threads = 256;
        const int blocks  = 256;
        std::vector<T> h_A(M * N), h_x(N);
        for (unsigned int j = 0; j < N; j++)
        {
            h_x[j] = j;
            for (unsigned int i = 0; i < M; i++)
            {
                h_A[i * N + j] = std::sin((T)(i * N + j + 1));
            }
        }
        T *d_A, *d_x, *d_y;
        cudaMalloc(&d_A, M * N * sizeof(T));
        cudaMalloc(&d_x, N * sizeof(T));
        cudaMalloc(&d_y, M * sizeof(T));
        cudaMemcpy(d_A, &h_A[0], M * N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, &h_x[0], N * sizeof(T), cudaMemcpyHostToDevice);
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            compute_matvec<T, false><<<blocks, threads>>>(N, M, d_A, d_x, d_y);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda1 = std::min(time_cuda1, time.elapsedSeconds());
        }
        result_cuda1[0] = thrust::transform_reduce(
            thrust::device, d_y, d_y + M,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // CUDA 2 kernels - Vector loading
    double time_cuda2 = std::numeric_limits<double>::max();
    std::vector<T> result_cuda2(1);
    {
        const int threads = 256;
        const int blocks  = 256;
        std::vector<T> h_A(M * N), h_x(N);
        for (unsigned int j = 0; j < N; j++)
        {
            h_x[j] = j;
            for (unsigned int i = 0; i < M; i++)
            {
                h_A[i * N + j] = std::sin((T)(i * N + j + 1));
            }
        }
        T *d_A, *d_x, *d_y;
        cudaMalloc(&d_A, M * N * sizeof(T));
        cudaMalloc(&d_x, N * sizeof(T));
        cudaMalloc(&d_y, M * sizeof(T));
        cudaMemcpy(d_A, &h_A[0], M * N * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, &h_x[0], N * sizeof(T), cudaMemcpyHostToDevice);
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            compute_matvec<T, true><<<blocks, threads>>>(N, M, d_A, d_x, d_y);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda2 = std::min(time_cuda2, time.elapsedSeconds());
        }
        result_cuda2[0] = thrust::transform_reduce(
            thrust::device, d_y, d_y + M,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_A);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout << "Size " << size
              << "           Kokkos      cuBLAS (rm)   cuBLAS (cm)     Cuda    "
                 "      Cuda (vl)"
              << std::endl;
    std::cout << "Size " << size << " norm: " << std::sqrt(result_kokkos[0])
              << "     " << std::sqrt(result_cublas1[0]) << "     "
              << std::sqrt(result_cublas2[0]) << "     "
              << std::sqrt(result_cuda1[0]) << "     "
              << std::sqrt(result_cuda2[0]) << std::endl;

    std::cout << "Size " << size
              << " GB/s: " << sizeof(T) * 1.0e-9 * M * N / time_kokkos
              << "     " << sizeof(T) * 1.0e-9 * M * N / time_cublas1 << "     "
              << sizeof(T) * 1.0e-9 * M * N / time_cublas2 << "     "
              << sizeof(T) * 1.0e-9 * M * N / time_cuda1 << "     "
              << sizeof(T) * 1.0e-9 * M * N / time_cuda2 << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Benchmark03 : Matrix-Vector Mult" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 2 << 6; size < 2 << 15; size *= 2)
    {
        run_test<float>(size);
    }
    Kokkos::finalize();
}
