#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "cooperative_groups.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

namespace cg = cooperative_groups;

class Timer
{
public:
    void start()
    {
        m_StartTime = std::chrono::high_resolution_clock::now();
        m_bRunning  = true;
    }

    void stop()
    {
        m_EndTime  = std::chrono::high_resolution_clock::now();
        m_bRunning = false;
    }

    double elapsedNanoseconds()
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

        if (m_bRunning)
        {
            endTime = std::chrono::high_resolution_clock::now();
        }
        else
        {
            endTime = m_EndTime;
        }

        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                    m_StartTime)
            .count();
    }

    double elapsedSeconds()
    {
        return elapsedNanoseconds() / 1.0e9;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool m_bRunning = false;
};

__host__ __device__ inline float4 &operator+=(float4 &a, const float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__host__ __device__ inline double2 &operator+=(double2 &a, const double2 b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ inline float4 &operator+=(float4 &a, const float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
    return a;
}

__host__ __device__ inline double2 &operator+=(double2 &a, const double b)
{
    a.x += b;
    a.y += b;
    return a;
}

__host__ __device__ inline float4 &operator-=(float4 &a, const float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

__host__ __device__ inline double2 &operator-=(double2 &a, const double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

__host__ __device__ inline float4 &operator-=(float4 &a, const float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
    return a;
}

__host__ __device__ inline double2 &operator-=(double2 &a, const double b)
{
    a.x -= b;
    a.y -= b;
    return a;
}

__host__ __device__ inline float4 &operator*=(float4 &a, const float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

__host__ __device__ inline double2 &operator*=(double2 &a, const double2 b)
{
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

__host__ __device__ inline float4 &operator*=(float4 &a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}

__host__ __device__ inline double2 &operator*=(double2 &a, const double b)
{
    a.x *= b;
    a.y *= b;
    return a;
}

__host__ __device__ inline float4 operator*(const float4 &a, const float4 &b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__host__ __device__ inline double2 operator*(const double2 &a, const double2 &b)
{
    return make_double2(a.x * b.x, a.y * b.y);
}

__host__ __device__ inline float4 operator*(const float &a, const float4 &b)
{
    return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

__host__ __device__ inline double2 operator*(const double &a, const double2 &b)
{
    return make_double2(a * b.x, a * b.y);
}

__host__ __device__ inline float4 operator+(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ inline double2 operator+(const double2 &a, const double2 &b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline float4 operator+(const float4 &a, const float &b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__host__ __device__ inline double2 operator+(const double2 &a, const double &b)
{
    return make_double2(a.x + b, a.y + b);
}

template <typename T, bool vl = true>
__global__ void l2norm_vl(T *__restrict__ sums, T *__restrict__ data,
                          unsigned int n)
{
    constexpr unsigned int vecsize = (16u / sizeof(T));
    auto grid                      = cg::this_grid();
    auto block                     = cg::this_thread_block();
    auto warp                      = cg::tiled_partition<32>(block);
    T v                            = 0;

    if constexpr (vl && std::is_same_v<T, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < n / vecsize;
             tid += grid.size())
        {
            const float4 tmp = reinterpret_cast<const float4 *>(data)[tid];
            v4 += tmp * tmp;
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<T, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < n / vecsize;
             tid += grid.size())
        {
            const double2 tmp = reinterpret_cast<const double2 *>(data)[tid];
            v2 += tmp * tmp;
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < n; tid += grid.size())
        {
            v += data[tid] * data[tid];
        }
    }

    // process final elements (if there are any)
    if constexpr (vl)
    {
        if (grid.thread_rank() < n % vecsize)
        {
            unsigned int tid = n - 1u - grid.thread_rank();
            v += data[tid] * data[tid];
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
        atomicAdd(&sums[block.group_index().x], v);
    }
}

template <typename T, bool vl = true>
__global__ void reduce_vl(T *__restrict__ sums, T *__restrict__ data,
                          unsigned int n)
{
    constexpr unsigned int vecsize = (16u / sizeof(T));
    auto grid                      = cg::this_grid();
    auto block                     = cg::this_thread_block();
    auto warp                      = cg::tiled_partition<32>(block);
    T v                            = 0;

    if constexpr (vl && std::is_same_v<T, float>)
    {
        float4 v4 = {0.0f, 0.0f, 0.0f, 0.0f}; // use v4 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < n / vecsize;
             tid += grid.size())
        {
            v4 += reinterpret_cast<const float4 *>(data)[tid];
        }
        v = v4.x + v4.y + v4.z + v4.w; // accumulate thread sums in v
    }
    else if constexpr (vl && std::is_same_v<T, double>)
    {
        double2 v2 = {0.0, 0.0}; // use v2 to read global memory
        for (unsigned int tid = grid.thread_rank(); tid < n / vecsize;
             tid += grid.size())
        {
            v2 += reinterpret_cast<const double2 *>(data)[tid];
        }
        v = v2.x + v2.y; // accumulate thread sums in v
    }
    else
    {
        for (unsigned int tid = grid.thread_rank(); tid < n; tid += grid.size())
        {
            v += data[tid];
        }
    }

    // process final elements (if there are any)
    if (vl && grid.thread_rank() < n % vecsize)
    {
        unsigned int tid = n - 1u - grid.thread_rank();
        v += data[tid];
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
        atomicAdd(&sums[block.group_index().x], v);
    }
}

template <typename T> void run_test(const unsigned int size)
{
    Timer time;
    const unsigned int n_tests = 40;

    // Kokkos results
    double time_kokkos = std::numeric_limits<double>::max();
    std::vector<T> result_kokkos(1);
    {
        Kokkos::View<T *> data("data", size);
        Kokkos::parallel_for(
            size, KOKKOS_LAMBDA(unsigned int i) {
                data(i) = i % 13 + (0.2 + 0.00001 * (i % 100191));
            });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_reduce(
                size,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += data(i) * data(i);
                },
                result_kokkos[0]);
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos      = std::min(time_kokkos, t_w);
        }
    }

    // Thrust kernels
    double time_thrust = std::numeric_limits<double>::max();
    std::vector<T> result_thrust(1);
    {
        thrust::device_vector<T> ddata(size);
        thrust::tabulate(ddata.begin(), ddata.end(),
                         [] __device__(unsigned int i)
                         { return i % 13 + (0.2 + 0.00001 * (i % 100191)); });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            result_thrust[0] = thrust::transform_reduce(
                ddata.begin(), ddata.end(),
                [] __device__(const T &x) { return x * x; }, (T)0.0,
                thrust::plus<T>());
            time.stop();
            time_thrust = std::min(time_thrust, time.elapsedSeconds());
        }
    }

    // CUDA kernels
    double time_cuda = std::numeric_limits<double>::max();
    std::vector<T> result_cuda(1);
    {
        constexpr int threads = 256;
        constexpr int blocks  = 256;
        thrust::device_vector<T> ddata(size);
        thrust::tabulate(ddata.begin(), ddata.end(),
                         [] __device__(unsigned int i)
                         { return i % 13 + (0.2 + 0.00001 * (i % 100191)); });
        for (unsigned int t = 0; t < n_tests; ++t)
        {
            time.start();
            T *sums, *result;
            cudaMalloc(&sums, blocks * sizeof(T));
            cudaMalloc(&result, sizeof(T));
            cudaMemset(sums, 0, blocks * sizeof(T));
            cudaMemset(result, 0, sizeof(T));
            l2norm_vl<<<blocks, threads>>>(sums, ddata.data().get(), size);
            reduce_vl<<<1, blocks>>>(result, sums, blocks);
            cudaMemcpy(result_cuda.data(), result, sizeof(T),
                       cudaMemcpyDeviceToHost);
            time.stop();
            time_cuda = std::min(time_cuda, time.elapsedSeconds());
            cudaFree(sums);
            cudaFree(result);
        }
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout << "Size " << size << "           Kokkos      Thrust      Cuda"
              << std::endl;
    std::cout << "Size " << size << " norm: " << std::sqrt(result_kokkos[0])
              << " " << std::sqrt(result_thrust[0]) << " "
              << std::sqrt(result_cuda[0]) << std::endl;

    std::cout << "Size " << size
              << " GB/s: " << sizeof(T) * 1e-9 * size / time_kokkos << " "
              << sizeof(T) * 1e-9 * size / time_thrust << " "
              << sizeof(T) * 1e-9 * size / time_cuda << std::endl;
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 1024; size < 1000000000u; size *= 2)
    {
        run_test<float>(size);
    }
    Kokkos::finalize();
}
