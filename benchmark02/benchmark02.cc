#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "cooperative_groups.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
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

template <typename T> void run_test(const unsigned int size)
{
    Timer time;
    const unsigned int n_tests = 40;

    // Kokkos results
    double time_kokkos = std::numeric_limits<double>::max();
    std::vector<T> result_kokkos(1);
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
            result_kokkos[0]);
    }

    // Thrust kernels
    double time_thrust = std::numeric_limits<double>::max();
    std::vector<T> result_thrust(1);
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
        result_thrust[0] = thrust::transform_reduce(
            ddata1.begin(), ddata1.end(),
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
    }

    // CUDA kernels
    double time_cuda = std::numeric_limits<double>::max();
    std::vector<T> result_cuda(1);
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
            add_vector<<<blocks, threads>>>(cuda_vector1, cuda_vector2, size);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda = std::min(time_cuda, time.elapsedSeconds());
        }
        result_cuda[0] = thrust::transform_reduce(
            thrust::device, cuda_vector1, cuda_vector1 + size,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(cuda_vector1);
        cudaFree(cuda_vector2);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout << "Size " << size << "           Kokkos      Thrust      Cuda"
              << std::endl;
    std::cout << "Size " << size << " norm: " << std::sqrt(result_kokkos[0])
              << " " << std::sqrt(result_thrust[0]) << " "
              << std::sqrt(result_cuda[0]) << std::endl;

    std::cout << "Size " << size
              << " GB/s: " << sizeof(T) * 3e-9 * size / time_kokkos << " "
              << sizeof(T) * 3e-9 * size / time_thrust << " "
              << sizeof(T) * 3e-9 * size / time_cuda << std::endl;
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
