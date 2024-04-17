#include <algorithm>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "cublas_v2.h"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "../utils/cuda_vectors.h"
#include "../utils/timer.h"

template <typename T>
__global__ void BwdTransQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ in, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = shared + nm0 * nq0;
    T *s_wsp0   = s_basis1 + nm1 * nq1;
    T *s_wsp1   = s_wsp0 + nmTot;

    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *outptr      = out + nq0 * nq1 * e;

        // Copy to shared memory.
        for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
        {
            unsigned int cnt_pi = nq0 * p;

            for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
            {
                s_basis0[cnt_pi + i] = basis0[cnt_pi + i];
            }
        }

        for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
        {
            unsigned int cnt_qj = nq1 * q;

            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                s_basis1[cnt_qj + j] = basis0[cnt_qj + j];
            }
        }

        for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
        {
            unsigned int cnt_qp = nm0 * q;

            for (unsigned int p = threadIdx.y; p < nm0; p += blockDim.y)
            {
                s_wsp0[cnt_qp + p] = inptr[cnt_qp + p];
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp = nm0 * q;

                T tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
                }
                s_wsp1[cnt_iq] = tmp;
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
        {
            for (unsigned int j = threadIdx.x; j < nq1; j += blockDim.x)
            {
                unsigned int cnt_iq = nm1 * i;
                unsigned int cnt_ji = nq0 * j + i;

                T tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
                {
                    tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
                }
                outptr[cnt_ji] = tmp;
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T> void run_test(const unsigned int size)
{
    Timer time;
    const unsigned int nelmt   = size;
    const unsigned int nq0     = 16u;
    const unsigned int nq1     = 16u;
    const unsigned int nm0     = nq0 - 1u;
    const unsigned int nm1     = nq1 - 1u;
    const unsigned int n_tests = 40u;

    // Kokkos results
    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    double time_kokkos = std::numeric_limits<double>::max();
    std::vector<T> result_kokkos(1);
    {
        Kokkos::View<T *> d_in("d_in", nelmt * nm0 * nm1);
        Kokkos::View<T *> d_out("d_out", nelmt * nq0 * nq1);
        Kokkos::View<T *> d_basis0("d_basis0", nm0 * nq0);
        Kokkos::View<T *> d_basis1("d_basis1", nm1 * nq1);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_handle &team) {
                unsigned int e = team.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                        team, nm0, nm1),
                    [&](const unsigned int &p, const unsigned int &q) {
                        d_in(e * nm0 * nm1 + p * nm1 + q) =
                            sin((T)(p * nm1 + q + 1));
                    });
            });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0u, 0u}, {nm0, nq0}),
            KOKKOS_LAMBDA(const unsigned p, const unsigned i) {
                d_basis0(p * nq0 + i) = cos((T)(p * nq0 + i));
            });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0u, 0u}, {nm1, nq1}),
            KOKKOS_LAMBDA(const unsigned q, const unsigned j) {
                d_basis1(q * nq1 + j) = cos((T)(q * nq1 + j));
            });
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            const unsigned int slevel = 0u;
            const unsigned int ssize =
                nm0 * nq0 + nm1 * nq1 + nm0 * nm1 + nq0 * nm1;

            const unsigned int shmem_size = Kokkos::View<
                T *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);

            Kokkos::parallel_for(
                Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO)
                    .set_scratch_size(slevel, Kokkos::PerTeam(shmem_size)),
                KOKKOS_LAMBDA(const team_handle &team) {
                    // element index
                    unsigned int e = team.league_rank();

                    // shared memory subview assignment
                    Kokkos::View<
                        T *,
                        Kokkos::DefaultExecutionSpace::scratch_memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                        scratch(team.team_scratch(slevel), ssize);
                    auto s_basis0 = Kokkos::subview(
                        scratch, Kokkos::make_pair(0u, nm0 * nq0));
                    auto s_basis1 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(nm0 * nq0, nm0 * nq0 + nm1 * nq1));
                    auto s_wsp0 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(nm0 * nq0 + nm1 * nq1,
                                          nm0 * nq0 + nm1 * nq1 + nm0 * nm1));
                    auto s_wsp1 = Kokkos::subview(
                        scratch, Kokkos::make_pair(
                                     nm0 * nq0 + nm1 * nq1 + nm0 * nm1, ssize));

                    // copy to shared memory
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                            team, nm0, nq0),
                        [&](const unsigned int &p, const unsigned int &i)
                        { s_basis0(p * nq0 + i) = d_basis0(p * nq0 + i); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                            team, nm1, nq1),
                        [&](const unsigned int &q, const unsigned int &j)
                        { s_basis1(q * nq1 + j) = d_basis1(q * nq1 + j); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                            team, nm1, nm0),
                        [&](const unsigned int &q, const unsigned int &p) {
                            s_wsp0(q * nm0 + p) =
                                d_in(e * nm0 * nm1 + q * nm0 + p);
                        });

                    team.team_barrier();

                    // direction 0
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                            team, nq0, nm1),
                        [&](const unsigned int &i, const unsigned int &q)
                        {
                            unsigned int cnt_iq = nm1 * i + q;
                            unsigned int cnt_qp = nm0 * q;

                            T tmp = 0.0;
                            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                            {
                                tmp += s_wsp0(cnt_qp) * s_basis0(p * nq0 + i);
                            }
                            s_wsp1(cnt_iq) = tmp;
                        });

                    team.team_barrier();

                    // direction 1
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                            team, nq0, nq1),
                        [&](const unsigned int &i, const unsigned int &j)
                        {
                            unsigned int cnt_iq = nm1 * i;
                            unsigned int cnt_ji = nq0 * j + i;

                            T tmp = 0.0;
                            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
                            {
                                tmp += s_wsp1(cnt_iq) * s_basis1(q * nq1 + j);
                            }
                            d_out(nq0 * nq1 * e + cnt_ji) = tmp;
                        });

                    team.team_barrier();
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos      = std::min(time_kokkos, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos[0]);
    }

    // cuBLAS kernels
    double time_cublas = std::numeric_limits<double>::max();
    std::vector<T> result_cublas(1);
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        std::vector<T> h_in(nelmt * nm0 * nm1);
        std::vector<T> h_out(nelmt * nq0 * nq1);
        std::vector<T> h_basis0(nm0 * nq0);
        std::vector<T> h_basis1(nm1 * nq1);
        for (unsigned int e = 0u; e < nelmt; e++)
        {
            for (unsigned int p = 0u; p < nm0; p++)
            {
                for (unsigned int q = 0u; q < nm1; q++)
                {
                    h_in[e * nm0 * nm1 + p * nm1 + q] =
                        sin((T)(p * nm1 + q + 1u));
                }
            }
        }
        for (unsigned int p = 0u; p < nm0; p++)
        {
            for (unsigned int i = 0u; i < nq0; i++)
            {
                h_basis0[p * nq0 + i] = std::cos((T)(p * nq0 + i));
            }
        }
        for (unsigned int q = 0u; q < nm1; q++)
        {
            for (unsigned int j = 0u; j < nq1; j++)
            {
                h_basis1[q * nq1 + j] = std::cos((T)(q * nq1 + j));
            }
        }
        T *d_in, *d_out, *d_basis0, *d_basis1, *d_wsp;
        cudaMalloc(&d_in, nelmt * nm0 * nm1 * sizeof(T));
        cudaMalloc(&d_out, nelmt * nq0 * nq1 * sizeof(T));
        cudaMalloc(&d_wsp, nelmt * nq0 * nm1 * sizeof(T));
        cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T));
        cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T));
        cudaMemcpy(d_in, &h_in[0], nelmt * nm0 * nm1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis0, &h_basis0[0], nm0 * nq0 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis1, &h_basis1[0], nm1 * nq1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        T alpha = 1.0;
        T beta  = 0.0;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            if constexpr (std::is_same_v<T, float>)
            {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nq0, nm1 * nelmt,
                            nm0, &alpha, d_basis0, nq0, d_in, nm0, &beta, d_wsp,
                            nq0);
                cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq0,
                                          nq1, nm1, &alpha, d_wsp, nq0,
                                          nq0 * nm1, d_basis1, nq1, 0, &beta,
                                          d_out, nq0, nq0 * nq1, nelmt);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nq0, nm1 * nelmt,
                            nm0, &alpha, d_basis0, nq0, d_in, nm0, &beta, d_wsp,
                            nq0);
                cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq0,
                                          nq1, nm1, &alpha, d_wsp, nq0,
                                          nq0 * nm1, d_basis1, nq1, 0, &beta,
                                          d_out, nq0, nq0 * nq1, nelmt);
            }
            cudaDeviceSynchronize();
            time.stop();
            time_cublas = std::min(time_cublas, time.elapsedSeconds());
        }
        result_cublas[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_in);
        cudaFree(d_out);
        cublasDestroy(handle);
    }

    // CUDA kernels
    double time_cuda = std::numeric_limits<double>::max();
    std::vector<T> result_cuda(1);
    {
        const int blocks = 256;
        std::vector<T> h_in(nelmt * nm0 * nm1);
        std::vector<T> h_out(nelmt * nq0 * nq1);
        std::vector<T> h_basis0(nm0 * nq0);
        std::vector<T> h_basis1(nm1 * nq1);
        for (unsigned int e = 0u; e < nelmt; e++)
        {
            for (unsigned int p = 0u; p < nm0; p++)
            {
                for (unsigned int q = 0u; q < nm1; q++)
                {
                    h_in[e * nm0 * nm1 + p * nm1 + q] =
                        sin((T)(p * nm1 + q + 1));
                }
            }
        }
        for (unsigned int p = 0u; p < nm0; p++)
        {
            for (unsigned int i = 0u; i < nq0; i++)
            {
                h_basis0[p * nq0 + i] = std::cos((T)(p * nq0 + i));
            }
        }
        for (unsigned int q = 0u; q < nm1; q++)
        {
            for (unsigned int j = 0u; j < nq1; j++)
            {
                h_basis1[q * nq1 + j] = std::cos((T)(q * nq1 + j));
            }
        }
        T *d_in, *d_out, *d_basis0, *d_basis1;
        cudaMalloc(&d_in, nelmt * nm0 * nm1 * sizeof(T));
        cudaMalloc(&d_out, nelmt * nq0 * nq1 * sizeof(T));
        cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T));
        cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T));
        cudaMemcpy(d_in, &h_in[0], nelmt * nm0 * nm1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis0, &h_basis0[0], nm0 * nq0 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis1, &h_basis1[0], nm1 * nq1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        const unsigned int ssize =
            nm0 * nq0 + nm1 * nq1 + nm0 * nm1 + nq0 * nm1;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel<<<blocks, dim3(nq0, nq1), sizeof(T) * ssize>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1, d_in,
                d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda = std::min(time_cuda, time.elapsedSeconds());
        }
        result_cuda[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_in);
        cudaFree(d_out);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout << "nelmt " << nelmt
              << "           Kokkos      cuBLAS        Cuda" << std::endl;
    std::cout << "nelmt " << nelmt << " norm: " << std::sqrt(result_kokkos[0])
              << "     " << std::sqrt(result_cublas[0]) << "     "
              << std::sqrt(result_cuda[0]) << std::endl;

    std::cout
        << "nelmt " << nelmt << " GB/s: "
        << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1 + nq0 * nq1) / time_kokkos
        << "     "
        << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1 + nq0 * nq1) / time_cublas
        << "     "
        << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1 + nq0 * nq1) / time_cuda
        << std::endl;
}

int main(int argc, char **argv)
{
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 2 << 6; size < 2 << 20; size <<= 1)
    {
        run_test<float>(size);
    }
    Kokkos::finalize();
}
