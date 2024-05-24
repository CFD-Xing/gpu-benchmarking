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

template <typename T, bool shmem = false>
__global__ void BwdTransQuadKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    T *__restrict__ basis0, T *__restrict__ basis1, const T *__restrict__ in,
    T *__restrict__ wsp, T *__restrict__ out)
{
    extern __shared__ T shared[];

    T *s_basis0 = shmem ? shared : basis0;
    T *s_basis1 = shmem ? s_basis0 + nm0 * nq0 : basis1;

    // Copy to shared memory.
    if constexpr (shmem)
    {
        unsigned int sIndex = threadIdx.x;
        while (sIndex < nm0 * nq0)
        {
            s_basis0[sIndex] = basis0[sIndex];
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nm1 * nq1)
        {
            s_basis1[sIndex] = basis1[sIndex];
            sIndex += blockDim.x;
        }
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
            {
                T tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += in[nmTot * e + cnt_qp] * s_basis0[p * nq0 + i];
                }
                wsp[nm1 * e + q] = tmp;
            }

            for (unsigned int j = 0u; j < nq1; ++j)
            {
                T tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += wsp[nm1 * e + q] * s_basis1[q * nq1 + j];
                }
                out[nq0 * nq1 * e + nq0 * j + i] = tmp;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename T, bool shmem = false>
__global__ void BwdTransQuadKernel_Coa(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    T *__restrict__ basis0, T *__restrict__ basis1, const T *__restrict__ in,
    T *__restrict__ wsp, T *__restrict__ out)
{
    constexpr unsigned int warpsize = 32u;

    extern __shared__ T shared[];

    T *s_basis0 = shmem ? shared : basis0;
    T *s_basis1 = shmem ? s_basis0 + nm0 * nq0 : basis1;

    // Copy to shared memory.
    if constexpr (shmem)
    {
        unsigned int sIndex = threadIdx.x;
        while (sIndex < nm0 * nq0)
        {
            s_basis0[sIndex] = basis0[sIndex];
            sIndex += blockDim.x;
        }

        sIndex = threadIdx.x;
        while (sIndex < nm1 * nq1)
        {
            s_basis1[sIndex] = basis1[sIndex];
            sIndex += blockDim.x;
        }
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        unsigned int iwarp = e / warpsize;
        unsigned int ilane = e % warpsize;
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
            {
                T tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += in[iwarp * warpsize * nmTot + warpsize * cnt_qp +
                              ilane] *
                           s_basis0[p * nq0 + i];
                }
                wsp[iwarp * warpsize * nm1 + warpsize * q + ilane] = tmp;
            }

            for (unsigned int j = 0u; j < nq1; ++j)
            {
                T tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q)
                {
                    tmp += wsp[iwarp * warpsize * nm1 + warpsize * q + ilane] *
                           s_basis1[q * nq1 + j];
                }
                out[iwarp * warpsize * nq0 * nq1 + warpsize * (nq0 * j + i) +
                    ilane] = tmp;
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ in, T *__restrict__ wsp, T *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *wspptr      = wsp + nq0 * nm1 * e;
        T *outptr      = out + nq0 * nq1 * e;

        // direction 0
        for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
        {
            for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
            {
                unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp = nm0 * q;

                T tmp = 0.0;
                for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                {
                    tmp += inptr[cnt_qp] * basis0[p * nq0 + i];
                }
                wspptr[cnt_iq] = tmp;
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                unsigned int cnt_ji = nq0 * j + i;
                unsigned int cnt_iq = nm1 * i;

                T tmp = 0.0;
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_iq)
                {
                    tmp += wspptr[cnt_iq] * basis1[q * nq1 + j];
                }
                outptr[cnt_ji] = tmp;
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransQuadKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ in, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_wsp0   = s_basis1 + nm1 * nq1;
    T *s_wsp1   = s_wsp0 + nmTot;

    unsigned int e = blockIdx.x;

    // Copy to shared memory.
    for (unsigned int p = threadIdx.y; p < nm0; p += blockDim.y)
    {
        const unsigned int cnt_pi = nq0 * p;

        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            s_basis0[cnt_pi + i] = basis0[cnt_pi + i];
        }
    }

    for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
    {
        const unsigned int cnt_qj = nq1 * q;

        for (unsigned int j = threadIdx.x; j < nq1; j += blockDim.x)
        {
            s_basis1[cnt_qj + j] = basis1[cnt_qj + j];
        }
    }

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *outptr      = out + nq0 * nq1 * e;

        // Copy to shared memory.
        for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
        {
            const unsigned int cnt_qp = nm0 * q;

            for (unsigned int p = threadIdx.x; p < nm0; p += blockDim.x)
            {
                s_wsp0[cnt_qp + p] = inptr[cnt_qp + p];
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
        {
            for (unsigned int q = threadIdx.x; q < nm1; q += blockDim.x)
            {
                const unsigned int cnt_iq = nm1 * i + q;
                unsigned int cnt_qp       = nm0 * q;

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
        for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
        {
            for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
            {
                const unsigned int cnt_ji = nq0 * j + i;
                unsigned int cnt_iq       = nm1 * i;

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

template <typename T>
__global__ void BwdTransQuadKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ in, T *__restrict__ wsp, T *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *wspptr      = wsp + nq0 * nm1 * e;
        T *outptr      = out + nq0 * nq1 * e;

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int i = tid / nm1;

            T tmp = 0.0;
            for (unsigned int p = 0u, cnt_qp = nm0 * q; p < nm0; ++p, ++cnt_qp)
            {
                tmp += inptr[cnt_qp] * basis0[p * nq0 + i];
            }
            wspptr[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = tid / nq0;

            T tmp = 0.0;
            for (unsigned int q = 0u, cnt_iq = nm1 * i; q < nm1; ++q, ++cnt_iq)
            {
                tmp += wspptr[cnt_iq] * basis1[q * nq1 + j];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransQuadKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nmTot,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ in, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_wsp0   = s_basis1 + nm1 * nq1;
    T *s_wsp1   = s_wsp0 + nmTot;

    unsigned int e = blockIdx.x;

    // Copy to shared memory.
    for (unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
    {
        s_basis0[tid] = basis0[tid];
    }

    for (unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
    {
        s_basis1[tid] = basis1[tid];
    }

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *outptr      = out + nq0 * nq1 * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nm0 * nm1; tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int i = tid / nm1;

            T tmp = 0.0;
            for (unsigned int p = 0u, cnt_qp = nm0 * q; p < nm0; ++p, ++cnt_qp)
            {
                tmp += s_wsp0[cnt_qp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = tid / nq0;

            T tmp = 0.0;
            for (unsigned int q = 0u, cnt_iq = nm1 * i; q < nm1; ++q, ++cnt_iq)
            {
                tmp += s_wsp1[cnt_iq] * s_basis1[q * nq1 + j];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
void run_test(const unsigned int size, const unsigned int _nq0,
              const unsigned int _nq1, const unsigned int _threads,
              const unsigned int _elblocks)
{
    Timer time;
    const unsigned int nelmt   = size;
    const unsigned int nq0     = _nq0;
    const unsigned int nq1     = _nq1;
    const unsigned int nm0     = nq0 - 1u;
    const unsigned int nm1     = nq1 - 1u;
    const unsigned int n_tests = 40u;

    // Kokkos results
    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    double time_kokkos1 = std::numeric_limits<double>::max();
    double time_kokkos2 = std::numeric_limits<double>::max();
    double time_kokkos3 = std::numeric_limits<double>::max();
    double time_kokkos4 = std::numeric_limits<double>::max();
    T result_kokkos1;
    T result_kokkos2;
    T result_kokkos3;
    T result_kokkos4;
    {
        Kokkos::View<T *> d_in("d_in", nelmt * nm0 * nm1);
        Kokkos::View<T *> d_in_coa("d_in", nelmt * nm0 * nm1);
        Kokkos::View<T *> d_out("d_out", nelmt * nq0 * nq1);
        Kokkos::View<T *> d_wsp("d_wsp", nelmt * nq0 * nm1);
        Kokkos::View<T *> d_basis0("d_basis0", nm0 * nq0);
        Kokkos::View<T *> d_basis1("d_basis1", nm1 * nq1);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_handle &team) {
                unsigned int e                  = team.league_rank();
                constexpr unsigned int warpsize = 32u;
                unsigned int iwarp              = e / warpsize;
                unsigned int ilane              = e % warpsize;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, team_handle>(
                        team, nm0, nm1),
                    [&](const unsigned int &p, const unsigned int &q)
                    {
                        d_in(e * nm0 * nm1 + p * nm1 + q) =
                            sin((T)(p * nm1 + q + 1));
                        d_in_coa(iwarp * nm0 * nm1 * warpsize +
                                 (p * nm1 + q) * warpsize + ilane) =
                            sin((T)(p * nm1 + q + 1));
                    });
            });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0u, 0u}, {nm0, nq0}),
            KOKKOS_LAMBDA(const unsigned &p, const unsigned &i) {
                d_basis0(p * nq0 + i) = cos((T)(p * nq0 + i));
            });
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0u, 0u}, {nm1, nq1}),
            KOKKOS_LAMBDA(const unsigned &q, const unsigned &j) {
                d_basis1(q * nq1 + j) = cos((T)(q * nq1 + j));
            });

        // Kokkos 1 - Uncoalesce
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
                    for (unsigned int i = 0u; i < nq0; ++i)
                    {
                        for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
                        {
                            T tmp = 0.0;
                            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                            {
                                tmp += d_in(nm0 * nm1 * e + cnt_qp) *
                                       d_basis0(p * nq0 + i);
                            }
                            d_wsp(nm1 * e + q) = tmp;
                        }

                        for (unsigned int j = 0u; j < nq1; ++j)
                        {
                            T tmp = 0.0;
                            for (unsigned int q = 0u; q < nm1; ++q)
                            {
                                tmp +=
                                    d_wsp(nm1 * e + q) * d_basis1(q * nq1 + j);
                            }
                            d_out(nq0 * nq1 * e + nq0 * j + i) = tmp;
                        }
                    }
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos1     = std::min(time_kokkos1, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos1);

        // Kokkos 2 - Coalesce
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
                    constexpr unsigned int warpsize = 32u;
                    unsigned int iwarp              = e / warpsize;
                    unsigned int ilane              = e % warpsize;
                    for (unsigned int i = 0u; i < nq0; ++i)
                    {
                        for (unsigned int q = 0u, cnt_qp = 0u; q < nm1; ++q)
                        {
                            T tmp = 0.0;
                            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_qp)
                            {
                                tmp += d_in_coa(iwarp * warpsize * nm0 * nm1 +
                                                warpsize * cnt_qp + ilane) *
                                       d_basis0(p * nq0 + i);
                            }
                            d_wsp(iwarp * warpsize * nq1 + warpsize * q +
                                  ilane) = tmp;
                        }

                        for (unsigned int j = 0u; j < nq1; ++j)
                        {
                            T tmp = 0.0;
                            for (unsigned int q = 0u; q < nm1; ++q)
                            {
                                tmp += d_wsp(iwarp * warpsize * nq1 +
                                             warpsize * q + ilane) *
                                       d_basis1(q * nq1 + j);
                            }
                            d_out(iwarp * warpsize * nq0 * nq1 +
                                  warpsize * (nq0 * j + i) + ilane) = tmp;
                        }
                    }
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos2     = std::min(time_kokkos2, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos2);

        // Kokkos 3 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
                KOKKOS_LAMBDA(const team_handle &team) {
                    // element index
                    const unsigned int e = team.league_rank();

                    // direction 0
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nm1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int q = tid % nm1;
                            const unsigned int i = tid / nm1;

                            T tmp = 0.0;
                            for (unsigned int p = 0u, cnt_qp = nm0 * q; p < nm0;
                                 ++p, ++cnt_qp)
                            {
                                tmp += d_in(nm0 * nm1 * e + cnt_qp) *
                                       d_basis0(p * nq0 + i);
                            }
                            d_wsp(nm1 * nq0 * e + tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 1
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nq1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int i = tid % nq0;
                            const unsigned int j = tid / nq0;

                            T tmp = 0.0;
                            for (unsigned int q = 0u, cnt_iq = nm1 * i; q < nm1;
                                 ++q, ++cnt_iq)
                            {
                                tmp += d_wsp(nm1 * nq0 * e + cnt_iq) *
                                       d_basis1(q * nq1 + j);
                            }
                            d_out(nq0 * nq1 * e + tid) = tmp;
                        });

                    team.team_barrier();
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos3     = std::min(time_kokkos3, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos3);

        // Kokkos 4 - Shared memory
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
                    const unsigned int e = team.league_rank();

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
                        Kokkos::TeamThreadRange(team, nm0 * nq0),
                        [&](const unsigned int &tid)
                        { s_basis0(tid) = d_basis0(tid); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm1 * nq1),
                        [&](const unsigned int &tid)
                        { s_basis1(tid) = d_basis1(tid); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm1 * nm0),
                        [&](const unsigned int &tid)
                        { s_wsp0(tid) = d_in(e * nm0 * nm1 + tid); });

                    team.team_barrier();

                    // direction 0
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nm1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int q = tid % nm1;
                            const unsigned int i = tid / nm1;

                            T tmp = 0.0;
                            for (unsigned int p = 0u, cnt_qp = nm0 * q; p < nm0;
                                 ++p, ++cnt_qp)
                            {
                                tmp += s_wsp0(cnt_qp) * s_basis0(p * nq0 + i);
                            }
                            s_wsp1(tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 1
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nq1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int i = tid % nq0;
                            const unsigned int j = tid / nq0;

                            T tmp = 0.0;
                            for (unsigned int q = 0u, cnt_iq = nm1 * i; q < nm1;
                                 ++q, ++cnt_iq)
                            {
                                tmp += s_wsp1(cnt_iq) * s_basis1(q * nq1 + j);
                            }
                            d_out(nq0 * nq1 * e + tid) = tmp;
                        });

                    team.team_barrier();
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos4     = std::min(time_kokkos4, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos4);
    }

    // cuBLAS kernels
    double time_cublas = std::numeric_limits<double>::max();
    T result_cublas;
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
        result_cublas = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_in);
        cudaFree(d_wsp);
        cudaFree(d_out);
        cublasDestroy(handle);
    }

    // CUDA kernels
    double time_cuda1 = std::numeric_limits<double>::max();
    double time_cuda2 = std::numeric_limits<double>::max();
    double time_cuda3 = std::numeric_limits<double>::max();
    double time_cuda4 = std::numeric_limits<double>::max();
    double time_cuda5 = std::numeric_limits<double>::max();
    double time_cuda6 = std::numeric_limits<double>::max();
    T result_cuda1;
    T result_cuda2;
    T result_cuda3;
    T result_cuda4;
    T result_cuda5;
    T result_cuda6;
    {
        const unsigned int threads = _threads;
        const unsigned int blocks  = nelmt / _elblocks; // QP only
        std::vector<T> h_in(nelmt * nm0 * nm1);
        std::vector<T> h_in_coa(nelmt * nm0 * nm1);
        std::vector<T> h_out(nelmt * nq0 * nq1);
        std::vector<T> h_basis0(nm0 * nq0);
        std::vector<T> h_basis1(nm1 * nq1);
        for (unsigned int e = 0u; e < nelmt; e++)
        {
            constexpr unsigned int warpsize = 32u;
            unsigned int iwarp              = e / warpsize;
            unsigned int ilane              = e % warpsize;
            for (unsigned int p = 0u; p < nm0; p++)
            {
                for (unsigned int q = 0u; q < nm1; q++)
                {
                    h_in[e * nm0 * nm1 + p * nm1 + q] =
                        sin((T)(p * nm1 + q + 1));
                    h_in_coa[nm0 * nm1 * warpsize * iwarp +
                             (p * nm1 + q) * warpsize + ilane] =
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
        T *d_in, *d_in_coa, *d_wsp0, *d_wsp1, *d_out, *d_basis0, *d_basis1;
        cudaMalloc(&d_in, nelmt * nm0 * nm1 * sizeof(T));
        cudaMalloc(&d_in_coa, nelmt * nm0 * nm1 * sizeof(T));
        cudaMalloc(&d_wsp0, nelmt * nm1 * sizeof(T));
        cudaMalloc(&d_wsp1, nelmt * nq0 * nm1 * sizeof(T));
        cudaMalloc(&d_out, nelmt * nq0 * nq1 * sizeof(T));
        cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T));
        cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T));
        cudaMemcpy(d_in, &h_in[0], nelmt * nm0 * nm1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_coa, &h_in_coa[0], nelmt * nm0 * nm1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis0, &h_basis0[0], nm0 * nq0 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis1, &h_basis1[0], nm1 * nq1 * sizeof(T),
                   cudaMemcpyHostToDevice);

        // Cuda 1 - Non coalesce
        const unsigned int ssize1 = nm0 * nq0 + nm1 * nq1;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel<T, true><<<(nelmt + threads - 1u) / threads,
                                          threads, sizeof(T) * ssize1>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1, d_in,
                d_wsp0, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda1 = std::min(time_cuda1, time.elapsedSeconds());
        }
        result_cuda1 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 2 - Coalesce
        const unsigned int ssize2 = nm0 * nq0 + nm1 * nq1;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel_Coa<T, true><<<(nelmt + threads - 1u) / threads,
                                              threads, sizeof(T) * ssize2>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1,
                d_in_coa, d_wsp0, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda2 = std::min(time_cuda2, time.elapsedSeconds());
        }
        result_cuda2 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 3 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel_QP<<<blocks, dim3(std::min(nq0, 16u),
                                                 std::min(nq1, 16u))>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1, d_in,
                d_wsp1, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda3 = std::min(time_cuda3, time.elapsedSeconds());
        }
        result_cuda3 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 4 - Shared memory
        const unsigned int ssize4 =
            nm0 * nq0 + nm1 * nq1 + nm0 * nm1 + nq0 * nm1;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel_QP<<<
                blocks, dim3(std::min(nq0, 16u), std::min(nq1, 16u)),
                sizeof(T) * ssize4>>>(nm0, nm1, nm0 * nm1, nq0, nq1, nelmt,
                                      d_basis0, d_basis1, d_in, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda4 = std::min(time_cuda4, time.elapsedSeconds());
        }
        result_cuda4 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 5 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel_QP_1D<<<blocks, std::min(nq0 * nq1, threads)>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1, d_in,
                d_wsp1, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda5 = std::min(time_cuda5, time.elapsedSeconds());
        }
        result_cuda5 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 6 - Shared memory
        const unsigned int ssize6 =
            nm0 * nq0 + nm1 * nq1 + nm0 * nm1 + nq0 * nm1;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransQuadKernel_QP_1D<<<blocks, std::min(nq0 * nq1, threads),
                                       sizeof(T) * ssize6>>>(
                nm0, nm1, nm0 * nm1, nq0, nq1, nelmt, d_basis0, d_basis1, d_in,
                d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda6 = std::min(time_cuda6, time.elapsedSeconds());
        }
        result_cuda6 = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_in);
        cudaFree(d_in_coa);
        cudaFree(d_wsp0);
        cudaFree(d_wsp1);
        cudaFree(d_out);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout
        << "nelmt " << nelmt
        << " Case: Kokkos (Uncoales) Kokkos (Coales) Kokkos (QP)   Kokkos "
           "(QP/Shared) cuBLAS     "
           "     Cuda "
           "(Uncoales)"
           " Cuda (Coales)    Cuda (QP)      Cuda (QP/Shared)  Cuda (QP-1D)  "
           " Cuda (QP-1D/Shared)"
        << std::endl;
    std::cout << "nelmt " << nelmt << " norm: " << std::sqrt(result_kokkos1)
              << "     " << std::sqrt(result_kokkos2) << "     "
              << std::sqrt(result_kokkos3) << "     "
              << std::sqrt(result_kokkos4) << "     "
              << std::sqrt(result_cublas) << "     " << std::sqrt(result_cuda1)
              << "     " << std::sqrt(result_cuda2) << "     "
              << std::sqrt(result_cuda3) << "     " << std::sqrt(result_cuda4)
              << "     " << std::sqrt(result_cuda5) << "     "
              << std::sqrt(result_cuda6) << std::endl;

    std::cout
        << "nelmt " << nelmt
        << " DOF/s: " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_kokkos1
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_kokkos2
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_kokkos3
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_kokkos4
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cublas
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda1
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda2
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda3
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda4
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda5
        << "     " << sizeof(T) * 1.0e-9 * nelmt * (nm0 * nm1) / time_cuda6
        << std::endl;
    std::cout << std::flush;
}

int main(int argc, char **argv)
{
    unsigned int nq0      = (argc > 1) ? atoi(argv[1]) : 8u;
    unsigned int nq1      = (argc > 2) ? atoi(argv[2]) : 8u;
    unsigned int threads  = (argc > 3) ? atoi(argv[3]) : 128u;
    unsigned int elblocks = (argc > 4) ? atoi(argv[4]) : 1u;

    std::cout << "--------------------------------" << std::endl;
    std::cout << "Benchmark04 : BwdTrans (2D)     " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "BwdTrans (NQ = " << nq0 << ", " << nq1 << ")" << std::endl;
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 2 << 6; size < 2 << 20; size <<= 1)
    {
        run_test<float>(size, nq0, nq1, threads, elblocks);
    }
    Kokkos::finalize();
}
