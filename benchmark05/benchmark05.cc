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

template <typename T, bool shmem = false, bool coal = false>
__global__ void BwdTransHexKernel(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, T *__restrict__ basis0,
    T *__restrict__ basis1, T *__restrict__ basis2, const T *__restrict__ in,
    T *__restrict__ wsp0, T *__restrict__ wsp1, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shmem ? shared : basis0;
    T *s_basis1 = shmem ? s_basis0 + nm0 * nq0 : basis1;
    T *s_basis2 = shmem ? s_basis1 + nm1 * nq1 : basis2;

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

        sIndex = threadIdx.x;
        while (sIndex < nm2 * nq2)
        {
            s_basis2[sIndex] = basis2[sIndex];
            sIndex += blockDim.x;
        }
    }

    __syncthreads();

    unsigned int e = blockDim.x * blockIdx.x + threadIdx.x;

    while (e < nelmt)
    {
        for (unsigned int i = 0u; i < nq0; ++i)
        {
            for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u; r < nm2; ++r)
            {
                for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                {
                    T tmp = 0.0;
                    if constexpr (coal)
                    {
                        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                        {
                            tmp +=
                                in[nelmt * cnt_rqp + e] * s_basis0[p * nq0 + i];
                        }
                        wsp0[nelmt * cnt_rq + e] = tmp;
                    }
                    else
                    {
                        for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                        {
                            tmp +=
                                in[nmTot * e + cnt_rqp] * s_basis0[p * nq0 + i];
                        }
                        wsp0[nm1 * nm2 * e + cnt_rq] = tmp;
                    }
                }
            }

            for (unsigned int j = 0u; j < nq1; ++j)
            {
                for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
                {
                    if constexpr (coal)
                    {
                        T tmp = 0.0;
                        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                        {
                            tmp += wsp0[nelmt * cnt_rq + e] *
                                   s_basis1[q * nq1 + j];
                        }
                        wsp1[nelmt * r + e] = tmp;
                    }
                    else
                    {
                        T tmp = 0.0;
                        for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                        {
                            tmp += wsp0[nm1 * nm2 * e + cnt_rq] *
                                   s_basis1[q * nq1 + j];
                        }
                        wsp1[nm2 * e + r] = tmp;
                    }
                }

                for (unsigned int k = 0u; k < nq2; ++k)
                {
                    if constexpr (coal)
                    {
                        T tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2; ++r)
                        {
                            tmp += wsp1[nelmt * r + e] * s_basis2[r * nq2 + k];
                        }
                        out[nelmt * (k * nq1 * nq0 + j * nq0 + i) + e] = tmp;
                    }
                    else
                    {
                        T tmp = 0.0;
                        for (unsigned int r = 0u; r < nm2; ++r)
                        {
                            tmp += wsp1[nm2 * e + r] * s_basis2[r * nq2 + k];
                        }
                        out[nq0 * nq1 * nq2 * e + k * nq1 * nq0 + j * nq0 + i] =
                            tmp;
                    }
                }
            }
        }

        e += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in,
    T *__restrict__ wsp1, T *__restrict__ wsp2, T *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *wspptr1     = wsp1 + nq0 * nm1 * nm2 * e;
        T *wspptr2     = wsp2 + nq0 * nq1 * nm2 * e;
        T *outptr      = out + nq0 * nq1 * nq2 * e;

        // direction 0
        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            for (unsigned int r = threadIdx.y; r < nm2; r += blockDim.y)
            {
                for (unsigned int q = threadIdx.z; q < nm1; q += blockDim.z)
                {
                    unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r + q;

                    T tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += inptr[cnt_rqp] * basis0[p * nq0 + i];
                    }
                    wspptr1[cnt_irq] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.x; j < nq1; j += blockDim.x)
        {
            for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
            {
                for (unsigned int r = threadIdx.z; r < nm2; r += blockDim.z)
                {
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i + r;

                    T tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                    {
                        tmp += wspptr1[cnt_irq] * basis1[q * nq1 + j];
                    }
                    wspptr2[cnt_jir] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.x; k < nq2; k += blockDim.x)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.z; i < nq0; i += blockDim.z)
                {
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;

                    T tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                    {
                        tmp += wspptr2[cnt_jir] * basis2[r * nq2 + k];
                    }
                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransHexKernel_QP(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0   = s_basis2 + nm2 * nq2;
    T *s_wsp1   = s_wsp0 + nmTot;
    T *s_wsp2   = s_wsp1 + (nq0 * nm1 * nm2);

    unsigned int e = blockIdx.x;

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
            s_basis1[cnt_qj + j] = basis1[cnt_qj + j];
        }
    }

    for (unsigned int r = threadIdx.x; r < nm2; r += blockDim.x)
    {
        unsigned int cnt_rk = nq2 * r;

        for (unsigned int k = threadIdx.y; k < nq2; k += blockDim.y)
        {
            s_basis2[cnt_rk + k] = basis2[cnt_rk + k];
        }
    }

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *outptr      = out + nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int r = threadIdx.x; r < nm2; r += blockDim.x)
        {
            for (unsigned int q = threadIdx.y; q < nm1; q += blockDim.y)
            {
                unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

                for (unsigned int p = threadIdx.z; p < nm0; p += blockDim.z)
                {
                    s_wsp0[cnt_rqp + p] = inptr[cnt_rqp + p];
                }
            }
        }

        __syncthreads();

        // direction 0
        for (unsigned int i = threadIdx.x; i < nq0; i += blockDim.x)
        {
            for (unsigned int r = threadIdx.y; r < nm2; r += blockDim.y)
            {
                for (unsigned int q = threadIdx.z; q < nm1; q += blockDim.z)
                {
                    unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r + q;

                    T tmp = 0.0;
                    for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                    {
                        tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
                    }
                    s_wsp1[cnt_irq] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 1
        for (unsigned int j = threadIdx.x; j < nq1; j += blockDim.x)
        {
            for (unsigned int i = threadIdx.y; i < nq0; i += blockDim.y)
            {
                for (unsigned int r = threadIdx.z; r < nm2; r += blockDim.z)
                {
                    unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i + r;

                    T tmp = 0.0;
                    for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                    {
                        tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
                    }
                    s_wsp2[cnt_jir] = tmp;
                }
            }
        }

        __syncthreads();

        // direction 2
        for (unsigned int k = threadIdx.x; k < nq2; k += blockDim.x)
        {
            for (unsigned int j = threadIdx.y; j < nq1; j += blockDim.y)
            {
                for (unsigned int i = threadIdx.z; i < nq0; i += blockDim.z)
                {
                    unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;
                    unsigned int cnt_kji = nq0 * nq1 * k + nq0 * j + i;

                    T tmp = 0.0;
                    for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                    {
                        tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
                    }
                    outptr[cnt_kji] = tmp;
                }
            }
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in,
    T *__restrict__ wsp1, T *__restrict__ wsp2, T *__restrict__ out)
{
    unsigned int e = blockIdx.x;

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *wspptr1     = wsp1 + nq0 * nm1 * nm2 * e;
        T *wspptr2     = wsp2 + nq0 * nq1 * nm2 * e;
        T *outptr      = out + nq0 * nq1 * nq2 * e;

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int r = (tid / nm1) % nm2;
            const unsigned int i = tid / (nm1 * nm2);
            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

            T tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
            {
                tmp += inptr[cnt_rqp] * basis0[p * nq0 + i];
            }
            wspptr1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int r = tid % nm2;
            const unsigned int i = (tid / nm2) % nq0;
            const unsigned int j = tid / (nm2 * nq0);
            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

            T tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
            {
                tmp += wspptr1[cnt_irq] * basis1[q * nq1 + j];
            }
            wspptr2[tid] = tmp;
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = (tid / nq0) % nq1;
            const unsigned int k = tid / (nq1 * nq0);
            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

            T tmp = 0.0;
            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
            {
                tmp += wspptr2[cnt_jir] * basis2[r * nq2 + k];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nmTot, const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in, T *__restrict__ out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0   = s_basis2 + nm2 * nq2;
    T *s_wsp1   = s_wsp0 + nmTot;
    T *s_wsp2   = s_wsp1 + (nq0 * nm1 * nm2);

    unsigned int e = blockIdx.x;

    // Copy to shared memory.
    for (unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = basis0[tid];
    }

    for (unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = basis1[tid];
    }

    for (unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = basis2[tid];
    }

    while (e < nelmt)
    {
        const T *inptr = in + nmTot * e;
        T *outptr      = out + nq0 * nq1 * nq2 * e;

        // Copy to shared memory.
        for (unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2;
             tid += blockDim.x)
        {
            s_wsp0[tid] = inptr[tid];
        }

        __syncthreads();

        // direction 0
        for (unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int q = tid % nm1;
            const unsigned int r = (tid / nm1) % nm2;
            const unsigned int i = tid / (nm1 * nm2);
            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

            T tmp = 0.0;
            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
            {
                tmp += s_wsp0[cnt_rqp] * s_basis0[p * nq0 + i];
            }
            s_wsp1[tid] = tmp;
        }

        __syncthreads();

        // direction 1
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2;
             tid += blockDim.x)
        {
            const unsigned int r = tid % nm2;
            const unsigned int i = (tid / nm2) % nq0;
            const unsigned int j = tid / (nm2 * nq0);
            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

            T tmp = 0.0;
            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
            {
                tmp += s_wsp1[cnt_irq] * s_basis1[q * nq1 + j];
            }
            s_wsp2[tid] = tmp;
        }

        __syncthreads();

        // direction 2
        for (unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2;
             tid += blockDim.x)
        {
            const unsigned int i = tid % nq0;
            const unsigned int j = (tid / nq0) % nq1;
            const unsigned int k = tid / (nq1 * nq0);
            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

            T tmp = 0.0;
            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
            {
                tmp += s_wsp2[cnt_jir] * s_basis2[r * nq2 + k];
            }
            outptr[tid] = tmp;
        }

        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
void run_test(const unsigned int size, const unsigned int _nq0,
              const unsigned int _nq1, const unsigned int _nq2)
{
    Timer time;
    const unsigned int nelmt   = size;
    const unsigned int nq0     = _nq0;
    const unsigned int nq1     = _nq1;
    const unsigned int nq2     = _nq2;
    const unsigned int nm0     = nq0 - 1u;
    const unsigned int nm1     = nq1 - 1u;
    const unsigned int nm2     = nq2 - 1u;
    const unsigned int n_tests = 40u;

    // Kokkos results
    typedef Kokkos::TeamPolicy<>::member_type team_handle;
    double time_kokkos1 = std::numeric_limits<double>::max();
    double time_kokkos2 = std::numeric_limits<double>::max();
    double time_kokkos3 = std::numeric_limits<double>::max();
    double time_kokkos4 = std::numeric_limits<double>::max();
    std::vector<T> result_kokkos1(1);
    std::vector<T> result_kokkos2(1);
    std::vector<T> result_kokkos3(1);
    std::vector<T> result_kokkos4(1);
    {
        Kokkos::View<T *> d_in("d_in", nelmt * nm0 * nm1 * nm2);
        Kokkos::View<T *> d_in_coa("d_in_coa", nelmt * nm0 * nm1 * nm2);
        Kokkos::View<T *> d_out("d_out", nelmt * nq0 * nq1 * nq2);
        Kokkos::View<T *> d_wsp1("d_wsp1", nelmt * nq0 * nm1 * nm2);
        Kokkos::View<T *> d_wsp2("d_wsp2", nelmt * nq0 * nq1 * nm2);
        Kokkos::View<T *> d_basis0("d_basis0", nm0 * nq0);
        Kokkos::View<T *> d_basis1("d_basis1", nm1 * nq1);
        Kokkos::View<T *> d_basis2("d_basis2", nm2 * nq2);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
            KOKKOS_LAMBDA(const team_handle &team) {
                unsigned int e = team.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, team_handle>(
                        team, nm0, nm1, nm2),
                    [&](const unsigned int &p, const unsigned int &q,
                        const unsigned int &r)
                    {
                        d_in(e * nm0 * nm1 * nm2 + p * nm1 * nm2 + q * nm2 +
                             r) = sin((T)(p * nm1 * nm2 + q * nm2 + r + 1));
                        d_in_coa((p * nm1 * nm2 + q * nm2 + r) * nelmt + e) =
                            sin((T)(p * nm1 * nm2 + q * nm2 + r + 1u));
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
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({0u, 0u}, {nm2, nq2}),
            KOKKOS_LAMBDA(const unsigned q, const unsigned j) {
                d_basis2(q * nq2 + j) = cos((T)(q * nq2 + j));
            });

        // Kokkos 1 - No coales
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
                    for (unsigned int i = 0u; i < nq0; ++i)
                    {
                        for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u;
                             r < nm2; ++r)
                        {
                            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                            {
                                T tmp = 0.0;
                                for (unsigned int p = 0u; p < nm0;
                                     ++p, ++cnt_rqp)
                                {
                                    tmp += d_in(nm0 * nm1 * nm2 * e + cnt_rqp) *
                                           d_basis0(p * nq0 + i);
                                }
                                d_wsp1(nm1 * nm2 * e + cnt_rq) = tmp;
                            }
                        }

                        for (unsigned int j = 0u; j < nq1; ++j)
                        {
                            for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
                            {
                                T tmp = 0.0;
                                for (unsigned int q = 0u; q < nm1;
                                     ++q, ++cnt_rq)
                                {
                                    tmp += d_wsp1(nm1 * nm2 * e + cnt_rq) *
                                           d_basis1(q * nq1 + j);
                                }
                                d_wsp2(nm2 * e + r) = tmp;
                            }

                            for (unsigned int k = 0u; k < nq2; ++k)
                            {
                                T tmp = 0.0;
                                for (unsigned int r = 0u; r < nm2; ++r)
                                {
                                    tmp += d_wsp2(nm2 * e + r) *
                                           d_basis2(r * nq2 + k);
                                }
                                d_out(nq0 * nq1 * nq2 * e + k * nq1 * nq0 +
                                      j * nq0 + i) = tmp;
                            }
                        }
                    }
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos1     = std::min(time_kokkos1, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1 * nq2,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos1[0]);

        // Kokkos 2 - Coales
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                nelmt, KOKKOS_LAMBDA(const unsigned int &e) {
                    for (unsigned int i = 0u; i < nq0; ++i)
                    {
                        for (unsigned int r = 0u, cnt_rqp = 0u, cnt_rq = 0u;
                             r < nm2; ++r)
                        {
                            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_rq)
                            {
                                T tmp = 0.0;
                                for (unsigned int p = 0u; p < nm0;
                                     ++p, ++cnt_rqp)
                                {
                                    tmp += d_in_coa(nelmt * cnt_rqp + e) *
                                           d_basis0(p * nq0 + i);
                                }
                                d_wsp1(nelmt * cnt_rq + e) = tmp;
                            }
                        }

                        for (unsigned int j = 0u; j < nq1; ++j)
                        {
                            for (unsigned int r = 0u, cnt_rq = 0u; r < nm2; ++r)
                            {
                                T tmp = 0.0;
                                for (unsigned int q = 0u; q < nm1;
                                     ++q, ++cnt_rq)
                                {
                                    tmp += d_wsp1(nelmt * cnt_rq + e) *
                                           d_basis1(q * nq1 + j);
                                }
                                d_wsp2(nelmt * r + e) = tmp;
                            }

                            for (unsigned int k = 0u; k < nq2; ++k)
                            {
                                T tmp = 0.0;
                                for (unsigned int r = 0u; r < nm2; ++r)
                                {
                                    tmp += d_wsp2(nelmt * r + e) *
                                           d_basis2(r * nq2 + k);
                                }
                                d_out(nelmt * (k * nq1 * nq0 + j * nq0 + i) +
                                      e) = tmp;
                            }
                        }
                    }
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos2     = std::min(time_kokkos2, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1 * nq2,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos2[0]);

        // Kokkos 3 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            Kokkos::parallel_for(
                "Kokkos 1", Kokkos::TeamPolicy<>(nelmt, Kokkos::AUTO),
                KOKKOS_LAMBDA(const team_handle &team) {
                    // element index
                    unsigned int e = team.league_rank();

                    // direction 0
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nm2 * nm1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int q = tid % nm1;
                            const unsigned int r = (tid / nm1) % nm2;
                            const unsigned int i = tid / (nm1 * nm2);
                            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

                            T tmp = 0.0;
                            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                            {
                                tmp += d_in(nm0 * nm1 * nm2 * e + cnt_rqp) *
                                       d_basis0(p * nq0 + i);
                            }
                            d_wsp1(nq0 * nm1 * nm2 * e + tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 1
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq1 * nq0 * nm2),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int r = tid % nm2;
                            const unsigned int i = (tid / nm2) % nq0;
                            const unsigned int j = tid / (nm2 * nq0);
                            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

                            T tmp = 0.0;
                            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                            {
                                tmp += d_wsp1(nq0 * nm1 * nm2 * e + cnt_irq) *
                                       d_basis1(q * nq1 + j);
                            }
                            d_wsp2(nq0 * nq1 * nm2 * e + tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 2
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq2 * nq1 * nq0),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int i = tid % nq0;
                            const unsigned int j = (tid / nq0) % nq1;
                            const unsigned int k = tid / (nq1 * nq0);
                            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

                            T tmp = 0.0;
                            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                            {
                                tmp += d_wsp2(nq0 * nq1 * nm2 * e + cnt_jir) *
                                       d_basis2(r * nq2 + k);
                            }
                            d_out(nq0 * nq1 * nq2 * e + tid) = tmp;
                        });

                    team.team_barrier();
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos3     = std::min(time_kokkos3, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1 * nq2,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos3[0]);

        // Kokkos 4 - Shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            const unsigned int slevel = 0u;
            const unsigned int ssize  = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                       nm0 * nm1 * nm2 + nq0 * nm1 * nm2 +
                                       nq0 * nq1 * nm2;

            const unsigned int shmem_size = Kokkos::View<
                T *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);

            Kokkos::parallel_for(
                "Kokkos 2",
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
                    auto s_basis2 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(nm0 * nq0 + nm1 * nq1,
                                          nm0 * nq0 + nm1 * nq1 + nm2 * nq2));
                    auto s_wsp0 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(nm0 * nq0 + nm1 * nq1 + nm2 * nq2,
                                          nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                              nm0 * nm1 * nm2));
                    auto s_wsp1 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(
                            nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nm0 * nm1 * nm2,
                            nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                nm0 * nm1 * nm2 + nq0 * nm1 * nm2));
                    auto s_wsp2 = Kokkos::subview(
                        scratch,
                        Kokkos::make_pair(nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                              nm0 * nm1 * nm2 + nq0 * nm1 * nm2,
                                          ssize));

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm0 * nq0),
                        [&](const unsigned int tid)
                        { s_basis0(tid) = d_basis0(tid); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm1 * nq1),
                        [&](const unsigned int tid)
                        { s_basis1(tid) = d_basis1(tid); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 * nq2),
                        [&](const unsigned int tid)
                        { s_basis2(tid) = d_basis2(tid); });

                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nm2 * nm1 * nm0),
                        [&](const unsigned int &tid)
                        { s_wsp0(tid) = d_in(tid); });

                    team.team_barrier();

                    // direction 0
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq0 * nm2 * nm1),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int q = tid % nm1;
                            const unsigned int r = (tid / nm1) % nm2;
                            const unsigned int i = tid / (nm1 * nm2);
                            unsigned int cnt_rqp = nm1 * nm0 * r + nm0 * q;

                            T tmp = 0.0;
                            for (unsigned int p = 0u; p < nm0; ++p, ++cnt_rqp)
                            {
                                tmp += s_wsp0(cnt_rqp) * s_basis0(p * nq0 + i);
                            }
                            s_wsp1(tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 1
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq1 * nq0 * nm2),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int r = tid % nm2;
                            const unsigned int i = (tid / nm2) % nq0;
                            const unsigned int j = tid / (nm2 * nq0);
                            unsigned int cnt_irq = nm1 * nm2 * i + nm1 * r;

                            T tmp = 0.0;
                            for (unsigned int q = 0u; q < nm1; ++q, ++cnt_irq)
                            {
                                tmp += s_wsp1(cnt_irq) * s_basis1(q * nq1 + j);
                            }
                            s_wsp2(tid) = tmp;
                        });

                    team.team_barrier();

                    // direction 2
                    Kokkos::parallel_for(
                        Kokkos::TeamThreadRange(team, nq2 * nq1 * nq0),
                        [&](const unsigned int &tid)
                        {
                            const unsigned int i = tid % nq0;
                            const unsigned int j = (tid / nq0) % nq1;
                            const unsigned int k = tid / (nq1 * nq0);
                            unsigned int cnt_jir = nq0 * nm2 * j + nm2 * i;

                            T tmp = 0.0;
                            for (unsigned int r = 0u; r < nm2; ++r, ++cnt_jir)
                            {
                                tmp += s_wsp2(cnt_jir) * s_basis2(r * nq2 + k);
                            }
                            d_out(nq0 * nq1 * nq2 * e + tid) = tmp;
                        });

                    team.team_barrier();
                });
            Kokkos::fence();
            time.stop();
            const double t_w = time.elapsedSeconds();
            time_kokkos4     = std::min(time_kokkos4, t_w);
        }
        Kokkos::parallel_reduce(
            nelmt * nq0 * nq1 * nq2,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            result_kokkos4[0]);
    }

    // cuBLAS kernels
    double time_cublas = std::numeric_limits<double>::max();
    std::vector<T> result_cublas(1);
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        std::vector<T> h_in(nelmt * nm0 * nm1 * nm2);
        std::vector<T> h_out(nelmt * nq0 * nq1 * nq2);
        std::vector<T> h_basis0(nm0 * nq0);
        std::vector<T> h_basis1(nm1 * nq1);
        std::vector<T> h_basis2(nm2 * nq2);
        for (unsigned int e = 0u; e < nelmt; e++)
        {
            for (unsigned int p = 0u; p < nm0; p++)
            {
                for (unsigned int q = 0u; q < nm1; q++)
                {
                    for (unsigned int r = 0u; r < nm2; r++)
                    {
                        h_in[e * nm0 * nm1 * nm2 + p * nm1 * nm2 + q * nm2 +
                             r] = sin((T)(p * nm1 * nm2 + q * nm2 + r + 1u));
                    }
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
        for (unsigned int r = 0u; r < nm2; r++)
        {
            for (unsigned int k = 0u; k < nq2; k++)
            {
                h_basis2[r * nq2 + k] = std::cos((T)(r * nq2 + k));
            }
        }
        T *d_in, *d_out, *d_basis0, *d_basis1, *d_basis2, *d_wsp1, *d_wsp2;
        cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T));
        cudaMalloc(&d_out, nelmt * nq0 * nq1 * nq2 * sizeof(T));
        cudaMalloc(&d_wsp1, nelmt * nm0 * nm1 * nq2 * sizeof(T));
        cudaMalloc(&d_wsp2, nelmt * nm0 * nq1 * nq2 * sizeof(T));
        cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T));
        cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T));
        cudaMalloc(&d_basis2, nm2 * nq2 * sizeof(T));
        cudaMemcpy(d_in, &h_in[0], nelmt * nm0 * nm1 * nm2 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis0, &h_basis0[0], nm0 * nq0 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis1, &h_basis1[0], nm1 * nq1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis2, &h_basis2[0], nm2 * nq2 * sizeof(T),
                   cudaMemcpyHostToDevice);
        T alpha = 1.0;
        T beta  = 0.0;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            if constexpr (std::is_same_v<T, float>)
            {
                cublasSgemmStridedBatched(
                    handle, CUBLAS_OP_N, CUBLAS_OP_T, nq2, nm0 * nm1, nm2,
                    &alpha, d_basis2, nq2, 0, d_in, nm0 * nm1, nm0 * nm1 * nm2,
                    &beta, d_wsp1, nq2 * nelmt, nq2, nelmt);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq1,
                            nq2 * nelmt * nm0, nm1, &alpha, d_basis1, nq1,
                            d_wsp1, nq2 * nelmt * nm0, &beta, d_wsp2, nq1);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq0,
                            nq1 * nq2 * nelmt, nm0, &alpha, d_basis0, nq0,
                            d_wsp2, nq1 * nq2 * nelmt, &beta, d_out, nq0);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                cublasDgemmStridedBatched(
                    handle, CUBLAS_OP_N, CUBLAS_OP_T, nq2, nm0 * nm1, nm2,
                    &alpha, d_basis2, nq2, 0, d_in, nm0 * nm1, nm0 * nm1 * nm2,
                    &beta, d_wsp1, nq2 * nelmt, nq2, nelmt);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq1,
                            nq2 * nelmt * nm0, nm1, &alpha, d_basis1, nq1,
                            d_wsp1, nq2 * nelmt * nm0, &beta, d_wsp2, nq1);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, nq0,
                            nq1 * nq2 * nelmt, nm0, &alpha, d_basis0, nq0,
                            d_wsp2, nq1 * nq2 * nelmt, &beta, d_out, nq0);
            }
            cudaDeviceSynchronize();
            time.stop();
            time_cublas = std::min(time_cublas, time.elapsedSeconds());
        }
        result_cublas[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_basis2);
        cudaFree(d_wsp1);
        cudaFree(d_wsp2);
        cudaFree(d_in);
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
    std::vector<T> result_cuda1(1);
    std::vector<T> result_cuda2(1);
    std::vector<T> result_cuda3(1);
    std::vector<T> result_cuda4(1);
    std::vector<T> result_cuda5(1);
    std::vector<T> result_cuda6(1);
    {
        const unsigned int threads = 256;
        const unsigned int blocks  = nelmt / 4;
        std::vector<T> h_in(nelmt * nm0 * nm1 * nm2);
        std::vector<T> h_in_coa(nelmt * nm0 * nm1 * nm2);
        std::vector<T> h_out(nelmt * nq0 * nq1 * nq2);
        std::vector<T> h_basis0(nm0 * nq0);
        std::vector<T> h_basis1(nm1 * nq1);
        std::vector<T> h_basis2(nm2 * nq2);
        for (unsigned int e = 0u; e < nelmt; e++)
        {
            for (unsigned int p = 0u; p < nm0; p++)
            {
                for (unsigned int q = 0u; q < nm1; q++)
                {
                    for (unsigned int r = 0u; r < nm2; r++)
                    {
                        h_in[e * nm0 * nm1 * nm2 + p * nm1 * nm2 + q * nm2 +
                             r] = sin((T)(p * nm1 * nm2 + q * nm2 + r + 1u));
                        h_in_coa[(p * nm1 * nm2 + q * nm2 + r) * nelmt + e] =
                            sin((T)(p * nm1 * nm2 + q * nm2 + r + 1u));
                    }
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
        for (unsigned int r = 0u; r < nm2; r++)
        {
            for (unsigned int k = 0u; k < nq2; k++)
            {
                h_basis2[r * nq2 + k] = std::cos((T)(r * nq2 + k));
            }
        }
        T *d_in, *d_in_coa, *d_wsp0, *d_wsp1, *d_wsp2, *d_wsp3, *d_out,
            *d_basis0, *d_basis1, *d_basis2;
        cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T));
        cudaMalloc(&d_in_coa, nelmt * nm0 * nm1 * nm2 * sizeof(T));
        cudaMalloc(&d_wsp0, nelmt * nm1 * nm2 * sizeof(T));
        cudaMalloc(&d_wsp1, nelmt * nm2 * sizeof(T));
        cudaMalloc(&d_wsp2, nelmt * nq0 * nm1 * nm2 * sizeof(T));
        cudaMalloc(&d_wsp3, nelmt * nq0 * nq1 * nm2 * sizeof(T));
        cudaMalloc(&d_out, nelmt * nq0 * nq1 * nq2 * sizeof(T));
        cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T));
        cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T));
        cudaMalloc(&d_basis2, nm2 * nq2 * sizeof(T));
        cudaMemcpy(d_in, &h_in[0], nelmt * nm0 * nm1 * nm2 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_coa, &h_in_coa[0], nelmt * nm0 * nm1 * nm2 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis0, &h_basis0[0], nm0 * nq0 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis1, &h_basis1[0], nm1 * nq1 * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_basis2, &h_basis2[0], nm2 * nq2 * sizeof(T),
                   cudaMemcpyHostToDevice);

        // Cuda 1 - Non coalesce
        const unsigned int ssize1 = nm0 * nq0 + nm1 * nq1 + nm2 * nq2;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel<T, true, false>
                <<<(nelmt + threads - 1u) / threads, threads,
                   sizeof(T) * ssize1>>>(nm0, nm1, nm2, nm0 * nm1 * nm2, nq0,
                                         nq1, nq2, nelmt, d_basis0, d_basis1,
                                         d_basis2, d_in, d_wsp0, d_wsp1, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda1 = std::min(time_cuda1, time.elapsedSeconds());
        }
        result_cuda1[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 2 - Coalesce
        const unsigned int ssize2 = nm0 * nq0 + nm1 * nq1 + nm2 * nq2;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel<T, true, true><<<(nelmt + threads - 1u) / threads,
                                               threads, sizeof(T) * ssize2>>>(
                nm0, nm1, nm2, nm0 * nm1 * nm2, nq0, nq1, nq2, nelmt, d_basis0,
                d_basis1, d_basis2, d_in_coa, d_wsp0, d_wsp1, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda2 = std::min(time_cuda2, time.elapsedSeconds());
        }
        result_cuda2[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 3 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel_QP<<<
                blocks,
                dim3(std::min(nq0, 32u), std::min(nq1, 32u),
                     std::min(nq2, std::max(1u, 1024u / (nq0 * nq1))))>>>(
                nm0, nm1, nm2, nm0 * nm1 * nm2, nq0, nq1, nq2, nelmt, d_basis0,
                d_basis1, d_basis2, d_in, d_wsp2, d_wsp3, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda3 = std::min(time_cuda3, time.elapsedSeconds());
        }
        result_cuda3[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 4 - Shared memory
        const unsigned int ssize4 = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                    nm0 * nm1 * nm2 + nm0 * nm1 * nq2 +
                                    nm0 * nq1 * nq2;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel_QP<<<
                blocks,
                dim3(std::min(nq0, 32u), std::min(nq1, 32u),
                     std::min(nq2, std::max(1u, 1024u / (nq0 * nq1)))),
                sizeof(T) * ssize4>>>(nm0, nm1, nm2, nm0 * nm1 * nm2, nq0, nq1,
                                      nq2, nelmt, d_basis0, d_basis1, d_basis2,
                                      d_in, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda4 = std::min(time_cuda4, time.elapsedSeconds());
        }
        result_cuda4[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 5 - No shared memory
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel_QP_1D<<<blocks,
                                      std::min(nq0 * nq1 * nq2, threads)>>>(
                nm0, nm1, nm2, nm0 * nm1 * nm2, nq0, nq1, nq2, nelmt, d_basis0,
                d_basis1, d_basis2, d_in, d_wsp2, d_wsp3, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda5 = std::min(time_cuda5, time.elapsedSeconds());
        }
        result_cuda5[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());

        // Cuda 6 - Shared memory
        const unsigned int ssize6 = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 +
                                    nm0 * nm1 * nm2 + nm0 * nm1 * nq2 +
                                    nm0 * nq1 * nq2;
        for (unsigned int t = 0u; t < n_tests; ++t)
        {
            time.start();
            BwdTransHexKernel_QP_1D<<<blocks,
                                      std::min(nq0 * nq1 * nq2, threads),
                                      sizeof(T) * ssize6>>>(
                nm0, nm1, nm2, nm0 * nm1 * nm2, nq0, nq1, nq2, nelmt, d_basis0,
                d_basis1, d_basis2, d_in, d_out);
            cudaDeviceSynchronize();
            time.stop();
            time_cuda6 = std::min(time_cuda6, time.elapsedSeconds());
        }
        result_cuda6[0] = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            [] __device__(const T &x) { return x * x; }, (T)0.0,
            thrust::plus<T>());
        cudaFree(d_basis0);
        cudaFree(d_basis1);
        cudaFree(d_basis2);
        cudaFree(d_in);
        cudaFree(d_in_coa);
        cudaFree(d_wsp0);
        cudaFree(d_wsp1);
        cudaFree(d_wsp2);
        cudaFree(d_wsp3);
        cudaFree(d_out);
    }

    // Display results
    std::cout << std::setprecision(10);
    std::cout
        << "nelmt " << nelmt
        << " Case: Kokkos (Uncoales) Kokkos (Coales) Kokkos (QP)   Kokkos "
           "(QP/Shared) cuBLAS       Cuda "
           "(Uncoales)"
           " Cuda (Coales)    Cuda (QP)      Cuda (QP/Shared)  Cuda (QP-1D)  "
           " Cuda (QP-1D/Shared)"
        << std::endl;
    std::cout << "nelmt " << nelmt << " norm: " << std::sqrt(result_kokkos1[0])
              << "     " << std::sqrt(result_kokkos2[0]) << "     "
              << std::sqrt(result_kokkos3[0]) << "     "
              << std::sqrt(result_kokkos4[0]) << "     "
              << std::sqrt(result_cublas[0]) << "     "
              << std::sqrt(result_cuda1[0]) << "     "
              << std::sqrt(result_cuda2[0]) << "     "
              << std::sqrt(result_cuda3[0]) << "     "
              << std::sqrt(result_cuda4[0]) << "     "
              << std::sqrt(result_cuda5[0]) << "     "
              << std::sqrt(result_cuda6[0]) << std::endl;

    std::cout << "nelmt " << nelmt << " GB/s: "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_kokkos1
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_kokkos2
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_kokkos3
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_kokkos4
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cublas
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda1
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda2
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda3
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda4
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda5
              << "     "
              << sizeof(T) * 1.0e-9 * nelmt *
                     (nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / time_cuda6
              << std::endl;
}

int main(int argc, char **argv)
{
    unsigned int nq0 = (argc > 1) ? atoi(argv[1]) : 8u;
    unsigned int nq1 = (argc > 2) ? atoi(argv[2]) : 8u;
    unsigned int nq2 = (argc > 3) ? atoi(argv[3]) : 8u;

    std::cout << "--------------------------------" << std::endl;
    std::cout << "Benchmark05 : BwdTrans (3D)     " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "BwdTrans (NQ = " << nq0 << ", " << nq1 << ", " << nq2 << ")" << std::endl;
    Kokkos::initialize(argc, argv);
    for (unsigned int size = 2 << 6; size < 2 << 20; size <<= 1)
    {
        run_test<float>(size, nq0, nq1, nq2);
    }
    Kokkos::finalize();
}
