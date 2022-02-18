/// \file noa/gpu/cuda/util/Block.cuh
/// \brief Block utilities.
/// \author Thomas - ffyr2w
/// \date 13 Feb 2022
#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/util/Warp.cuh"

namespace noa::cuda::util::block {
    /// Synchronizes the block.
    NOA_FD void synchronize() {
        __syncthreads();
    }

    /// Retrieves the  dynamically allocated per-block shared memory.
    /// \details For using dynamically-sized (i.e. "extern" with unspecified-size array) shared memory in templated
    ///          kernels, this kind of utility is necessary to avoid errors with non-basic types (e.g. cfloat_t).
    ///          Also, since the documentation is unclear about the alignment and whether it comes with any alignment
    ///          guarantees other than the alignment of the type used in the declaration (thus whether or not the
    ///          __align__ attribute has any effect on shared memory), use double2 to ensure 16-byte alignment,
    ///          then cast to the desired type. See https://stackoverflow.com/questions/27570552.
    template<typename T>
    NOA_FD T* dynamicSharedResource() {
        static_assert(alignof(T) <= alignof(double2));
        extern __shared__ double2 buffer_align16[];
        return reinterpret_cast<T*>(buffer_align16);
    }

    /// Each thread loads \p ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    /// \tparam BLOCK_SIZE              The number of threads per block. It assumes a 1D contiguous block.
    /// \tparam ELEMENTS_PER_THREAD     The number of elements to load, per thread.
    /// \tparam VEC_SIZE                Size, in \p T elements, to load at the same time.
    ///                                 If 1, there's no vectorization.
    /// \param[in] per_block_input      Contiguous input array to load from. This is per block, and should point at
    ///                                 the first element of the block's work space. It should be aligned to VEC_SIZE.
    /// \param[out] per_thread_output   Per thread output array. At least ELEMENTS_PER_THREAD elements.
    /// \param tidx                     Thread index in the 1D block. Usually threadIdx.x.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE, typename T, typename U>
    NOA_ID void vectorizedLoad(const T* per_block_input, U* per_thread_output, uint tidx) {
        static_assert(ELEMENTS_PER_THREAD >= VEC_SIZE); // TODO This could be improved...
        using vec_t = traits::aligned_vector_t<T, VEC_SIZE>;
        const auto* from = reinterpret_cast<const vec_t*>(per_block_input);
        constexpr int COUNT = ELEMENTS_PER_THREAD / VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < COUNT; ++i) {
            vec_t v = from[i * BLOCK_SIZE + tidx];
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; ++j)
                per_thread_output[VEC_SIZE * i + j] = v.val[j];
        }
    }

    /// Each thread stores \p ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    /// \tparam BLOCK_SIZE              The number of threads per block. It assumes a 1D contiguous block.
    /// \tparam ELEMENTS_PER_THREAD     The number of elements to store, per thread.
    /// \tparam VEC_SIZE                Size, in \p T elements, to store at the same time.
    ///                                 If 1, there's no vectorization.
    /// \param[in] per_thread_input     Per thread input array to store. At least ELEMENTS_PER_THREAD elements.
    /// \param[out] per_block_output    Contiguous output array to write into. This is per block, and should point at
    ///                                 the first element of the block's work space. It should be aligned to VEC_SIZE.
    /// \param tidx                     Thread index in the 1D block. Usually threadIdx.x.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE, typename T>
    NOA_ID void vectorizedStore(const T* per_thread_input, T* per_block_output, uint tidx) {
        static_assert(ELEMENTS_PER_THREAD >= VEC_SIZE); // TODO This could be improved...
        using vec_t = traits::aligned_vector_t<T, VEC_SIZE>;
        auto* to = reinterpret_cast<vec_t*>(per_block_output);
        constexpr int COUNT = ELEMENTS_PER_THREAD / VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < COUNT; i++) {
            vec_t v;
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++)
                v.val[j] = per_thread_input[VEC_SIZE * i + j];
            to[i * BLOCK_SIZE + tidx] = v;
        }
    }

    /// Reduces BLOCK_SIZE elements from s_data..
    /// \param[in,out] s_data   Shared memory to reduce. Should be at least BLOCK_SIZE elements.
    ///                         The state in which it is left is undefined.
    /// \param tid              Thread index. From 0 to BLOCK_SIZE - 1.
    /// \param reduce_op        Reduction operator.
    /// \return Reduced value in tid 0 (undefined in other threads).
    template<uint BLOCK_SIZE, typename T, typename ReduceOp>
    NOA_ID T reduceShared1D(T* s_data, int tid, ReduceOp reduce_op) {
        static_assert(BLOCK_SIZE == 1024 || BLOCK_SIZE == 512 ||
                      BLOCK_SIZE == 256 || BLOCK_SIZE == 128 || BLOCK_SIZE == 64);
        T* s_data_tid = s_data + tid;

        #pragma unroll
        for (uint SIZE = BLOCK_SIZE / 2; SIZE >= 32; SIZE /= 2) {
            if (tid < SIZE)
                *s_data_tid = reduce_op(*s_data_tid, s_data_tid[SIZE]);
            synchronize();
        }
        T value;
        if (tid < 32)
            value = util::warp::reduce(s_data, tid, reduce_op);
        return value;
    }

    /// Reduces min(BLOCK_SIZE * ELEMENTS_PER_THREAD, elements) elements from input.
    /// \tparam BLOCK_SIZE          Number of threads in the dimension to reduce.
    /// \tparam ELEMENTS_PER_THREAD Number of elements to load, for each thread. Should be >= \p VEC_SIZE
    /// \tparam VEC_SIZE            Vector size. Either 4, 2, or 1. If 1, a sequential load is used.
    /// \param[in] input            Input array (usually pointing at global memory) to reduce.
    ///                             It should start at the first element to reduce.
    /// \param stride               Stride between each element. This is ignored if \p VEC_SIZE >= 1.
    /// \param elements             Maximum number of elements that can be reduced.
    ///                             The function tries to BLOCK_SIZE * ELEMENTS_PER_THREAD elements, but will
    ///                             stop if it reaches \p input + \p elements first.
    /// \param transform_op         Transform operator, op(\p T) -> \p U, to apply on the input before reduction.
    /// \param reduce_op            Reduction operator: op(\p U, \p U) -> \p U.
    /// \param reduced              Per-thread left-hand side argument of \p reduce_op.
    ///                             It is updated with the final reduced value.
    /// \param tidx                 Thread index in the dimension to reduce.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE,
             typename T, typename U, typename TransformOp, typename ReduceOp>
    NOA_ID void reduceGlobal1D(const T* input, [[maybe_unused]] uint stride, uint elements,
                               TransformOp transform_op, ReduceOp reduce_op, U* reduced, uint tidx) {
        if constexpr (VEC_SIZE > 1) {
            (void) stride; // assume contiguous
            if (elements < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint tid = BLOCK_SIZE * i + tidx;
                    if (tid < elements) {
                        const U transformed = static_cast<U>(transform_op(input[tid]));
                        *reduced = reduce_op(*reduced, transformed);
                    }
                }
            } else {
                T args[ELEMENTS_PER_THREAD];
                util::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(input, args, tidx);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const U transformed = static_cast<U>(transform_op(args[i]));
                    *reduced = reduce_op(*reduced, transformed);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint tid = BLOCK_SIZE * i + tidx;
                if (tid < elements) {
                    const U transformed = static_cast<U>(transform_op(input[tid * stride]));
                    *reduced = reduce_op(*reduced, transformed);
                }
            }
        }
    }
}
