#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/utils/Traits.h"
#include "noa/gpu/cuda/utils/Warp.cuh"

// TODO CUDA's cub seems to have some load and store functions. Surely some of them can be use here.

namespace noa::cuda::utils::block {
    // Synchronizes the block.
    NOA_FD void synchronize() {
        __syncthreads();
    }

    // Retrieves the  dynamically allocated per-block shared memory.
    // For using dynamically-sized (i.e. "extern" with unspecified-size array) shared memory in templated
    // kernels, this kind of utility is necessary to avoid errors with non-basic types (e.g. cfloat_t).
    // Also, since the documentation is unclear about the alignment and whether it comes with any alignment
    // guarantees other than the alignment of the type used in the declaration (thus whether the
    // __align__ attribute has any effect on shared memory), use double2 to ensure 16-byte alignment,
    // then cast to the desired type. See https://stackoverflow.com/questions/27570552.
    template<typename T>
    NOA_FD T* dynamicSharedResource() {
        static_assert(alignof(T) <= alignof(double2));
        extern __shared__ double2 buffer_align16[];
        return reinterpret_cast<T*>(buffer_align16);
    }

    // Each thread loads ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    // BLOCK_SIZE:          The number of threads per block. It assumes a 1D contiguous block.
    // ELEMENTS_PER_THREAD: The number of elements to load, per thread.
    // VEC_SIZE:            Size, in elements, to load at the same time. If 1, there's no vectorization.
    // per_block_input:     Contiguous input array to load from. This is per block, and should point at
    //                      the first element of the block's work space. It should be aligned to VEC_SIZE.
    // per_thread_output:   Per thread output array. At least ELEMENTS_PER_THREAD elements.
    // tidx:                Thread index in the 1D block. Usually threadIdx.x.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE, typename T>
    NOA_ID void vectorizedLoad(const T* __restrict__ per_block_input, T* __restrict__ per_thread_output, uint tidx) {
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

    // Each thread stores ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    // BLOCK_SIZE:          The number of threads per block. It assumes a 1D contiguous block.
    // ELEMENTS_PER_THREAD: The number of elements to store, per thread.
    // VEC_SIZE:            Size, in elements, to store at the same time. If 1, there's no vectorization.
    // per_thread_input:    Per thread input array to store. At least ELEMENTS_PER_THREAD elements.
    // per_block_output:    Contiguous output array to write into. This is per block, and should point at
    //                      the first element of the block's work space. It should be aligned to VEC_SIZE.
    // tidx:                Thread index in the 1D block. Usually threadIdx.x.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE, typename T>
    NOA_ID void vectorizedStore(const T* __restrict__ per_thread_input, T* __restrict__ per_block_output, uint tidx) {
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

    // Reduces BLOCK_SIZE elements from s_data. Returns reduced value in tid 0 (undefined in other threads).
    // s_data:      Shared memory to reduce. Should be at least BLOCK_SIZE elements. It is overwritten.
    // tid:         Thread index. From 0 to BLOCK_SIZE - 1.
    // reduce_op:   Reduction operator.
    template<uint BLOCK_SIZE, typename T, typename reduce_op_t>
    NOA_ID T reduceShared1D(T* s_data, int tid, reduce_op_t reduce_op) {
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
            value = utils::warp::reduce(s_data[tid], reduce_op);
        return value;
    }

    // Find the best element within BLOCK_SIZE elements according to an update operator.
    // Returns the reduced {value, offset} in tid 0 (undefined in other threads).
    // s_values:    Shared memory with the input values to search. At least BLOCK_SIZE elements. It is overwritten.
    // s_offsets:   Shared memory with the corresponding indexes. At least BLOCK_SIZE elements. It is overwritten.
    // tid:         Thread index. From 0 to BLOCK_SIZE - 1.
    // find_op:     Find operator: ``operator()(current, candidate) -> reduced``.
    template<uint BLOCK_SIZE, typename value_t, typename offset_t, typename find_op_t>
    NOA_ID Pair<value_t, offset_t> findShared1D(value_t* __restrict__ s_values,
                                                offset_t* __restrict__ s_offsets, uint tid, find_op_t find_op) {
        static_assert(BLOCK_SIZE == 1024 || BLOCK_SIZE == 512 ||
                      BLOCK_SIZE == 256 || BLOCK_SIZE == 128 || BLOCK_SIZE == 64);
        using pair_t = Pair<value_t, offset_t>;
        value_t* s_values_tid = s_values + tid;
        offset_t* s_offsets_tid = s_offsets + tid;
        pair_t current{*s_values_tid, *s_offsets_tid};

        #pragma unroll
        for (uint SIZE = BLOCK_SIZE / 2; SIZE >= 32; SIZE /= 2) {
            if (tid < SIZE) {
                current = find_op(current, pair_t{s_values_tid[SIZE], s_offsets_tid[SIZE]});
                *s_values_tid = current.first;
                *s_offsets_tid = current.second;
            }
            synchronize();
        }
        pair_t reduced;
        if (tid < 32)
            reduced = utils::warp::find(s_values[tid], s_offsets[tid], find_op);
        return reduced;
    }

    // Reduces min(BLOCK_SIZE * ELEMENTS_PER_THREAD, elements) elements from input.
    // BLOCK_SIZE:          Number of threads in the dimension to reduce.
    // ELEMENTS_PER_THREAD: Number of elements to load, for each thread. Should be >= VEC_SIZE
    // VEC_SIZE:            Vector size. Either 4, 2, or 1. If 1, a sequential load is used.
    // input:               Input array (usually pointing at global memory) to reduce.
    //                      It should start at the first element to reduce.
    // stride:              Stride between each element. This is ignored if VEC_SIZE >= 1.
    // elements:            Maximum number of elements that can be reduced.
    //                      The function tries to BLOCK_SIZE * ELEMENTS_PER_THREAD elements, but will
    //                      stop if it reaches input + elements first.
    // transform_op:        Transform operator, op(value_t) -> X, to apply on the input before reduction.
    //                      Its output is explicitly cast to reduce_value_t.
    // reduce_op:           Reduction operator: op(reduce_value_t, reduce_value_t) -> reduce_value_t.
    // reduced:             Per-thread left-hand side argument of reduce_op.
    //                      It is updated with the final reduced value.
    // tidx:                Thread index in the dimension to reduce.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE,
             typename value_t, typename reduce_value_t, typename transform_op_t, typename reduce_op_t>
    NOA_ID void reduceUnaryGlobal1D(const value_t* __restrict__ input, [[maybe_unused]] uint stride, uint elements,
                                    transform_op_t transform_op, reduce_op_t reduce_op,
                                    reduce_value_t* __restrict__ reduced, uint tidx) {
        if constexpr (VEC_SIZE > 1) {
            (void) stride; // assume contiguous
            if (elements < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint tid = BLOCK_SIZE * i + tidx;
                    if (tid < elements) {
                        const auto transformed = transform_op(input[tid]);
                        *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(transformed));
                    }
                }
            } else {
                value_t args[ELEMENTS_PER_THREAD];
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(input, args, tidx);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const auto transformed = transform_op(args[i]);
                    *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(transformed));
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint tid = BLOCK_SIZE * i + tidx;
                if (tid < elements) {
                    const auto transformed = transform_op(input[tid * stride]);
                    *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(transformed));
                }
            }
        }
    }

    // Reduces min(BLOCK_SIZE * ELEMENTS_PER_THREAD, elements) elements from inputs.
    // BLOCK_SIZE:          Number of threads in the dimension to reduce.
    // ELEMENTS_PER_THREAD: Number of elements to load, for each thread. Should be >= VEC_SIZE
    // VEC_SIZE:            Vector size. Either 4, 2, or 1. If 1, a sequential load is used.
    // lhs, rhs:            Left and right-hand side input arrays (usually pointing at global memory) to reduce.
    //                      Should start at the first element to reduce.
    // lhs_stride:          Stride between each element in lhs. This is ignored if VEC_SIZE >= 1.
    // rhs_stride:          Stride between each element in rhs. This is ignored if VEC_SIZE >= 1.
    // elements:            Maximum number of elements that can be reduced.
    //                      The function tries to BLOCK_SIZE * ELEMENTS_PER_THREAD elements, but will
    //                      stop if it reaches the limit set by elements.
    // transform_op_lhs:    Transform operator, op(lhs_value_t) -> Xl, to apply on lhs before combination.
    // transform_op_rhs:    Transform operator, op(rhs_value_t) -> Xr, to apply on rhs before combination.
    // combine_op:          Combine operator, op(Xl, Xr), to apply on the left and right transformed value before
    //                      reduction. The output value of this operator is cast to reduce_value_t.
    // reduce_op:           Reduction operator: op(reduce_value_t, reduce_value_t) -> reduce_value_t.
    // reduced:             Per-thread left-hand side argument of reduce_op.
    //                      It is updated with the final reduced value.
    // tidx:                Thread index in the dimension to reduce.
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE,
             typename lhs_value_t, typename rhs_value_t, typename reduce_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t,
             typename combine_op_t, typename reduce_op_t>
    NOA_ID void reduceBinaryGlobal1D(const lhs_value_t* lhs, uint lhs_stride,
                                     const rhs_value_t* rhs, uint rhs_stride, uint elements,
                                     transform_op_lhs_t lhs_transform_op, transform_op_rhs_t rhs_transform_op,
                                     combine_op_t combine_op, reduce_op_t reduce_op,
                                     reduce_value_t* reduced, uint tidx) {
        if constexpr (VEC_SIZE > 1) {
            (void) lhs_stride; // assume contiguous
            (void) rhs_stride; // assume contiguous
            if (elements < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint tid = BLOCK_SIZE * i + tidx;
                    if (tid < elements) {
                        const auto combined = combine_op(lhs_transform_op(lhs[tid]), rhs_transform_op(rhs[tid]));
                        *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(combined));
                    }
                }
            } else {
                lhs_value_t lhs_args[ELEMENTS_PER_THREAD];
                rhs_value_t rhs_args[ELEMENTS_PER_THREAD];
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(lhs, lhs_args, tidx);
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(rhs, rhs_args, tidx);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const auto combined = combine_op(lhs_transform_op(lhs_args[i]), rhs_transform_op(rhs_args[i]));
                    *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(combined));
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint tid = BLOCK_SIZE * i + tidx;
                if (tid < elements) {
                    const auto combined = combine_op(lhs_transform_op(lhs[tid * lhs_stride]),
                                                     rhs_transform_op(rhs[tid * rhs_stride]));
                    *reduced = reduce_op(*reduced, static_cast<reduce_value_t>(combined));
                }
            }
        }
    }

    // TODO Add documentation
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE,
            typename value_t, typename transformed_t, typename offset_t, typename transform_op_t, typename find_op_t>
    NOA_ID void findGlobal1D(const value_t* __restrict__ input, uint gidx, uint tidx, uint stride, uint elements,
                             transform_op_t transform_op, find_op_t find_op,
                             Pair<transformed_t, offset_t>* __restrict__ reduced) {
        using pair_t = Pair<transformed_t, offset_t>;
        const uint remaining = elements - gidx;
        if constexpr (VEC_SIZE > 1) {
            input += gidx;
            (void) stride; // assume contiguous
            if (remaining < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint tid = BLOCK_SIZE * i + tidx;
                    if (tid < remaining) {
                        const pair_t candidate{static_cast<transformed_t>(transform_op(input[tid])),
                                               static_cast<offset_t>(gidx + tid)};
                        *reduced = find_op(*reduced, candidate);
                    }
                }
            } else {
                value_t args[ELEMENTS_PER_THREAD];
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(input, args, tidx);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint offset = gidx + (tidx + (i / VEC_SIZE) * BLOCK_SIZE) * VEC_SIZE + i % VEC_SIZE;
                    const pair_t candidate{static_cast<transformed_t>(transform_op(args[i])),
                                           static_cast<offset_t>(offset)};
                    *reduced = find_op(*reduced, candidate);
                }
            }
        } else {
            input += gidx * stride;
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint tid = BLOCK_SIZE * i + tidx;
                if (tid < remaining) {
                    const pair_t candidate{static_cast<transformed_t>(transform_op(input[tid * stride])),
                                           static_cast<offset_t>((gidx + tid) * stride)};
                    *reduced = find_op(*reduced, candidate);
                }
            }
        }
    }

    // TODO Add documentation
    template<uint BLOCK_SIZE, uint ELEMENTS_PER_THREAD, uint VEC_SIZE,
            typename value_t, typename transformed_t, typename offset_t, typename transform_op_t, typename find_op_t>
    NOA_ID void findGlobal1D(const value_t* __restrict__ values, const offset_t* __restrict__ offsets,
                             uint gidx, uint tidx, uint elements,
                             transform_op_t transform_op, find_op_t find_op,
                             Pair<transformed_t, offset_t>* __restrict__ reduced) {
        using pair_t = Pair<transformed_t, offset_t>;
        const uint remaining = elements - gidx;
        values += gidx;
        offsets += gidx;
        if constexpr (VEC_SIZE > 1) {
            if (remaining < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint tid = BLOCK_SIZE * i + tidx;
                    if (tid < remaining) {
                        const pair_t candidate{static_cast<transformed_t>(transform_op(values[tid])),
                                               static_cast<offset_t>(offsets[tid])};
                        *reduced = find_op(*reduced, candidate);
                    }
                }
            } else {
                value_t args[ELEMENTS_PER_THREAD];
                offset_t offs[ELEMENTS_PER_THREAD];
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, args, tidx);
                utils::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(offsets, offs, tidx);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const pair_t candidate{static_cast<transformed_t>(transform_op(args[i])),
                                           static_cast<offset_t>(offs[i])};
                    *reduced = find_op(*reduced, candidate);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint tid = BLOCK_SIZE * i + tidx;
                if (tid < remaining) {
                    const pair_t candidate{static_cast<transformed_t>(transform_op(values[tid])),
                                           static_cast<offset_t>(offsets[tid])};
                    *reduced = find_op(*reduced, candidate);
                }
            }
        }
    }
}
