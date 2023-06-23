#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Warp.cuh"

// TODO CUDA's cub seems to have some load and store functions. Surely some of them can be used here.

namespace noa::cuda::utils {
    // Synchronizes the block.
    NOA_FD void block_synchronize() {
        __syncthreads();
    }

    // Retrieves the dynamically allocated per-block shared memory.
    // For using dynamically-sized (i.e. "extern" with unspecified-size array) shared memory in templated
    // kernels, this kind of utility is necessary to avoid errors with non-basic types (e.g. c32).
    // Also, since the documentation is unclear about the alignment and whether it comes with any alignment
    // guarantees other than the alignment of the type used in the declaration (thus whether the
    // __align__ attribute has any effect on shared memory), use double2 to ensure 16-byte alignment,
    // then cast to the desired type. See https://stackoverflow.com/questions/27570552.
    template<typename T>
    NOA_FD T* block_dynamic_shared_resource() {
        static_assert(alignof(T) <= alignof(double2));
        extern __shared__ double2 buffer_align16[];
        return reinterpret_cast<T*>(buffer_align16);
    }

    template<typename T>
    struct proclaim_uninitialized_type { using type = T; };
    template<>
    struct proclaim_uninitialized_type<f16> { using type = ::half; };
    template<>
    struct proclaim_uninitialized_type<c16> { using type = ::half2; };
    template<>
    struct proclaim_uninitialized_type<c32> { using type = ::float2; };
    template<>
    struct proclaim_uninitialized_type<c64> { using type = ::double2; };

    template<typename Lhs, typename Rhs>
    struct proclaim_uninitialized_type<noa::Pair<Lhs, Rhs>> {
        using type = noa::Pair<typename proclaim_uninitialized_type<Lhs>::type,
                               typename proclaim_uninitialized_type<Rhs>::type>;
    };

    // Static initialization of shared variables is illegal in CUDA. Some types (e.g. f16) cannot be used with
    // the __shared__ attribute. This trait returns an equivalent type of T, i.e. same size and alignment,
    // meant to be used to declare static shared arrays/pointers. The returned type can be the same as T.
    // Once declared, this region of shared memory can be reinterpreted to T "safely". While these types are
    // very similar (again, same size, same alignment), we are likely in C++ undefined behavior.
    // However, this is CUDA C++ and I doubt this would cause any issue (they reinterpret pointers to very
    // different types quite often in their examples, so give me a break).
    // Update: nvcc 11.7 may support zero-initialization (which is what the default constructor for our types do),
    // so we may not need this anymore. I couldn't find it in the changelog though, but it does compile...
    template<typename T>
    struct uninitialized_type { using type = typename proclaim_uninitialized_type<T>::type; };
    template<typename T>
    using uninitialized_type_t = typename uninitialized_type<T>::type;

    // Each thread loads ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    // If ELEMENTS_PER_THREAD > VECTOR_SIZE, multiple reads per thread will be necessary.
    // BLOCK_SIZE:          The number of threads per block. It assumes a 1D contiguous block.
    // ELEMENTS_PER_THREAD: The number of elements to load, per thread.
    // VECTOR_SIZE:         Size, in elements, to load at the same time. If 1, there's no vectorization.
    // per_block_input:     Contiguous input array to load from. This is per block, and should point at
    //                      the first element of the block's work space. It should be aligned to VECTOR_SIZE.
    // per_thread_output:   Per thread output array. At least ELEMENTS_PER_THREAD elements.
    // per_thread_index:    Thread index in the 1D block. Usually threadIdx.x.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, i32 VECTOR_SIZE, typename Value>
    NOA_ID void block_load(
            const Value* __restrict__ per_block_input,
            Value* __restrict__ per_thread_output,
            i32 thread_index
    ) {
        // The input type is reinterpreted to this aligned vector. VECTOR_SIZE should be
        // correctly set so that the alignment of the input pointer is enough for this vector type.
        using aligned_vector_t = AlignedVector<Value, VECTOR_SIZE>;
        const auto* __restrict__ source = reinterpret_cast<const aligned_vector_t*>(per_block_input);
        NOA_ASSERT(!(reinterpret_cast<std::uintptr_t>(per_block_input) % alignof(aligned_vector_t)));

        // The elements that belong to the same vector are saved next to each other.
        // If we need more than one vectorized load, we offset the input by
        // the entire block size and offset the output by the vector size.
        static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
        static_assert(!(ELEMENTS_PER_THREAD % VECTOR_SIZE));
        constexpr i32 VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;
        #pragma unroll
        for (i32 i = 0; i < VECTORIZED_LOADS; ++i) {
            aligned_vector_t loaded_vector = source[i * BLOCK_SIZE + thread_index]; // vectorized load
            #pragma unroll
            for (i32 j = 0; j < VECTOR_SIZE; ++j)
                per_thread_output[VECTOR_SIZE * i + j] = loaded_vector.data[j];
        }
    }

    // Each thread stores ELEMENTS_PER_THREAD elements, using vectorized store instructions if possible.
    // If ELEMENTS_PER_THREAD > VECTOR_SIZE, multiple writes per thread will be necessary.
    // BLOCK_SIZE:          The number of threads per block. It assumes a 1D contiguous block.
    // ELEMENTS_PER_THREAD: The number of elements to store, per thread.
    // VECTOR_SIZE:         Size, in elements, to store at the same time. If 1, there's no vectorization.
    // per_thread_input:    Per thread input array to store. At least ELEMENTS_PER_THREAD elements.
    // per_block_output:    Contiguous output array to write into. This is per block, and should point at
    //                      the first element of the block's work space. It should be aligned to VECTOR_SIZE.
    // thread_index:        Thread index in the 1D block. Usually threadIdx.x.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, i32 VECTOR_SIZE, typename Value>
    NOA_ID void block_store(
            const Value* __restrict__ per_thread_input,
            Value* __restrict__ per_block_output,
            i32 thread_index
    ) {
        using aligned_vector_t = AlignedVector<Value, VECTOR_SIZE>;
        auto* destination = reinterpret_cast<aligned_vector_t*>(per_block_output);
        NOA_ASSERT(!(reinterpret_cast<std::uintptr_t>(per_block_output) % alignof(aligned_vector_t)));

        static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
        static_assert(!(ELEMENTS_PER_THREAD % VECTOR_SIZE));
        constexpr i32 VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;
        #pragma unroll
        for (i32 i = 0; i < VECTORIZED_LOADS; i++) {
            aligned_vector_t vector_to_store;
            #pragma unroll
            for (i32 j = 0; j < VECTOR_SIZE; j++)
                vector_to_store.data[j] = per_thread_input[VECTOR_SIZE * i + j];
            destination[i * BLOCK_SIZE + thread_index] = vector_to_store; // vectorized store
        }
    }

    // Reduces BLOCK_SIZE elements from shared_data.
    // The first thread (thread_index == 0) returns with the reduced value
    // The returned value is undefined for the other threads.
    // shared_data:     Shared memory to reduce. Should be at least BLOCK_SIZE elements. It is overwritten.
    // thread_index:    Thread index. From 0 to BLOCK_SIZE - 1.
    // reduce_op:       Reduction operator.
    template<i32 BLOCK_SIZE, typename Reduced, typename ReduceOp>
    NOA_ID Reduced block_reduce_shared(
            Reduced* shared_data,
            i32 thread_index,
            ReduceOp reduce_op
    ) {
        constexpr i32 WARP_SIZE = cuda::Constant::WARP_SIZE;
        static_assert(!(BLOCK_SIZE % WARP_SIZE) && !((BLOCK_SIZE / WARP_SIZE) % 2));

        // Reduce shared data.
        if constexpr (BLOCK_SIZE > WARP_SIZE) {
            Reduced* shared_data_tid = shared_data + thread_index;
            #pragma unroll
            for (i32 i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
                if (thread_index < i)
                    *shared_data_tid = reduce_op(*shared_data_tid, shared_data_tid[i]);
                block_synchronize();
            }
        }

        // Final reduction within a warp.
        Reduced value;
        if (thread_index < WARP_SIZE)
            value = warp_reduce(shared_data[thread_index], reduce_op);
        return value;
    }

    // Reduces min(BLOCK_SIZE * ELEMENTS_PER_THREAD, elements) elements from input.
    // BLOCK_SIZE:          Number of threads in the dimension to reduce.
    // ELEMENTS_PER_THREAD: Number of elements to load, for each thread. Should be >= VECTOR_SIZE
    // VECTOR_SIZE:         Vector size. If 1, there's no vectorization.
    // per_block_input:     Input array to reduce. It starts at the first element to reduce.
    // stride:              Stride between each element. This is ignored if VECTOR_SIZE > 1.
    // elements:            Maximum number of elements that can be reduced from per_block_input.
    // preprocess_op:       Preprocess operator: op(Input) -> Reduced, or op(Input, offset) -> Reduced.
    // reduce_op:           Reduction operator: op(Reduced, Reduced) -> Reduced.
    // reduced:             Per-thread initial value used for the reduction.
    //                      It is updated with the final reduced value.
    // thread_index:        Thread index.
    // global_offset:       Per block offset corresponding to the beginning of per_block_input.
    //                      This is used to compute the memory offset of each reduced elements
    //                      when preprocess_op is a binary operator, otherwise it is ignored.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, i32 VECTOR_SIZE,
            typename Input, typename Reduced, typename Index, typename PreprocessOp, typename ReduceOp>
    NOA_ID void block_reduce_global_unary(
            const Input* __restrict__ per_block_input, Index stride, Index elements,
            PreprocessOp preprocess_op, ReduceOp reduce_op,
            Reduced* __restrict__ reduced,
            Index thread_index, Index global_offset = 0
    ) {
        constexpr bool EXPECT_OFFSET = noa::traits::is_detected_v<
                noa::traits::has_binary_operator, PreprocessOp, Input, Index>;

        if constexpr (VECTOR_SIZE > 1) {
            (void) stride; // assume contiguous
            if (elements < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const Index tid = BLOCK_SIZE * i + thread_index;
                    if (tid < elements) {
                        if constexpr (EXPECT_OFFSET) {
                            const auto offset = global_offset + tid; // here tid is the thread offset
                            *reduced = reduce_op(*reduced, preprocess_op(per_block_input[tid], offset));
                        } else {
                            *reduced = reduce_op(*reduced, preprocess_op(per_block_input[tid]));
                        }
                    }
                }
            } else {
                Input values_to_reduce[ELEMENTS_PER_THREAD];
                block_load<BLOCK_SIZE, ELEMENTS_PER_THREAD, VECTOR_SIZE>(
                        per_block_input, values_to_reduce, thread_index);
                #pragma unroll
                for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    if constexpr (EXPECT_OFFSET) {
                        // Here computing the global offset is more complicated because
                        // of the vectorization. The values in "values_to_reduce" are
                        // saved by chunks of VECTOR_SIZE. There are multiple chunks if
                        // ELEMENTS_PER_THREAD > VECTOR_SIZE. In this case, chunks are
                        // "separated" in the per_block_input by the BLOCK_SIZE.
                        const auto block_offset = BLOCK_SIZE * (i / VECTOR_SIZE);
                        const auto thread_offset = (thread_index + block_offset) * VECTOR_SIZE + i % VECTOR_SIZE;
                        const auto offset = global_offset + thread_offset;
                        *reduced = reduce_op(*reduced, preprocess_op(values_to_reduce[i], offset));
                    } else {
                        *reduced = reduce_op(*reduced, preprocess_op(values_to_reduce[i]));
                    }
                }
            }
        } else { // no vectorized loads
            #pragma unroll
            for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const Index tid = BLOCK_SIZE * i + thread_index;
                if (tid < elements) {
                    if constexpr (EXPECT_OFFSET) {
                        const auto offset = global_offset + tid * stride;
                        *reduced = reduce_op(*reduced, preprocess_op(per_block_input[tid * stride], offset));
                    } else {
                        *reduced = reduce_op(*reduced, preprocess_op(per_block_input[tid * stride]));
                    }
                }
            }
        }
    }

    // Reduces min(BLOCK_SIZE * ELEMENTS_PER_THREAD, elements) elements from inputs.
    // BLOCK_SIZE:          Number of threads in the dimension to reduce.
    // ELEMENTS_PER_THREAD: Number of elements to load, for each thread. Should be >= VECTOR_SIZE
    // VECTOR_SIZE:         Vector size. If 1, there's no vectorization.
    // lhs, rhs:            Left and right-hand side input arrays to reduce.
    //                      Should start at the first element to reduce.
    // lhs_stride:          Stride between each element in lhs. This is ignored if VECTOR_SIZE > 1.
    // rhs_stride:          Stride between each element in rhs. This is ignored if VECTOR_SIZE > 1.
    // elements:            Maximum number of elements that can be reduced starting from input.
    // preprocess_op:       Preprocess operator: op(Input) -> Reduced, or op(Input, offset) -> Reduced.
    // reduce_op:           Reduction operator: op(Reduced, Reduced) -> Reduced.
    // reduced:             Per-thread left-hand side argument of reduce_op.
    //                      It is updated with the final reduced value.
    // thread_index:        Thread index.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, i32 VECTOR_SIZE,
            typename Lhs, typename Rhs, typename Reduced, typename Index,
            typename PreprocessOp, typename ReduceOp>
    NOA_ID void block_reduce_global_binary(
            const Lhs* per_block_lhs, Index lhs_stride,
            const Rhs* per_block_rhs, Index rhs_stride, Index elements,
            PreprocessOp preprocess_op, ReduceOp reduce_op,
            Reduced* __restrict__ per_thread_reduced,
            Index thread_index
    ) {
        if constexpr (VECTOR_SIZE > 1) {
            (void) lhs_stride; // assume contiguous
            (void) rhs_stride; // assume contiguous
            if (elements < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const Index tid = BLOCK_SIZE * i + thread_index;
                    if (tid < elements) {
                        *per_thread_reduced = reduce_op(
                                *per_thread_reduced,
                                preprocess_op(per_block_lhs[tid], per_block_rhs[tid]));
                    }
                }
            } else {
                Lhs lhs_values_to_reduce[ELEMENTS_PER_THREAD];
                Rhs rhs_values_to_reduce[ELEMENTS_PER_THREAD];
                block_load<BLOCK_SIZE, ELEMENTS_PER_THREAD, VECTOR_SIZE>(
                        per_block_lhs, lhs_values_to_reduce, thread_index);
                block_load<BLOCK_SIZE, ELEMENTS_PER_THREAD, VECTOR_SIZE>(
                        per_block_rhs, rhs_values_to_reduce, thread_index);
                #pragma unroll
                for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    *per_thread_reduced = reduce_op(
                            *per_thread_reduced,
                            preprocess_op(lhs_values_to_reduce[i], rhs_values_to_reduce[i]));
                }
            }
        } else {
            #pragma unroll
            for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const Index tid = BLOCK_SIZE * i + thread_index;
                if (tid < elements) {
                    *per_thread_reduced = reduce_op(
                            *per_thread_reduced,
                            preprocess_op(per_block_lhs[tid * lhs_stride], per_block_rhs[tid * rhs_stride]));
                }
            }
        }
    }
}
