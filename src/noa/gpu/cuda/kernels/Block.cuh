#pragma once

#include "noa/core/Config.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Pointers.hpp"
#include "noa/gpu/cuda/kernels/Warp.cuh"

// TODO CUDA's cub seems to have some load and store functions. Surely some of them can be used here.

// Static initialization of shared variables:
// We add some logic to support types that default initializes, but this seems to have been fixed in 11.7
// (nvcc zero initializes by default). I couldn't find it in the changelog though, but it does compile...

namespace noa::cuda::guts {
    template<typename Tup>
    struct vectorized_tuple { using type = Tup; };

    template<typename... T>
    struct vectorized_tuple<Tuple<T...>> { using type = Tuple<AccessorValue<typename T::value_type>...>; };

    /// Convert a tuple of accessors to AccessorValue.
    /// This is used to store the values for vectorized load/stores, while preserving compatibility with the core
    /// interfaces. Note that the constness is preserved, so to access the values from the AccessorValue,
    /// .deref_() should be used.
    template<typename T>
    using vectorized_tuple_t = vectorized_tuple<std::decay_t<T>>::type;

    template<size_t N, typename Index, typename T>
    struct joined_tuple { using type = T; };

    template<size_t N, typename Index, typename... T>
    struct joined_tuple<N, Index, Tuple<T...>> {
        using type = Tuple<AccessorRestrictContiguous<typename T::mutable_value_type, N, Index>...>;
    };

    /// Convert a tuple of AccessorValue(s) to a tuple of N-d AccessorsContiguousRestrict.
    /// This is used to store the reduced values of reduce_ewise or reduce_axes_ewise in an struct of arrays,
    /// which can then be loaded as an input using vectorized loads. Note that the constness of the AccessorValue(s)
    /// has to be dropped (because kernels do expect to write then read from these accessors), but this is fine since
    /// the reduced AccessorValue(s) should not be const anyway.
    template<size_t N, typename Index, typename T>
    using joined_tuple_t = joined_tuple<N, Index, std::decay_t<T>>::type;

    /// Synchronizes the block.
    /// TODO Cooperative groups may be the way to go and do offer more granularity.
    NOA_FD void block_synchronize() {
        __syncthreads();
    }

    /// Returns a per-thread unique ID, i.e. each thread in the grid gets a unique value.
    template<size_t N = 0>
    NOA_FD u32 thread_uid() {
        if constexpr (N == 1) {
            return blockIdx.x * blockDim.x + threadIdx.x;
        } else if constexpr (N == 2) {
            const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
            const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
            return bid_y * blockDim.x + bid_x;
        } else {
            const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
            const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
            const auto bid_z = blockIdx.z * blockDim.z + threadIdx.z;
            return (bid_z * blockDim.y + bid_y) * blockDim.x + bid_x;
        }
    }

    /// Each thread loads ELEMENTS_PER_THREAD elements, using vectorized load instructions if possible.
    /// If ELEMENTS_PER_THREAD > VECTOR_SIZE, multiple reads per thread will be necessary.
    /// \tparam BLOCK_SIZE          The number of threads per block. It assumes a 1d contiguous block.
    /// \tparam VECTOR_SIZE         Number of elements to load at the same time. If 1, there's no vectorization.
    /// \tparam ELEMENTS_PER_THREAD The number of elements to load, per thread.
    /// \param per_block_input      Tuple of accessors (Accessor or AccessorValue) to load from.
    ///                             Accessor should be 1d contiguous, pointing at the start of the block work space.
    ///                             Pointers should have (at least) the same alignment as sizeof(value_type)*VECTOR_SIZE.
    /// \param per_thread_output    Per thread output tuple of AccessorValue(s).
    /// \param thread_index         Thread index in the 1d block. Usually threadIdx.x.
    template<i32 BLOCK_SIZE, i32 VECTOR_SIZE, i32 ELEMENTS_PER_THREAD, typename Input, typename Output>
    requires ((nt::is_tuple_of_accessor_v<Input> and nt::is_tuple_of_accessor_value_v<Output>) or
               nt::are_empty_tuple_v<Input, Output>)
    NOA_ID void block_load(
            const Input& per_block_input,
            Output* __restrict__ per_thread_output,
            i32 thread_index
    ) {
        per_block_input.for_each_enumerate([&]<size_t I, typename T>(T& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                auto& input_value = accessor.deref();
                for (size_t i{0}; i < ELEMENTS_PER_THREAD; ++i) {
                    // Use .deref_() to be able to write into AccessorValue<const V> types.
                    // We want to keep the const on the core interfaces, but here we actually
                    // need to initialize the underlying values.
                    per_thread_output[i][Tag<I>{}].deref_() = input_value;
                }
            } else if constexpr (nt::is_accessor_v<T>) {
                // The input type is reinterpreted to this aligned vector. VECTOR_SIZE should be
                // correctly set so that the alignment of the input pointer is enough for this vector type.
                using value_t = T::mutable_value_type;
                using aligned_vector_t = AlignedVector<value_t, VECTOR_SIZE>; // TODO use Vec<value_t, VECTOR_SIZE, sizeof(value_t) * VECTOR_SIZE>?
                const auto* vectorized_input = reinterpret_cast<const aligned_vector_t*>(accessor.get());
                NOA_ASSERT(!(reinterpret_cast<std::uintptr_t>(per_block_input) % alignof(aligned_vector_t)));

                // If we need more than one vectorized load, we offset the input by
                // the entire block size and offset the output by the vector size.
                static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
                static_assert(is_multiple_of(ELEMENTS_PER_THREAD, VECTOR_SIZE));
                constexpr i32 VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;
                for (i32 i = 0; i < VECTORIZED_LOADS; ++i) {
                    aligned_vector_t loaded_vector = vectorized_input[i * BLOCK_SIZE + thread_index];
                    for (i32 j = 0; j < VECTOR_SIZE; ++j)
                        per_thread_output[VECTOR_SIZE * i + j][Tag<I>{}].deref_() = loaded_vector.data[j];
                }
            } else {
                static_assert(nt::always_false_v<T>);
            }
        });
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
    template<i32 BLOCK_SIZE, i32 VECTOR_SIZE, i32 ELEMENTS_PER_THREAD, typename Input, typename Output>
    requires ((nt::is_tuple_of_accessor_value_v<Input> and
               (nt::is_tuple_of_accessor_v<Output> and not nt::is_tuple_of_accessor_value_v<Output>)) or
              nt::are_empty_tuple_v<Input, Output>)
    NOA_ID void block_store(
            Input* __restrict__ per_thread_input,
            const Output& per_block_output,
            i32 thread_index
    ) {
        per_block_output.for_each_enumerate([&]<size_t I, typename T>(T& accessor) {
            if constexpr (nt::is_accessor_v<T>) {
                using value_t = T::mutable_value_type; // or value_type since output is mutable
                using aligned_vector_t = AlignedVector<value_t, VECTOR_SIZE>;
                auto* vectorized_output = reinterpret_cast<aligned_vector_t*>(accessor.get());
                NOA_ASSERT(!(reinterpret_cast<std::uintptr_t>(per_block_output) % alignof(aligned_vector_t)));

                static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
                static_assert(is_multiple_of(ELEMENTS_PER_THREAD, VECTOR_SIZE));
                constexpr i32 VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;
                for (i32 i = 0; i < VECTORIZED_LOADS; i++) {
                    aligned_vector_t vector_to_store;
                    for (i32 j = 0; j < VECTOR_SIZE; j++)
                        vector_to_store.data[j] = per_thread_input[VECTOR_SIZE * i + j][Tag<I>{}].deref();
                    vectorized_output[i * BLOCK_SIZE + thread_index] = vector_to_store;
                }
            } else {
                static_assert(nt::always_false_v<T>);
            }
        });
    }

    // Reduces BLOCK_SIZE elements from shared_data.
    // The first thread (thread_index == 0) returns with the reduced value
    // The returned value is undefined for the other threads.
    // shared_data:     Shared memory to reduce. Should be at least BLOCK_SIZE elements. It is overwritten.
    // thread_index:    Thread index. From 0 to BLOCK_SIZE - 1.
    // reduce_op:       Reduction operator.
    template<typename Interface, i32 BLOCK_SIZE, typename Reduced, typename Op>
    NOA_ID Reduced block_reduce_shared(
            Op op,
            Reduced* __restrict__ shared_data,
            i32 thread_index
    ) {
        constexpr i32 WARP_SIZE = noa::cuda::Constant::WARP_SIZE;
        static_assert(is_multiple_of(BLOCK_SIZE, WARP_SIZE) and
                      is_multiple_of((BLOCK_SIZE / WARP_SIZE), 2));

        // Reduce shared data.
        if constexpr (BLOCK_SIZE > WARP_SIZE) {
            Reduced* shared_data_tid = shared_data + thread_index;
            #pragma unroll
            for (i32 i = BLOCK_SIZE / 2; i >= WARP_SIZE; i /= 2) {
                if (thread_index < i)
                    Interface::join(op, shared_data_tid[i], *shared_data_tid);
                block_synchronize();
            }
        }

        // Final reduction within a warp.
        Reduced value;
        if (thread_index < WARP_SIZE)
            value = warp_reduce<Interface>(op, shared_data[thread_index]);
        return value;
    }

    template<typename Interface, u32 BLOCK_SIZE,
             typename Op, typename Reduced, typename Index>
    NOA_ID void block_reduce_join(
            Op op,
            Reduced& reduced, // Tuple of AccessorValue(s); reduced value(s) of the current thread
            Reduced* __restrict__ joined, // per-block reduced value(s)
            Index index_within_block,
            Index index_within_grid
    ) {
        __shared__ Reduced shared[BLOCK_SIZE];
        shared[index_within_block] = reduced;
        block_synchronize();
        reduced = block_reduce_shared<Interface, BLOCK_SIZE>(op, shared, index_within_block);
        if (index_within_block == 0)
            joined[index_within_grid] = reduced;
    }

    template<typename Interface, u32 BLOCK_SIZE,
             typename Op, typename Reduced, typename Joined, typename Index, typename... Indices>
    NOA_ID void block_reduce_join(
            Op op,
            Reduced& reduced, // Tuple of AccessorValue(s); e.g. Tuple<AccessorValue<i32>, AccessorValue<f64>>
            Joined& joined, // Tuple of Accessor(s), corresponding to Reduced, e.g. Tuple<Accessor<i32, 1>, Accessor<f64, 1>>
            Index index_within_block,
            Indices... indices_within_grid // per block nd-indices where to save the per-block reduced value
    ) {
        __shared__ Reduced shared[BLOCK_SIZE];
        shared[index_within_block] = reduced;
        block_synchronize();
        reduced = block_reduce_shared<Interface, BLOCK_SIZE>(op, shared, index_within_block);
        if (index_within_block == 0) {
            joined.for_each_enumerate([&]<size_t I>(auto& accessor){
                accessor(indices_within_grid...) = reduced[Tag<I>{}].deref();
            });
        }
    }

    template<typename Interface, u32 BLOCK_SIZE,
             typename Op, typename Reduced, typename Output, typename Index>
    NOA_ID void block_reduce_join_and_final(
            Op op,
            Reduced& reduced, // Tuple of AccessorValue(s)
            Output& output, // Tuple of 1d Accessor(s)
            Index index_within_block,
            Index index_within_output = 0
    ) {
        __shared__ Reduced shared[BLOCK_SIZE];
        shared[index_within_block] = reduced;
        block_synchronize();
        reduced = block_reduce_shared<Interface, BLOCK_SIZE>(op, shared, index_within_block);
        if (index_within_block == 0)
            Interface::final(op, reduced, output, index_within_output);
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
             typename Interface, typename Input, typename Reduced, typename Index, typename Op>
    requires (nt::is_tuple_of_accessor_v<Input> and
              nt::is_tuple_of_accessor_value_v<Reduced>)
    NOA_ID void block_reduce_ewise_1d_init(
            Op op,
            const Input& per_block_input,
            Index n_elements_to_reduce,
            Reduced& reduced,
            Index thread_index
    ) {
        if (VECTOR_SIZE == 1 or n_elements_to_reduce < ELEMENTS_PER_THREAD * BLOCK_SIZE) {
            #pragma unroll
            for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const Index tid = thread_index + BLOCK_SIZE * i;
                if (tid < n_elements_to_reduce)
                    Interface::init(op, per_block_input, reduced, tid);
            }
        } else {
            using ivec_t = vectorized_tuple_t<Input>;
            ivec_t vectorized_input[ELEMENTS_PER_THREAD];
            block_load<BLOCK_SIZE, VECTOR_SIZE, ELEMENTS_PER_THREAD>(per_block_input, vectorized_input, thread_index);

            #pragma unroll
            for (Index i = 0; i < ELEMENTS_PER_THREAD; ++i)
                Interface::init(op, vectorized_input[i], reduced, 0);
        }
    }

    template<i32 BLOCK_SIZE, i32 N_ELEMENTS_PER_THREAD, i32 LOAD_VECTOR_SIZE,
             typename Interface, typename Op, typename Index, typename Joined, typename Reduced>
    requires (nt::is_tuple_of_accessor_pure_v<Joined> and
              nt::is_tuple_of_accessor_value_v<Reduced>)
    NOA_ID void block_reduce_ewise_1d_join(
            Op op,
            const Joined& per_block_joined,
            Index n_elements_to_reduce,
            Reduced& reduced,
            Index thread_index
    ) {
        // Tuple<Accessor, ...>(soa) -> Tuple<AccessorValue, ...>(aos)
        using ivec_t = vectorized_tuple_t<Joined>;

        if (LOAD_VECTOR_SIZE == 1 or n_elements_to_reduce < N_ELEMENTS_PER_THREAD * BLOCK_SIZE) {
            #pragma unroll
            for (Index i = 0; i < N_ELEMENTS_PER_THREAD; ++i) {
                const Index tid = thread_index + BLOCK_SIZE * i;
                if (tid < n_elements_to_reduce) {
                    ivec_t joined = per_block_joined.map([](auto& accessor) {
                        using accessor_value_t = AccessorValue<nt::value_type_t<decltype(accessor)>>;
                        return accessor_value_t(accessor(tid));
                    });
                    Interface::join(op, joined, reduced);
                }
            }
        } else {
            ivec_t vectorized_joined[N_ELEMENTS_PER_THREAD];
            block_load<BLOCK_SIZE, LOAD_VECTOR_SIZE, N_ELEMENTS_PER_THREAD>(
                    per_block_joined, vectorized_joined, thread_index);

            #pragma unroll
            for (Index i = 0; i < N_ELEMENTS_PER_THREAD; ++i)
                Interface::join(op, vectorized_joined[i], reduced);
        }
    }
}
