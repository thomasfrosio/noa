#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/gpu/cuda/Constants.hpp"
#include "noa/gpu/cuda/Pointers.hpp"
#include "noa/gpu/cuda/Warp.cuh"

// TODO CUDA's cub seems to have some load and store functions. Surely some of them can be used here.

// Static initialization of shared variables:
// We add some logic to support types that default initializes, but this seems to have been fixed in 11.7
// (nvcc zero initializes by default). I couldn't find it in the changelog though, but it does compile...

namespace noa::cuda::guts {
    template<typename Tup>
    struct vectorized_tuple { using type = Tup; };

    template<typename... T>
    struct vectorized_tuple<Tuple<T...>> { using type = Tuple<AccessorValue<typename T::value_type>...>; };

    /// Convert a tuple of accessors to a tuple of AccessorValue.
    /// This is used to store the values for vectorized load/stores, while preserving compatibility with the core
    /// interfaces. Note that the constness is preserved, so to access the values from the AccessorValue,
    /// .ref_() should be used.
    template<typename T>
    using vectorized_tuple_t = vectorized_tuple<std::decay_t<T>>::type;

    template<size_t N, typename Index, typename T>
    struct joined_tuple { using type = T; };

    template<size_t N, typename Index, typename... T>
    struct joined_tuple<N, Index, Tuple<T...>> {
        using type = Tuple<AccessorRestrictContiguous<typename T::mutable_value_type, N, Index>...>;
    };

    /// Convert a tuple of AccessorValue(s) to a tuple of N-d AccessorsContiguousRestrict.
    /// This is used to store the reduced values of reduce_ewise or reduce_axes_ewise in a struct of arrays,
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
    /// \tparam ELEMENTS_PER_THREAD The number of elements to load, per thread.
    /// \param per_block_input      Tuple of accessors (Accessor or AccessorValue) to load from.
    ///                             Accessor should be 1d contiguous, pointing at the start of the block work space.
    /// \param per_thread_output    Per thread output tuple of AccessorValue(s).
    /// \param thread_index         Thread index in the 1d block. Usually threadIdx.x.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, typename AlignedBuffers, typename Input, typename Output>
    requires ((nt::tuple_of_accessor<Input> and nt::tuple_of_accessor_value<Output>) or nt::empty_tuple<Input, Output>)
    NOA_ID void block_load(
        const Input& per_block_input,
        Output* __restrict__ per_thread_output,
        i32 thread_index
    ) {
        per_block_input.for_each_enumerate([&]<size_t I, typename T>(T& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                auto& input_value = accessor.ref();
                for (size_t i{}; i < ELEMENTS_PER_THREAD; ++i) {
                    // Use .ref_() to be able to write into AccessorValue<const V> types.
                    // We want to keep the const on the core interfaces, but here we actually
                    // need to initialize the underlying values.
                    per_thread_output[i][Tag<I>{}].ref_() = input_value;
                }
            } else if constexpr (nt::is_accessor_v<T>) {
                // The input type is reinterpreted to this aligned buffer type.
                using aligned_buffer_t = std::tuple_element_t<I, AlignedBuffers>;

                // The beginning of every block should be correctly aligned.
                // This should be guaranteed before launching the kernel, but better check here too.
                NOA_ASSERT(is_multiple_of(reinterpret_cast<uintptr_t>(accessor.get()), alignof(aligned_buffer_t)));

                // If we need more than one vectorized load, we offset the input by
                // the entire block size and offset the output by the vector size.
                constexpr size_t VECTOR_SIZE = aligned_buffer_t::SIZE;
                static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
                static_assert(is_multiple_of(ELEMENTS_PER_THREAD, VECTOR_SIZE));

                const auto* aligned_buffer = reinterpret_cast<const aligned_buffer_t*>(accessor.get());
                constexpr i32 N_VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;
                for (i32 i{}; i < N_VECTORIZED_LOADS; ++i) {
                    aligned_buffer_t loaded_buffer = aligned_buffer[i * BLOCK_SIZE + thread_index];
                    for (i32 j{}; j < VECTOR_SIZE; ++j)
                        per_thread_output[VECTOR_SIZE * i + j][Tag<I>{}].ref_() = loaded_buffer.data[j];
                }
            } else {
                static_assert(nt::always_false<T>);
            }
        });
    }

    /// Each thread stores ELEMENTS_PER_THREAD elements, using vectorized store instructions if possible.
    /// If ELEMENTS_PER_THREAD > VECTOR_SIZE, multiple writes per thread will be necessary.
    /// \tparam BLOCK_SIZE          The number of threads per block. It assumes a 1D contiguous block.
    /// \tparam ELEMENTS_PER_THREAD The number of elements to store, per thread.
    /// \param per_thread_input     Per thread input array to store. At least ELEMENTS_PER_THREAD elements.
    /// \param per_block_output     Contiguous output array to write into. This is per block, and should point at
    ///                             the first element of the block's work space.
    /// \param thread_index         Thread index in the 1D block. Usually threadIdx.x.
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, typename AlignedBuffers, typename Input, typename Output>
    requires ((nt::tuple_of_accessor_value<Input> and
               (nt::tuple_of_accessor<Output> and not nt::tuple_of_accessor_value<Output>)) or
              nt::empty_tuple<Input, Output>)
    NOA_ID void block_store(
        Input* __restrict__ per_thread_input,
        const Output& per_block_output,
        i32 thread_index
    ) {
        per_block_output.for_each_enumerate([&]<size_t I, typename T>(T& accessor) {
            if constexpr (nt::is_accessor_v<T>) {
                using aligned_buffer_t = std::tuple_element_t<I, AlignedBuffers>;
                NOA_ASSERT(is_multiple_of(reinterpret_cast<uintptr_t>(accessor.get()), alignof(aligned_buffer_t)));

                constexpr size_t VECTOR_SIZE = aligned_buffer_t::SIZE;
                static_assert(ELEMENTS_PER_THREAD >= VECTOR_SIZE);
                static_assert(is_multiple_of(ELEMENTS_PER_THREAD, VECTOR_SIZE));
                constexpr i32 VECTORIZED_LOADS = ELEMENTS_PER_THREAD / VECTOR_SIZE;

                auto* aligned_buffer = reinterpret_cast<aligned_buffer_t*>(accessor.get());
                for (i32 i{}; i < VECTORIZED_LOADS; i++) {
                    aligned_buffer_t buffer_to_store;
                    for (i32 j{}; j < VECTOR_SIZE; j++)
                        buffer_to_store.data[j] = per_thread_input[VECTOR_SIZE * i + j][Tag<I>{}].ref();
                    aligned_buffer[i * BLOCK_SIZE + thread_index] = buffer_to_store;
                }
            } else {
                static_assert(nt::always_false<T>);
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
        constexpr i32 WARP_SIZE = Constant::WARP_SIZE;
        static_assert(is_multiple_of(BLOCK_SIZE, WARP_SIZE) and
                      (BLOCK_SIZE == WARP_SIZE or is_multiple_of((BLOCK_SIZE / WARP_SIZE), 2)));

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
        if (thread_index < WARP_SIZE) {
            if constexpr (wrap_reduceable<Reduced>) {
                value = warp_reduce<Interface>(op, shared_data[thread_index]);
            } else {
                // If the reduced type(s) cannot be warp reduced,
                // let the first thread of the block do it.
                if (thread_index == 0) {
                    value = shared_data[0];
                    for (i32 i = 1; i < WARP_SIZE; ++i)
                        Interface::join(op, shared_data[i], value);
                }
            }
        }
        return value;
    }

    template<typename Interface, u32 BLOCK_SIZE,
             typename Op, typename Reduced, typename Joined, typename Index, typename... Indices>
    NOA_ID void block_reduce_join(
        Op op,
        Reduced& reduced, // Tuple of AccessorValue(s); e.g. Tuple<AccessorValue<i32>, AccessorValue<f64>>
        Joined& joined,
        Index index_within_block,
        Indices... indices_within_grid // per block nd-indices where to save the per-block reduced value in joined
    ) {
        __shared__ Reduced shared[BLOCK_SIZE];
        shared[index_within_block] = reduced;
        block_synchronize();
        reduced = block_reduce_shared<Interface, BLOCK_SIZE>(op, shared, index_within_block);
        if (index_within_block == 0) {
            if constexpr (nt::is_tuple_of_accessor_v<Joined>) {
                // Tuple of Accessor(s), corresponding to Reduced, e.g. Tuple<Accessor<i32, 1>, Accessor<f64, 1>>
                joined.for_each_enumerate([&]<size_t I>(auto& accessor) {
                    accessor(indices_within_grid...) = reduced[Tag<I>{}].ref();
                });
            } else if constexpr (nt::is_accessor_v<Joined>) {
                joined(indices_within_grid...) = reduced;
            } else if constexpr (nt::is_pointer_v<Joined> and sizeof...(Indices) == 1) {
                auto index = forward_as_tuple(indices_within_grid...)[Tag<0>{}]; // TODO C++23 [,] should fix that
                joined[index] = reduced;
            } else {
                static_assert(nt::always_false<Joined>);
            }
        }
    }

    template<typename Interface, u32 BLOCK_SIZE,
             typename Op, typename Reduced, typename Output, typename Index, typename... Indices>
    NOA_ID void block_reduce_join_and_final(
        Op op,
        Reduced& reduced, // Tuple of AccessorValue(s)
        Output& output, // Tuple of 1d Accessor(s)
        Index index_within_block,
        Indices... indices_within_output
    ) {
        __shared__ Reduced shared[BLOCK_SIZE];
        shared[index_within_block] = reduced;
        block_synchronize();
        reduced = block_reduce_shared<Interface, BLOCK_SIZE>(op, shared, index_within_block);
        if (index_within_block == 0)
            Interface::final(op, reduced, output, indices_within_output...);
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
    template<i32 BLOCK_SIZE, i32 ELEMENTS_PER_THREAD, typename AlignedBuffers, typename Interface,
             nt::tuple_of_accessor_nd<1> Input,
             nt::tuple_of_accessor_value Reduced,
             typename Index, typename Op>
    NOA_ID void block_reduce_ewise_1d_init(
        Op op,
        const Input& per_block_input,
        Index n_elements_to_reduce,
        Reduced& reduced,
        Index thread_index
    ) {
        if constexpr (is_vectorized<AlignedBuffers>()) {
            if (n_elements_to_reduce >= ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                using ivec_t = vectorized_tuple_t<Input>;
                ivec_t vectorized_input[ELEMENTS_PER_THREAD];
                block_load<BLOCK_SIZE, ELEMENTS_PER_THREAD, AlignedBuffers>(
                    per_block_input, vectorized_input, thread_index);

                #pragma unroll
                for (Index i{}; i < ELEMENTS_PER_THREAD; ++i)
                    Interface::init(op, vectorized_input[i], reduced, 0);
                return;
            }
        }

        #pragma unroll
        for (Index i{}; i < ELEMENTS_PER_THREAD; ++i) {
            const Index tid = thread_index + BLOCK_SIZE * i;
            if (tid < n_elements_to_reduce)
                Interface::init(op, per_block_input, reduced, tid);
        }
    }

    template<i32 BLOCK_SIZE, i32 N_ELEMENTS_PER_THREAD, typename AlignedBuffers,
             typename Interface, typename Op, typename Index,
             nt::tuple_of_accessor_nd<1> Joined,
             nt::tuple_of_accessor_value Reduced>
    requires (nt::tuple_of_accessor_pure<Joined> or nt::tuple_of_accessor_reference<Joined>)
    NOA_ID void block_reduce_ewise_1d_join(
        Op op,
        const Joined& per_block_joined,
        Index n_elements_to_reduce,
        Reduced& reduced,
        Index thread_index
    ) {
        // Tuple<Accessor, ...>(soa) -> Tuple<AccessorValue, ...>(aos)
        using ivec_t = vectorized_tuple_t<Joined>;

        if constexpr (is_vectorized<AlignedBuffers>()) {
            if (n_elements_to_reduce >= N_ELEMENTS_PER_THREAD * BLOCK_SIZE) {
                ivec_t vectorized_joined[N_ELEMENTS_PER_THREAD];
                block_load<BLOCK_SIZE, N_ELEMENTS_PER_THREAD, AlignedBuffers>(
                    per_block_joined, vectorized_joined, thread_index);

                #pragma unroll
                for (Index i{}; i < N_ELEMENTS_PER_THREAD; ++i)
                    Interface::join(op, vectorized_joined[i], reduced);
                return;
            }
        }

        #pragma unroll
        for (Index i{}; i < N_ELEMENTS_PER_THREAD; ++i) {
            const Index tid = thread_index + BLOCK_SIZE * i;
            if (tid < n_elements_to_reduce) {
                // We need to reconstruct the reference type, which is a tuple of AccessorValue(s).
                ivec_t joined = per_block_joined.map([tid]<typename T>(T& accessor) {
                    using accessor_value_t = AccessorValue<nt::value_type_t<T>>;
                    return accessor_value_t(accessor(tid));
                });
                Interface::join(op, joined, reduced);
            }
        }
    }
}
