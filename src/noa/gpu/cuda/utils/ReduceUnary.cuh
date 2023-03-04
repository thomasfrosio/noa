#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/memory/PtrPinned.hpp"

#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

// These reduction kernels are adapted from different sources, but the main logic comes from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::utils {
    template<u32 ElementsPerThread, u32 BlockSize, u32 MaxBlockCount>
    struct ReduceUnaryConfig {
        static constexpr u32 ELEMENTS_PER_THREAD = ElementsPerThread;
        static constexpr u32 BLOCK_SIZE = BlockSize;
        static constexpr u32 MAX_GRID_SIZE = MaxBlockCount;
    };
    using ReduceUnaryConfigDefault = ReduceUnaryConfig<8, 512, 4096>;
}

namespace noa::cuda::utils::details {
    // gridDim.x is the number of blocks, at most Config::MAX_GRID_SIZE, working on a given batch.
    // gridDim.y is the number of batches. If reduce_batch=true, there's only one batch.
    // Each batch is reduced to the number of blocks working the given batch, i.e. gridDim.x elements.
    // As such, the output of this kernel should be able to fit gridDim.x * gridDim.y reduced elements.
    template<typename Input, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp,
             StridesTraits StridesTrait, typename Config, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_unary_1d_large(
            AccessorRestrict<const Input, 2, Index, StridesTrait> input_batched,
            Index elements_per_batch, Reduced initial_reduced,
            AccessorRestrictContiguous<Reduced, 2, Index> tmp_reduced,
            PreProcessOp pre_process_op, ReduceOp reduce_op) {

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        // Batches are kept independent of each other.
        const Index batch = blockIdx.y; // if reduce_batch=true, batch == 0
        const auto input = input_batched[batch];

        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        const Index thread_index = threadIdx.x;
        const Index block_index = blockIdx.x;
        const Index block_count = gridDim.x;
        const Index block_offset = block_index * BLOCK_WORK_SIZE;
        const Index grid_work_size = block_count * BLOCK_WORK_SIZE;

        // Initial reduction to bring the input to BLOCK_SIZE * block_count elements.
        Reduced reduced = initial_reduced;
        for (Index cid = block_offset; cid < elements_per_batch; cid += grid_work_size) {
            const Index remaining = elements_per_batch - cid;
            const auto stride = input.template stride<0>();
            const auto global_offset = noa::indexing::at(cid, stride);
            block_reduce_global_unary<BLOCK_SIZE, EPT, VECTOR_SIZE>(
                    input.get() + global_offset, stride, remaining,
                    pre_process_op, reduce_op, &reduced,
                    thread_index, global_offset);
        }

        // Share thread's result to the other threads.
        __shared__ uninitialized_type_t<Reduced> s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<Reduced*>(s_data_);
        s_data[thread_index] = reduced;
        block_synchronize();

        // Reduce shared data to one element, i.e. block_count elements in total.
        const Reduced final = block_reduce_shared<BLOCK_SIZE>(s_data, thread_index, reduce_op);
        if (thread_index == 0)
            tmp_reduced(batch, block_index) = final;
    }

    // Here the input is organized has a series of rows. Given the original shape of the input,
    // the linear row index/offset, can be decomposed into its BDH index. Each dimension can
    // have an arbitrary stride, but if the rows themselves are contiguous (if the W stride is 1),
    // then vectorized load/stores can be used to load/store elements from the rows.
    // gridDim.x is the number of blocks to reduce the rows of a given batch.
    // gridDim.y is the number of batches.
    template<typename Input, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp,
             StridesTraits StridesTrait, typename Config,
             u32 BLOCK_DIM_X, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_unary_4d_large(
            AccessorRestrict<const Input, 4, Index, StridesTrait> input_batched,
            Shape4<Index> shape, Index rows_per_batch, Reduced initial_reduce,
            AccessorRestrictContiguous<Reduced, 2, Index> tmp_reduced,
            PreProcessOp pre_process_op, ReduceOp reduce_op) {

        // BLOCK_DIM_X is blockDim.x, but is passed at compile time to read rows more easily.
        NOA_ASSERT(BLOCK_DIM_X == blockDim.x);

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE_X = BLOCK_DIM_X * EPT;

        // Offset to the current batch. If reduce_batch=true, then batch==0 and this does nothing.
        const Index batch = blockIdx.y;
        const auto input_ptr = input_batched.offset_accessor(batch).get();

        // This uses 2D blocks:
        const Index block_index = blockIdx.x;
        const Index block_count = gridDim.x;
        const Index rows_per_block = blockDim.y;
        const Index rows_per_grid = block_count * rows_per_block;
        const Index initial_row = rows_per_block * block_index + threadIdx.y;

        // Initial reduction. Loop until all rows are consumed.
        Reduced reduced = initial_reduce;
        for (Index row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            const auto bdh_indexes = noa::indexing::offset2index(row, shape[1], shape[2]); // row -> B,D,H
            const auto offset_row = noa::indexing::at(bdh_indexes, input_batched.strides());
            const auto input_row = input_ptr + offset_row;

            // Consume the row:
            for (Index cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const Index remaining = shape[3] - cid;
                const auto stride = input_batched.template stride<3>();
                const auto offset = noa::indexing::at(cid, stride);
                const auto global_offset = offset_row + offset;
                block_reduce_global_unary<BLOCK_DIM_X, EPT, VECTOR_SIZE>(
                        input_row + offset, stride, remaining,
                        pre_process_op, reduce_op, &reduced,
                        static_cast<Index>(threadIdx.x), global_offset);
            }
        }

        // Share thread's result to the other threads.
        const Index thread_index = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        __shared__ uninitialized_type_t<Reduced> s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<Reduced*>(s_data_);
        s_data[thread_index] = reduced;
        block_synchronize();

        // Reduce shared data to one element, i.e. block_count elements in total
        const Reduced final = block_reduce_shared<BLOCK_SIZE>(s_data, thread_index, reduce_op);
        if (thread_index == 0)
            tmp_reduced(batch, block_index) = final;
    }

    // 1D grid, with 2D blocks. Each block reduces its batch to one element.
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp,
             StridesTraits StridesTrait, typename Config,
             u32 BLOCK_DIM_X, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_unary_4d_small(
            AccessorRestrict<const Input, 4, Index, StridesTrait> input_batched,
            Shape4<Index> shape, Index rows_per_batch, Reduced initial_reduce,
            AccessorRestrict<Output, 1, Index> output,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op) {

        // BLOCK_DIM_X is blockDim.x, but is passed at compile time to read rows more easily.
        NOA_ASSERT(BLOCK_DIM_X == blockDim.x);

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE_X = BLOCK_DIM_X * EPT;

        // Offset to the current batch. If reduce_batch=true, then batch==0 and this does nothing.
        const Index batch = blockIdx.x;
        const auto input_ptr = input_batched.offset_accessor(batch).get();

        // This uses 2D blocks:
        const Index rows_per_block = blockDim.y;
        const Index initial_row = threadIdx.y;

        // Initial reduction. Loop until all rows are consumed.
        Reduced reduced = initial_reduce;
        for (Index row = initial_row; row < rows_per_batch; row += rows_per_block) {
            const auto bdh_indexes = noa::indexing::offset2index(row, shape[1], shape[2]); // row -> B,D,H
            const auto offset_row = noa::indexing::at(bdh_indexes, input_batched.strides());
            const auto input_row = input_ptr + offset_row;

            // Consume the row:
            for (Index cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const Index remaining = shape[3] - cid;
                const auto stride = input_batched.template stride<3>();
                const auto offset = noa::indexing::at(cid, stride);
                const auto global_offset = offset_row + offset;
                block_reduce_global_unary<BLOCK_DIM_X, EPT, VECTOR_SIZE>(
                        input_row + offset, stride, remaining,
                        pre_process_op, reduce_op, &reduced,
                        static_cast<Index>(threadIdx.x), global_offset);
            }
        }

        // Share thread's result to the other threads.
        const Index thread_index = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        __shared__ uninitialized_type_t<Reduced> s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<Reduced*>(s_data_);
        s_data[thread_index] = reduced;
        block_synchronize();

        // Reduce shared data to one element.
        const Reduced final = block_reduce_shared<BLOCK_SIZE>(s_data, thread_index, reduce_op);
        if (thread_index == 0)
            output[batch] = post_process_op(final);
    }

    // Each block is assigned a batch. A block reduces its batch to a single element.
    // This is also used to reduce the "tmp_reduced" output.
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp,
             StridesTraits StridesTrait, typename Config, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_unary_1d_small(
            AccessorRestrict<const Input, 2, Index, StridesTrait> input_batched,
            Index elements_per_batch, Reduced initial_reduce,
            AccessorRestrict<Output, 1, Index> output,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op) {

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const Index thread_index = threadIdx.x;
        const Index batch = blockIdx.x; // if reduce_batch=true, batch==0
        const auto input = input_batched[batch];

        // elements -> one element per thread.
        Reduced reduced = initial_reduce;
        for (Index cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const Index remaining = elements_per_batch - cid;
            const auto stride = input.template stride<0>();
            const auto global_offset = noa::indexing::at(cid, stride);
            block_reduce_global_unary<BLOCK_SIZE, EPT, VECTOR_SIZE>(
                    input.get() + global_offset, stride, remaining,
                    pre_process_op, reduce_op, &reduced,
                    thread_index, global_offset);
        }

        // one element per thread -> one element per block.
        __shared__ uninitialized_type_t<Reduced> s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<Reduced*>(s_data_);
        s_data[thread_index] = reduced;
        block_synchronize();
        const Reduced final = block_reduce_shared<BLOCK_SIZE>(s_data, thread_index, reduce_op);

        if (thread_index == 0)
            output[batch] = post_process_op(final);
    }

    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = ReduceUnaryConfigDefault,
             typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_unary_small_1d(
            const char* name,
            const Input* input, Strides4<Index> input_strides, u32 batches, Index elements,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {

        const auto input_strides_2d = input_strides.filter(0, 3);

        // Try to vectorize the loads for the input.
        auto vector_size = input_strides_2d[1] == 1 ? std::min(max_vector_count(input), i64{8}) : 1;
        if (batches > 1) {
            for (; vector_size >= 2; vector_size /= 2) {
                if (!(input_strides_2d[0] % vector_size))
                    break;
            }
        }

        const auto config = LaunchConfig{batches, Config::BLOCK_SIZE};
        if (vector_size > 1) {
            const auto input_accessor = AccessorRestrictContiguous<const Input, 2, Index>(input, input_strides_2d);
            if (vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_unary_1d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 8>,
                               config, input_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else if (vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_unary_1d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 4>,
                               config, input_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else {
                stream.enqueue(name,
                               details::reduce_unary_1d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 2>,
                               config, input_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            }
        } else {
            const auto input_accessor = AccessorRestrict<const Input, 2, Index, StridesTrait>(input, input_strides_2d);
            stream.enqueue(name,
                           details::reduce_unary_1d_small<
                                   Input, Reduced, Output, Index,
                                   PreProcessOp, ReduceOp, PostProcessOp,
                                   StridesTrait, Config, 1>,
                           config, input_accessor, elements, initial_reduce, output_accessor,
                           pre_process_op, reduce_op, post_process_op);
        }
    }

    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = ReduceUnaryConfigDefault,
             typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_unary_large_1d(
            const char* name,
            const Input* input, Strides4<Index> input_strides, u32 batches, Index elements,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {
        // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
        // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
        // Here we use 1D blocks to go through each batch (if reduce_batch=true, there's only one batch).
        // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.

        // Try to vectorize the loads for the input.
        const auto input_strides_2d = input_strides.filter(0, 3);
        u32 input_vector_size = input_strides_2d[1] == 1 ? std::min(max_vector_count(input), i64{8}) : 1;
        if (batches > 1) {
            for (; input_vector_size >= 2; input_vector_size /= 2) {
                if (!(input_strides_2d[0] % input_vector_size))
                    break;
            }
        }

        const Index block_work_size = Config::BLOCK_SIZE * std::max(input_vector_size, Config::ELEMENTS_PER_THREAD);
        const u32 blocks_x = std::min(
                static_cast<u32>(noa::math::divide_up(elements, block_work_size)),
                Config::MAX_GRID_SIZE);
        const dim3 blocks(blocks_x, batches);
        const auto first_config = LaunchConfig{blocks, Config::BLOCK_SIZE};
        const auto second_config = LaunchConfig{batches, Config::BLOCK_SIZE};

        // Large reductions need two kernel launches. One that reduces the input to a bunch of reduced values,
        // and a second to reduce these values to the final output. Therefore, we need to allocate space
        // for these temporary reduced values. Here, pad so that the second kernel can use vectorized loads.
        constexpr u32 REDUCE_VECTOR_SIZE =
                noa::math::is_power_of_2(sizeof(Reduced)) ?
                noa::math::clamp(i64{16 / sizeof(Reduced)}, i64{1}, i64{8}) : 1;
        const u32 pitch = noa::math::next_multiple_of(blocks.x, REDUCE_VECTOR_SIZE);
        const auto reduced_buffer = noa::cuda::memory::PtrDevice<Reduced>::alloc(pitch * blocks.y, stream);
        const auto reduced_accessor = AccessorRestrictContiguous<Reduced, 2, Index>(
                reduced_buffer.get(), Strides2<Index>{pitch, 1});

        // (batch * elements) -> (blocks.x * blocks.y) elements.
        if (input_vector_size > 1) {
            const auto input_accessor = AccessorRestrictContiguous<const Input, 2, Index>(input, input_strides_2d);
            if (input_vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_unary_1d_large<
                                       Input, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       StridesTraits::CONTIGUOUS, Config, 8>,
                               first_config, input_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_unary_1d_large<
                                       Input, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       StridesTraits::CONTIGUOUS, Config, 4>,
                               first_config, input_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_unary_1d_large<
                                       Input, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       StridesTraits::CONTIGUOUS, Config, 2>,
                               first_config, input_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            }
        } else {
            const auto input_accessor = AccessorRestrict<const Input, 2, Index, StridesTrait>(input, input_strides_2d);
            stream.enqueue(name,
                           details::reduce_unary_1d_large<
                                   Input, Reduced, Index,
                                   PreProcessOp, ReduceOp,
                                   StridesTrait, Config, 1>,
                           first_config, input_accessor, elements, initial_reduce,
                           reduced_accessor, pre_process_op, reduce_op);
        }

        // (blocks.x * blocks.y) -> (blocks.y) elements.
        const auto const_reduced_accessor = AccessorRestrictContiguous<const Reduced, 2, Index>(reduced_accessor);
        stream.enqueue(name,
                       details::reduce_unary_1d_small<
                               Reduced, Reduced, Output, Index,
                               noa::copy_t, ReduceOp, PostProcessOp,
                               StridesTraits::CONTIGUOUS, Config, REDUCE_VECTOR_SIZE>,
                       second_config, const_reduced_accessor, blocks.x, initial_reduce, output_accessor,
                       noa::copy_t{}, reduce_op, post_process_op);
    }

    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
            typename Config = ReduceUnaryConfigDefault,
            typename Input, typename Reduced, typename Output, typename Index,
            typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_unary_large_4d(
            const char* name,
            const Input* input, Strides4<Index> input_strides,
            Shape4<Index> shape, u32 batches, bool reduce_batch,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {
        // In this config, the input cannot be easily interpreted as a 1D array.
        // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
        // Since the reduceUnaryLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3 outermost
        // dimensions can be strided. If the innermost dimension is contiguous, blocks can use vectorize loads
        // to read their row(s).

        // If rows are large, switch to more threads per row.
        const u32 block_dim_x = shape[3] > 512 ? 256 : 64; // FIXME
        const dim3 threads(block_dim_x, std::max(Config::BLOCK_SIZE / block_dim_x, u32{1}));
        const auto rows = safe_cast<u32>(shape[2] * shape[1] * (reduce_batch ? shape[0] : 1));
        const dim3 blocks(noa::math::min(noa::math::divide_up(rows, threads.y), Config::MAX_GRID_SIZE), batches);
        const auto first_config = LaunchConfig{blocks, threads};
        const auto second_config = LaunchConfig{batches, Config::BLOCK_SIZE};

        // Try to vectorize the loads within a row. For that, we have to
        // check that the beginning of each row is at the same alignment.
        u32 input_vector_size = input_strides[3] == 1 ? std::min(max_vector_count(input), i64{8}) : 1;
        for (; input_vector_size >= 2; input_vector_size /= 2) {
            if ((!(input_strides[2] % input_vector_size) || shape[2] == 1) &&
                (!(input_strides[1] % input_vector_size) || shape[1] == 1) &&
                (!(input_strides[0] % input_vector_size) || shape[0] == 1))
                break;
        }

        constexpr u32 REDUCE_VECTOR_SIZE =
                noa::math::is_power_of_2(sizeof(Reduced)) ?
                noa::math::clamp(i64{16 / sizeof(Reduced)}, i64{1}, i64{8}) : 1;
        const u32 pitch = noa::math::next_multiple_of(blocks.x, REDUCE_VECTOR_SIZE);
        const auto reduced_buffer = noa::cuda::memory::PtrDevice<Reduced>::alloc(pitch * blocks.y, stream);
        const auto reduced_accessor = AccessorRestrictContiguous<Reduced, 2, Index>(
                reduced_buffer.get(), Strides2<Index>{pitch, 1});

        const auto input_accessor = AccessorRestrict<const Input, 4, Index, StridesTrait>(input, input_strides);
        if (threads.x == 256) {
            if (input_vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 256, 8>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 256, 4>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 2) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 256, 2>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 256, 1>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            }
        } else {
            if (input_vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 64, 8>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 64, 4>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 2) {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 64, 2>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_unary_4d_large<
                                       Input, Reduced, Index, PreProcessOp, ReduceOp,
                                       StridesTrait, Config, 64, 1>,
                               first_config, input_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            }
        }

        const auto const_reduced_accessor = AccessorRestrictContiguous<const Reduced, 2, Index>(reduced_accessor);
        stream.enqueue(name,
                       details::reduce_unary_1d_small<
                               Reduced, Reduced, Output, Index,
                               noa::copy_t, ReduceOp, PostProcessOp,
                               StridesTraits::CONTIGUOUS, Config, REDUCE_VECTOR_SIZE>,
                       second_config, const_reduced_accessor, blocks.x, initial_reduce, output_accessor,
                       noa::copy_t{}, reduce_op, post_process_op);
    }

    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
            typename Config = ReduceUnaryConfigDefault,
            typename Input, typename Reduced, typename Output, typename Index,
            typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_unary_small_4d(
            const char* name,
            const Input* input, Strides4<Index> input_strides,
            Shape4<Index> shape, u32 batches, bool reduce_batch,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {

        // If rows are large, switch to more threads per row.
        const u32 block_dim_x = noa::cuda::Constant::WARP_SIZE; // FIXME
        const dim3 threads(block_dim_x, std::max(Config::BLOCK_SIZE / block_dim_x, u32{1}));
        const auto config = LaunchConfig{batches, threads};
        const auto rows = safe_cast<u32>(shape[2] * shape[1] * (reduce_batch ? shape[0] : 1));

        // Try to vectorize the loads within a row. For that, we have to
        // check that the beginning of each row is at the same alignment.
        u32 input_vector_size = input_strides[3] == 1 ? std::min(max_vector_count(input), i64{8}) : 1;
        for (; input_vector_size >= 2; input_vector_size /= 2) {
            if ((!(input_strides[2] % input_vector_size) || shape[2] == 1) &&
                (!(input_strides[1] % input_vector_size) || shape[1] == 1) &&
                (!(input_strides[0] % input_vector_size) || shape[0] == 1))
                break;
        }

        if (input_vector_size > 1) {
            const auto input_accessor = AccessorRestrictContiguous<const Input, 4, Index>(input, input_strides);
            if (input_vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_unary_4d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 64, 8>,
                               config, input_accessor, shape, rows, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else if (input_vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_unary_4d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 64, 4>,
                               config, input_accessor, shape, rows, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else {
                stream.enqueue(name,
                               details::reduce_unary_4d_small<
                                       Input, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       StridesTraits::CONTIGUOUS, Config, 64, 2>,
                               config, input_accessor, shape, rows, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            }
        } else {
            const auto input_accessor = AccessorRestrict<const Input, 4, Index, StridesTrait>(input, input_strides);
            stream.enqueue(name,
                           details::reduce_unary_4d_small<
                                   Input, Reduced, Output, Index,
                                   PreProcessOp, ReduceOp, PostProcessOp,
                                   StridesTrait, Config, 64, 1>,
                           config, input_accessor, shape, rows, initial_reduce, output_accessor,
                           pre_process_op, reduce_op, post_process_op);
        }
    }
}

namespace noa::cuda::utils {
    // (B)DHW reduction.
    // name:            Name of the function. Used for logging if a kernel launch fails.
    // input:           On the device. Input array to reduce.
    // strides:         BDHW strides of input.
    // shape:           BDHW shape of input.
    // output:          On the host or device. Reduced element(s).
    //                  If reduce_batch=false, there should be one element per batch.
    // output_stride:   Stride of the output. This is ignored if reduce_batch=true.
    // initial_reduce:  Per-thread initial value for the reduction.
    // pre_process_op:  Preprocessing operator, op(Input) -> Reduce, or op(Input, Index) -> Reduce.
    // reduce_op:       Reduction operator: op(Reduce, Reduce) -> Reduce.
    // post_process:    Post process operator. op(Reduce) -> Output.
    // reduce_batch:    Whether the batch dimension should be reduced.
    // swap_layout:     Whether the layout can be reordered for maximum performance.
    //                  If reduce_batch=false, only the DHW dimensions can be reordered.
    // stream:          Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // It is the responsibility of the caller to ensure that the input and output stay valid until completion.
    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = ReduceUnaryConfigDefault,
             typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp = noa::copy_t, typename ReduceOp, typename PostProcessOp = noa::copy_t>
    void reduce_unary(const char* name,
                      const Input* input, Strides4<Index> input_strides, Shape4<Index> shape,
                      Output* output, Strides1<Index> output_stride, Reduced initial_reduce,
                      PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
                      bool reduce_batch, bool swap_layout, cuda::Stream& stream) {
        NOA_ASSERT(output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());

        // Rearrange to rightmost order.
        if (swap_layout) {
            if (reduce_batch) {
                const auto order = noa::indexing::order(input_strides, shape);
                shape = noa::indexing::reorder(shape, order);
                input_strides = noa::indexing::reorder(input_strides, order);
            } else {
                const auto order_3d = noa::indexing::order(input_strides.pop_front(), shape.pop_front()) + 1;
                const auto order = order_3d.push_front(0);
                shape = noa::indexing::reorder(shape, order);
                input_strides = noa::indexing::reorder(input_strides, order);
            }
        }

        const u32 batches = reduce_batch ? 1 : safe_cast<u32>(shape[0]);
        const auto elements = reduce_batch ? shape.elements() : shape[1] * shape[2] * shape[3];
        auto is_contiguous = noa::indexing::is_contiguous(input_strides, shape);
        if (!reduce_batch)
            is_contiguous[0] = true; // batches are kept independent

        // The output pointers are allowed to not be on the stream's device,
        // so make sure device memory is allocated for the output.
        using output_unique_t = typename noa::cuda::memory::PtrDevice<Output>::unique_type;
        output_unique_t output_buffer;
        Output* output_ptr = device_pointer(output, stream.device());
        Strides1<Index> output_ptr_stride = output_stride;
        if (!output_ptr) {
            output_ptr_stride = 1;
            output_buffer = noa::cuda::memory::PtrDevice<Output>::alloc(batches, stream);
            output_ptr = output_buffer.get();
        }
        const auto output_accessor = AccessorRestrict<Output, 1, Index>(output_ptr, output_ptr_stride);

        constexpr auto SMALL_THRESHOLD = Config::ELEMENTS_PER_THREAD * Config::BLOCK_SIZE * 4;
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            if (elements <= SMALL_THRESHOLD) {
                details::launch_reduce_unary_small_1d<StridesTrait, Config>(
                        name, input, input_strides, batches, elements, output_accessor, initial_reduce,
                        pre_process_op, reduce_op, post_process_op, stream);
            } else {
                details::launch_reduce_unary_large_1d<StridesTrait, Config>(
                        name, input, input_strides, batches, elements,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            }
        } else {
            if (elements <= SMALL_THRESHOLD) {
                details::launch_reduce_unary_small_4d<StridesTrait, Config>(
                        name, input, input_strides, shape, batches, reduce_batch,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            } else {
                details::launch_reduce_unary_large_4d<StridesTrait, Config>(
                        name, input, input_strides, shape, batches, reduce_batch,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            }
        }

        // A temporary may have been allocated for the device to store the results.
        // In this case, copy back to the original output location.
        if (output_buffer.get() != nullptr) {
            const auto output_shape = Shape4<i64>{1, 1, batches, 1};
            noa::cuda::memory::copy(output_ptr, output_ptr_stride[0], output, output_stride[0], output_shape, stream);
        }
    }

    // Returns the variance of the input array.
    // STD:             Whether the standard deviation should be computed instead.
    // Input:           f32, f64, c32, c64.
    // Output:          If Input is complex, should be the corresponding real type. Otherwise, same as Input.
    // input:           On the device. Input array to reduce.
    // input_strides:   BDHW strides of input.
    // shape:           BDHW shape of input.
    // output:          On the host or device. Output variance(s) (or stddev).
    // output_stride:   Stride of output.
    // ddof:            Delta Degree Of Freedom used to calculate the variance.
    // reduce_batch:    Whether the batch dimension should be reduced too.
    //                  If false, there should be one output value per batch.
    // swap_layout:     Whether the layout can be reordered for maximum performance.
    //                  If reduce_batch=false, only the DHW dimensions can be reordered.
    // stream:          Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // input and output should stay valid until completion.
    template<bool STD, typename Input, typename Output, typename Index>
    void reduce_variance(
            const char* name,
            const Input* input, const Strides4<Index>& input_strides,
            const Shape4<Index>& shape,
            Output* output, Strides1<Index> output_stride,
            i64 ddof, bool reduce_batch, bool swap_layout, Stream& stream) {

        const Index batches = reduce_batch ? 1 : shape[0];
        const auto shape_to_reduce = reduce_batch ? shape : Shape4<Index>{1, shape[1], shape[2], shape[3]};
        const Index elements = shape_to_reduce.elements();

        // Get the sum:
        const auto sums = noa::cuda::memory::PtrPinned<Input>::alloc(batches);
        reduce_unary(name, input, input_strides, shape,
                     sums.get(), Strides1<Index>{1}, Input{0},
                     {}, noa::plus_t{}, {},
                     reduce_batch, swap_layout, stream);
        stream.synchronize();

        // Get the variance:
        const auto inv_count = Output{1} / (static_cast<Output>(elements) - static_cast<Output>(ddof));
        auto post_process_op = [inv_count]__device__(Output dist2) -> Output {
            if constexpr (STD)
                return noa::math::sqrt(dist2 * inv_count);
            return dist2 * inv_count;
        };

        for (i64 batch = 0; batch < batches; ++batch) {
            Input mean = sums[batch] * inv_count;
            auto pre_process_op = [mean]__device__(Input value) -> Output {
                if constexpr (noa::traits::is_complex_v<Input>) {
                    const auto distance = noa::math::abs(value - mean);
                    return distance * distance;
                } else {
                    const auto distance = value - mean;
                    return distance * distance;
                }
                return {}; // unreachable
            };
            reduce_unary(name, input + input_strides[0] * batch, input_strides, shape_to_reduce,
                         output + output_stride[0] * batch, Strides1<Index>{1}, Output{0},
                         pre_process_op, noa::plus_t{}, post_process_op,
                         true, swap_layout, stream);
        }
    }
}
