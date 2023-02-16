#pragma once

#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/memory/PtrPinned.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"

// These reduction kernels are similar to ReduceUnary.cuh, except that
// they take two inputs, and the processor is in charge of combining them.

namespace noa::cuda::utils {
    template<u32 ElementsPerThread, u32 BlockSize, u32 MaxBlockCount>
    struct ReduceBinaryConfig {
        static constexpr u32 ELEMENTS_PER_THREAD = ElementsPerThread;
        static constexpr u32 BLOCK_SIZE = BlockSize;
        static constexpr u32 MAX_GRID_SIZE = MaxBlockCount;
    };
    using ReduceUnaryConfigDefault = ReduceUnaryConfig<8, 512, 4096>;
}

namespace noa::cuda::utils::details {
    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp,
             PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_binary_1d_large(
            Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait> rhs_batched,
            Index elements_per_batch, Reduced initial_reduced,
            AccessorRestrictContiguous<Reduced, 2, Index> tmp_reduced,
            PreProcessOp pre_process_op, ReduceOp reduce_op) {

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        // Batches are kept independent of each other.
        const Index batch = blockIdx.y; // if reduce_batch=true, batch == 0
        const auto lhs = lhs_batched[batch];
        const auto rhs = rhs_batched[batch];

        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        const Index thread_index = threadIdx.x;
        const Index block_index = blockIdx.x;
        const Index block_count = gridDim.x;
        const Index block_offset = block_index * BLOCK_WORK_SIZE;
        const Index grid_work_size = BLOCK_WORK_SIZE * block_count;

        // Initial reduction to bring the input to BLOCK_SIZE * block_count elements.
        Reduced reduced = initial_reduced;
        for (Index cid = block_offset; cid < elements_per_batch; cid += grid_work_size) {
            const Index remaining = elements_per_batch - cid;
            const auto lhs_stride = lhs.template stride<0>();
            const auto rhs_stride = rhs.template stride<0>();
            block_reduce_global_binary<BLOCK_SIZE, EPT, VECTOR_SIZE>(
                    lhs.get() + noa::indexing::at(cid, lhs_stride), lhs_stride,
                    rhs.get() + noa::indexing::at(cid, rhs_stride), rhs_stride,
                    remaining, pre_process_op, reduce_op, &reduced, thread_index);
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

    template<typename Lhs, typename Rhs, typename Reduced, typename Index,
             typename PreProcessOp, typename ReduceOp,
             PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config, u32 BLOCK_DIM_X, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_binary_4d_large(
            Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait> rhs_batched,
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
        const auto lhs = lhs_batched.offset_accessor(batch);
        const auto rhs = rhs_batched.offset_accessor(batch);

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
            const auto lhs_row = lhs[bdh_indexes[0]][bdh_indexes[1]][bdh_indexes[2]];
            const auto rhs_row = rhs[bdh_indexes[0]][bdh_indexes[1]][bdh_indexes[2]];

            // Consume the row:
            for (Index cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const Index remaining = shape[3] - cid;
                const auto lhs_stride = lhs_row.template stride<0>();
                const auto rhs_stride = rhs_row.template stride<0>();
                block_reduce_global_binary<BLOCK_DIM_X, EPT, VECTOR_SIZE>(
                        lhs_row.get() + cid * lhs_stride, lhs_stride,
                        rhs_row.get() + cid * rhs_stride, rhs_stride,
                        remaining, pre_process_op, reduce_op,
                        &reduced, static_cast<Index>(threadIdx.x));
            }
        }

        // Share thread's result to the other threads.
        const Index tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        __shared__ uninitialized_type_t<Reduced> s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<Reduced*>(s_data_);
        s_data[tid] = reduced;
        block_synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        const Reduced final = block_reduce_shared<BLOCK_SIZE>(s_data, tid, reduce_op);
        if (tid == 0)
            tmp_reduced(batch, block_index) = final;
    }

    template<typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp,
             PointerTraits PointerTrait, StridesTraits StridesTrait, typename Config,
             u32 BLOCK_DIM_X, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_binary_4d_small(
            Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait> rhs_batched,
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
        const auto lhs = lhs_batched.offset_accessor(batch);
        const auto rhs = rhs_batched.offset_accessor(batch);

        // This uses 2D blocks:
        const Index rows_per_block = blockDim.y;
        const Index initial_row = threadIdx.y;

        // Initial reduction. Loop until all rows are consumed.
        Reduced reduced = initial_reduce;
        for (Index row = initial_row; row < rows_per_batch; row += rows_per_block) {
            const auto bdh_indexes = noa::indexing::offset2index(row, shape[1], shape[2]); // row -> B,D,H
            const auto lhs_row = lhs[bdh_indexes[0]][bdh_indexes[1]][bdh_indexes[2]];
            const auto rhs_row = rhs[bdh_indexes[0]][bdh_indexes[1]][bdh_indexes[2]];

            // Consume the row:
            for (Index cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const Index remaining = shape[3] - cid;
                const auto lhs_stride = lhs_row.template stride<0>();
                const auto rhs_stride = rhs_row.template stride<0>();
                block_reduce_global_binary<BLOCK_DIM_X, EPT, VECTOR_SIZE>(
                        lhs_row.get() + cid * lhs_stride, lhs_stride,
                        rhs_row.get() + cid * rhs_stride, rhs_stride,
                        remaining, pre_process_op, reduce_op,
                        &reduced, static_cast<Index>(threadIdx.x));
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

    // One block per batch.
    template<typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp,
             PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void reduce_binary_1d_small(
            Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait> rhs_batched,
            Index elements_per_batch, Reduced initial_reduce,
            AccessorRestrict<Output, 1, Index> output,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op) {

        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const Index thread_index = threadIdx.x;
        const Index batch = blockIdx.x; // if reduce_batch=true, batch==0
        const auto lhs = lhs_batched[batch];
        const auto rhs = rhs_batched[batch];

        // elements -> one element per thread.
        Reduced reduced = initial_reduce;
        for (Index cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const Index remaining = elements_per_batch - cid;
            const auto lhs_stride = lhs.template stride<0>();
            const auto rhs_stride = rhs.template stride<0>();
            block_reduce_global_binary<BLOCK_SIZE, EPT, VECTOR_SIZE>(
                    lhs.get() + noa::indexing::at(cid, lhs_stride), lhs_stride,
                    rhs.get() + noa::indexing::at(cid, rhs_stride), rhs_stride,
                    remaining, pre_process_op, reduce_op, &reduced, thread_index);
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

    template<PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config = ReduceUnaryConfigDefault,
             typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_binary_small_1d(
            const char* name,
            const Lhs* lhs, Strides4<Index> lhs_strides,
            const Rhs* rhs, Strides4<Index> rhs_strides, u32 batches, Index elements,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {

        const auto lhs_strides_2d = lhs_strides.filter(0, 3);
        const auto rhs_strides_2d = rhs_strides.filter(0, 3);

        // Try to vectorize the loads for the input.
        auto vector_size = lhs_strides_2d[1] == 1 && rhs_strides_2d[1] == 1 ?
                        std::min({max_vector_count(lhs), max_vector_count(rhs), i64{8}}) : 1;
        if (batches > 1) {
            for (; vector_size >= 2; vector_size /= 2) {
                if (!(lhs_strides_2d[0] % vector_size) && !(rhs_strides_2d[0] % vector_size))
                    break;
            }
        }

        const auto config = LaunchConfig{batches, Config::BLOCK_SIZE};
        if (vector_size > 1) {
            const auto lhs_accessor = AccessorContiguous<const Lhs, 2, Index, PointerTrait>(lhs, lhs_strides_2d);
            const auto rhs_accessor = AccessorContiguous<const Rhs, 2, Index, PointerTrait>(rhs, rhs_strides_2d);
            if (vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_binary_1d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 8>,
                               config, lhs_accessor, rhs_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else if (vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_binary_1d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 4>,
                               config, lhs_accessor, rhs_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else {
                stream.enqueue(name,
                               details::reduce_binary_1d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 2>,
                               config, lhs_accessor, rhs_accessor, elements, initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            }
        } else {
            const auto lhs_accessor = Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait>(lhs, lhs_strides_2d);
            const auto rhs_accessor = Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait>(rhs, rhs_strides_2d);
            stream.enqueue(name,
                           details::reduce_binary_1d_small<
                                   Lhs, Rhs, Reduced, Output, Index,
                                   PreProcessOp, ReduceOp, PostProcessOp,
                                   PointerTrait, StridesTrait, Config, 1>,
                           config, lhs_accessor, rhs_accessor, elements, initial_reduce, output_accessor,
                           pre_process_op, reduce_op, post_process_op);
        }
    }

    template<PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config = ReduceUnaryConfigDefault,
             typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_binary_large_1d(
            const char* name,
            const Lhs* lhs, Strides4<Index> lhs_strides,
            const Rhs* rhs, Strides4<Index> rhs_strides, u32 batches, Index elements,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {

        const auto lhs_strides_2d = lhs_strides.filter(0, 3);
        const auto rhs_strides_2d = rhs_strides.filter(0, 3);

        // Try to vectorize the loads for the input.
        u32 input_vector_size = lhs_strides_2d[1] == 1 && rhs_strides_2d[1] == 1 ?
                        std::min({max_vector_count(lhs), max_vector_count(rhs), i64{8}}) : 1;
        if (batches > 1) {
            for (; input_vector_size >= 2; input_vector_size /= 2) {
                if (!(lhs_strides_2d[0] % input_vector_size) && !(rhs_strides_2d[0] % input_vector_size))
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

        // Large reductions need two kernel launches. One that reduces the input to a bunch of reduced values.
        // And a second, to reduce these values to the final output. Therefore, we need to allocate space
        // for these reduced values. Here, pad so that the second kernel can use vectorized loads.
        constexpr u32 REDUCE_VECTOR_SIZE =
                noa::math::is_power_of_2(sizeof(Reduced)) ?
                noa::math::clamp(i64{16 / sizeof(Reduced)}, i64{1}, i64{8}) : 1;
        const u32 pitch = noa::math::next_multiple_of(blocks.x, REDUCE_VECTOR_SIZE);
        const auto reduced_buffer = noa::cuda::memory::PtrDevice<Reduced>::alloc(pitch * blocks.y, stream);
        const auto reduced_accessor = AccessorRestrictContiguous<Reduced, 2, Index>(
                reduced_buffer.get(), Strides2<Index>{pitch, 1});

        // (batch * elements) -> (blocks.x * blocks.y) elements.
        if (input_vector_size > 1) {
            const auto lhs_accessor = AccessorContiguous<const Lhs, 2, Index, PointerTrait>(lhs, lhs_strides_2d);
            const auto rhs_accessor = AccessorContiguous<const Rhs, 2, Index, PointerTrait>(rhs, rhs_strides_2d);
            if (input_vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_binary_1d_large<
                                       Lhs, Rhs, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 8>,
                               first_config, lhs_accessor, rhs_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (input_vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_binary_1d_large<
                                       Lhs, Rhs, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 4>,
                               first_config, lhs_accessor, rhs_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_binary_1d_large<
                                       Lhs, Rhs, Reduced, Index,
                                       PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 2>,
                               first_config, lhs_accessor, rhs_accessor, elements, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            }
        } else {
            const auto lhs_accessor = Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait>(lhs, lhs_strides_2d);
            const auto rhs_accessor = Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait>(rhs, rhs_strides_2d);
            stream.enqueue(name,
                           details::reduce_binary_1d_large<
                                   Lhs, Rhs, Reduced, Index,
                                   PreProcessOp, ReduceOp,
                                   PointerTrait, StridesTrait, Config, 1>,
                           first_config, lhs_accessor, rhs_accessor, elements, initial_reduce,
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


    template<PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config = ReduceUnaryConfigDefault,
             typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_binary_large_4d(
            const char* name,
            const Lhs* lhs, Strides4<Index> lhs_strides,
            const Rhs* rhs, Strides4<Index> rhs_strides,
            Shape4<Index> shape, u32 batches, bool reduce_batch,
            AccessorRestrict<Output, 1, Index> output_accessor, Reduced initial_reduce,
            PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
            cuda::Stream& stream) {

        // If rows are large, switch to more threads per row.
        const u32 block_dim_x = noa::cuda::Constant::WARP_SIZE * (shape[3] > 512 ? 8 : 2); // FIXME
        const dim3 threads(block_dim_x, std::max(Config::BLOCK_SIZE / block_dim_x, u32{1}));
        const auto rows = safe_cast<u32>(shape[2] * shape[1] * (reduce_batch ? shape[0] : 1));
        const dim3 blocks(noa::math::min(noa::math::divide_up(rows, threads.y), Config::MAX_GRID_SIZE), batches);
        const auto first_config = LaunchConfig{blocks, threads};
        const auto second_config = LaunchConfig{batches, Config::BLOCK_SIZE};

        // Try to vectorize the loads within a row. For that, we have to
        // check that the beginning of each row is at the same alignment.
        u32 vector_size = lhs_strides[3] == 1 && rhs_strides[3] == 1 ?
                       std::min({max_vector_count(lhs), max_vector_count(rhs), i64{8}}) : 1;
        for (; vector_size >= 2; vector_size /= 2) {
            if (((!(lhs_strides[2] % vector_size) && !(rhs_strides[2] % vector_size)) || shape[2] == 1) &&
                ((!(lhs_strides[1] % vector_size) && !(rhs_strides[1] % vector_size)) || shape[1] == 1) &&
                ((!(lhs_strides[0] % vector_size) && !(rhs_strides[0] % vector_size)) || shape[0] == 1))
                break;
        }

        constexpr u32 REDUCE_VECTOR_SIZE =
                noa::math::is_power_of_2(sizeof(Reduced)) ?
                noa::math::clamp(i64{16 / sizeof(Reduced)}, i64{1}, i64{8}) : 1;
        const u32 pitch = noa::math::next_multiple_of(blocks.x, REDUCE_VECTOR_SIZE);
        const auto reduced_buffer = noa::cuda::memory::PtrDevice<Reduced>::alloc(pitch * blocks.y, stream);
        const auto reduced_accessor = AccessorRestrictContiguous<Reduced, 2, Index>(
                reduced_buffer.get(), Strides2<Index>{pitch, 1});

        const auto lhs_accessor = Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait>(lhs, lhs_strides);
        const auto rhs_accessor = Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait>(rhs, rhs_strides);
        if (threads.x == 256) {
            if (vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 256, 8>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 256, 4>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (vector_size == 2) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 256, 2>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 256, 1>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            }
        } else {
            if (vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 64, 8>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 64, 4>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else if (vector_size == 2) {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 64, 2>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
                               reduced_accessor, pre_process_op, reduce_op);
            } else {
                stream.enqueue(name,
                               details::reduce_binary_4d_large<
                                       Lhs, Rhs, Reduced, Index, PreProcessOp, ReduceOp,
                                       PointerTrait, StridesTrait, Config, 64, 1>,
                               first_config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce,
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

    template<PointerTraits PointerTrait, StridesTraits StridesTrait,
             typename Config = ReduceUnaryConfigDefault,
             typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    void launch_reduce_binary_small_4d(
            const char* name,
            const Lhs* lhs, Strides4<Index> lhs_strides,
            const Rhs* rhs, Strides4<Index> rhs_strides,
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
        u32 vector_size = lhs_strides[3] == 1 && rhs_strides[3] == 1 ?
                       std::min({max_vector_count(lhs), max_vector_count(rhs), i64{8}}) : 1;
        for (; vector_size >= 2; vector_size /= 2) {
            if (((!(lhs_strides[2] % vector_size) && !(rhs_strides[2] % vector_size)) || shape[2] == 1) &&
                ((!(lhs_strides[1] % vector_size) && !(rhs_strides[1] % vector_size)) || shape[1] == 1) &&
                ((!(lhs_strides[0] % vector_size) && !(rhs_strides[0] % vector_size)) || shape[0] == 1))
                break;
        }

        if (vector_size > 1) {
            const auto lhs_accessor = AccessorContiguous<const Lhs, 4, Index, PointerTrait>(lhs, lhs_strides);
            const auto rhs_accessor = AccessorContiguous<const Rhs, 4, Index, PointerTrait>(rhs, rhs_strides);
            if (vector_size == 8) {
                stream.enqueue(name,
                               details::reduce_binary_4d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 64, 8>,
                               config, lhs_accessor, rhs_accessor, shape, rows,
                               initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else if (vector_size == 4) {
                stream.enqueue(name,
                               details::reduce_binary_4d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 64, 4>,
                               config, lhs_accessor, rhs_accessor, shape, rows,
                               initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            } else {
                stream.enqueue(name,
                               details::reduce_binary_4d_small<
                                       Lhs, Rhs, Reduced, Output, Index,
                                       PreProcessOp, ReduceOp, PostProcessOp,
                                       PointerTrait, StridesTraits::CONTIGUOUS, Config, 64, 2>,
                               config, lhs_accessor, rhs_accessor, shape, rows,
                               initial_reduce, output_accessor,
                               pre_process_op, reduce_op, post_process_op);
            }
        } else {
            const auto lhs_accessor = Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait>(lhs, lhs_strides);
            const auto rhs_accessor = Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait>(rhs, rhs_strides);
            stream.enqueue(name,
                           details::reduce_binary_4d_small<
                                   Lhs, Rhs, Reduced, Output, Index,
                                   PreProcessOp, ReduceOp, PostProcessOp,
                                   PointerTrait, StridesTrait, Config, 64, 1>,
                           config, lhs_accessor, rhs_accessor, shape, rows, initial_reduce, output_accessor,
                           pre_process_op, reduce_op, post_process_op);
        }
    }
}

namespace noa::cuda::utils {
    // (B)DHW reduction.
    // name:            Name of the function. Used for logging if a kernel launch fails.
    // lhs:             On the device. Lhs array to reduce.
    // rhs:             On the device. Lhs array to reduce.
    // lhs_strides:     BDHW strides of lhs.
    // rhs_strides:     BDHW strides of rhs.
    // shape:           BDHW shape of input arrays.
    // output:          On the host or device. Reduced element(s).
    //                  If reduce_batch=false, there should be one element per batch.
    // output_stride:   Stride of the output. This is ignored if reduce_batch=true.
    // initial_reduce:  Per-thread initial value for the reduction.
    // pre_process_op:  Preprocessing operator, op(Lhs, Rhs) -> Reduce.
    // reduce_op:       Reduction operator: op(Reduce, Reduce) -> Reduce.
    // post_process:    Post process operator. op(Reduce) -> Output.
    // reduce_batch:    Whether the batch dimension should be reduced.
    // swap_layout:     Whether the layout can be reordered for maximum performance.
    //                  If reduce_batch=false, only the DHW dimensions can be reordered.
    // stream:          Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // It is the responsibility of the caller to ensure that the inputs and output stay valid until completion.
    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = ReduceUnaryConfigDefault,
             typename Lhs, typename Rhs, typename Reduced, typename Output, typename Index,
             typename PreProcessOp = noa::copy_t, typename ReduceOp, typename PostProcessOp = noa::copy_t>
    void reduce_binary(const char* name,
                       const Lhs* lhs, Strides4<Index> lhs_strides,
                       const Rhs* rhs, Strides4<Index> rhs_strides, Shape4<Index> shape,
                       Output* output, Strides1<Index> output_stride, Reduced initial_reduce,
                       PreProcessOp pre_process_op, ReduceOp reduce_op, PostProcessOp post_process_op,
                       bool reduce_batch, bool swap_layout, cuda::Stream& stream) {
        NOA_ASSERT(output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());

        // Rearrange to rightmost order.
        if (swap_layout) {
            if (reduce_batch) {
                const auto lhs_order = noa::indexing::order(lhs_strides, shape);
                const auto rhs_order = noa::indexing::order(rhs_strides, shape);
                if (noa::all(lhs_order == rhs_order) && noa::any(lhs_order != Vec4<Index>{0, 1, 2, 3})) {
                    shape = noa::indexing::reorder(shape, lhs_order);
                    lhs_strides = noa::indexing::reorder(lhs_strides, lhs_order);
                    rhs_strides = noa::indexing::reorder(rhs_strides, lhs_order);
                }
            } else {
                const auto shape_3d = shape.pop_front();
                const auto lhs_order_3d = noa::indexing::order(lhs_strides.pop_front(), shape_3d) + 1;
                const auto rhs_order_3d = noa::indexing::order(rhs_strides.pop_front(), shape_3d) + 1;
                if (noa::all(lhs_order_3d == rhs_order_3d) && noa::any(lhs_order_3d != Vec3<Index>{1, 2, 3})) {
                    const auto order = lhs_order_3d.push_front(0);
                    shape = noa::indexing::reorder(shape, order);
                    lhs_strides = noa::indexing::reorder(lhs_strides, order);
                    rhs_strides = noa::indexing::reorder(rhs_strides, order);
                }
            }
        }

        const u32 batches = reduce_batch ? 1 : safe_cast<u32>(shape[0]);
        const auto elements = reduce_batch ? shape.elements() : shape[1] * shape[2] * shape[3];
        auto are_contiguous =
                noa::indexing::is_contiguous(lhs_strides, shape) &&
                noa::indexing::is_contiguous(rhs_strides, shape);
        if (!reduce_batch)
            are_contiguous[0] = true; // batches are kept independent

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
        if (are_contiguous[0] && are_contiguous[1] && are_contiguous[2]) {
            if (elements <= SMALL_THRESHOLD) {
                details::launch_reduce_binary_small_1d<PointerTrait, StridesTrait, Config>(
                        name, lhs, lhs_strides, rhs, rhs_strides, batches, elements,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            } else {
                details::launch_reduce_binary_large_1d<PointerTrait, StridesTrait, Config>(
                        name, lhs, lhs_strides, rhs, rhs_strides, batches, elements,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            }
        } else {
            if (elements <= SMALL_THRESHOLD) {
                details::launch_reduce_binary_small_4d<PointerTrait, StridesTrait, Config>(
                        name, lhs, lhs_strides, rhs, rhs_strides, shape, batches, reduce_batch,
                        output_accessor, initial_reduce, pre_process_op, reduce_op, post_process_op, stream);
            } else {
                details::launch_reduce_binary_large_4d<PointerTrait, StridesTrait, Config>(
                        name, lhs, lhs_strides, rhs, rhs_strides, shape, batches, reduce_batch,
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
}
