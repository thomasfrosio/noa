#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

// These reduction kernels are adapted from different sources, but the main logic comes from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::guts {
    // 1d grid of 1d blocks.
    // Each block writes one element in "joined", which should thus have as many elements as there are blocks.
    template<typename Config, u32 VectorSize, bool IsFinal = false>
    struct ReduceEwise2dConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 vector_size = VectorSize;
        static constexpr u32 n_elements_per_thread = max(VectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool is_final = IsFinal;
    };

    template<typename Config, u32 BlockSizeX, u32 VectorSize, bool IsFinal = false>
    struct ReduceEwise4dConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = max(block_size / block_size_x, 1u);
        static constexpr u32 vector_size_x = VectorSize;
        static constexpr u32 n_elements_per_thread_x = max(vector_size_x, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr bool is_final = IsFinal;
    };

    // Reduce element-wise 1d or 2d input accessors.
    // 2d grid (y is the batch) of 1d blocks.
    template<typename Config, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_2d(Op op, Input input, Index n_elements_per_batch, Reduced reduced, Output output) {
        const Index batch = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index tid = threadIdx.x;
        const Index starting_index = Config::block_work_size * bid;
        const Index grid_work_size = Config::block_work_size * gridDim.x;

        auto input_1d = std::move(input).map([batch]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor);
            } else if constexpr (nt::is_accessor_nd_v<T, 2>) {
                // Offset the input accessor to the current batch,
                // so that it can be used later to reset the pointer.
                accessor.offset_accessor(batch);
                return accessor[0]; // 1d AccessorReference
            } else {
                static_assert(nt::always_false_v<T>);
            }
        });

        for (Index cid = starting_index; cid < n_elements_per_batch; cid += grid_work_size) {
            input_1d.for_each_enumerate([&input, cid]<size_t I>(auto& accessor) {
                accessor.reset_pointer(input[Tag<I>{}].get());
                accessor.offset_accessor(cid);
            });
            block_reduce_ewise_1d_init
                    <Config::block_size, Config::n_elements_per_thread, Config::vector_size, Config::interface>
                    (op, input_1d, n_elements_per_batch - cid, reduced, tid);
        }

        if constexpr (Config::is_final) {
            // There's one block per batch, so compute the reduced value for the block
            // and save it in the output at the batch index.
            block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
        } else {
            // The output is the "joined" buffer, which is a buffer with one value per block and per batch.
            // These values will then be reduced by the second reduction kernel (see below).
            block_reduce_join<Config::interface, Config::block_size>(op, reduced, output, tid, batch, bid);
        }
    }

    // Here the input is organized has a series of rows. Given the original DHW shape of the input and the row index,
    // we can derive the BDH indices. Each dimension can have an arbitrary stride, but if the rows themselves are
    // contiguous (if the W stride is 1), then vectorized load/stores can be used to load/store elements from the rows.
    //
    // This kernel explicitly supports per-batch reductions (see reduce_axes_ewise), in which case the grid should be
    // 2d (gridDim.x is the number of blocks to reduce the rows of a given batch and gridDim.y is the number of batches)
    // and joined should be 2d Accessors, where the outer dimension is the batch.
    template<typename Config, typename Op, typename Index,
             typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_4d(
            Op op,
            Input input,
            Reduced reduced,
            Shape3<Index> shape_dhw,
            Index n_rows,
            Output output
    ) {
        const Index batch = blockIdx.y;
        const Index bid = blockIdx.x;
        const Index n_blocks_per_batch = gridDim.x;
        const Index n_rows_per_block = blockDim.y;
        const Index n_rows_per_grid = n_blocks_per_batch * n_rows_per_block;
        const Index initial_row = bid * n_rows_per_block + threadIdx.y;

        auto input_1d = std::move(input).map([batch]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor);
            } else {
                // Offset the input accessor to the current batch,
                // so that it can be used later to reset the pointer.
                accessor.offset_accessor(batch);
                return accessor[0][0][0]; // 1d AccessorReference
            }
        });

        for (Index row = initial_row; row < n_rows; row += n_rows_per_grid) { // for every row (within a batch)
            // If there batched are fused (gridDim.y==0), bdh[0] is always 0.
            Vec3<Index> bdh = ni::offset2index(row, shape_dhw[0], shape_dhw[1]);

            for (Index cid = 0; cid < shape_dhw[2]; cid += Config::block_work_size_x) { // consume the row
                input_1d.for_each_enumerate([&input, &bdh, &cid]<size_t I>(auto& accessor_1d) {
                    if constexpr (not nt::is_accessor_value_v<decltype(accessor_1d)>) {
                        auto& accessor = input[Tag<I>{}];
                        auto new_pointer = accessor.offset_pointer(accessor.get(), bdh[0], bdh[1], bdh[2], cid);
                        accessor_1d.reset_pointer(new_pointer);
                    }
                });
                block_reduce_ewise_1d_init
                        <Config::block_size_x, Config::n_elements_per_thread_x, Config::vector_size_x, Config::interface>
                        (op, input_1d, shape_dhw[2] - cid, reduced, static_cast<Index>(threadIdx.x));
            }
        }

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        if constexpr (Config::is_final)
            block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
        else
            block_reduce_join<Config::interface, Config::block_size>(op, reduced, output, tid, batch, bid);
    }

    // One 1d block per batch to finish joining the reduced values and compute the final output.
    template<typename Config, typename Op, typename Index,
             typename Joined, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_ewise_second(
            Op op,
            Joined joined, // Tuple of 2d Accessor(s) corresponding to Reduced
            Index n_elements,
            Reduced reduced, // Tuple of AccessorValue(s)
            Output output // Tuple of 1d Accessor(s)
    ) {
        const Index batch = blockIdx.x;
        const Index tid = threadIdx.x;

        auto joined_1d = joined.map([&](auto& accessor) { return accessor[batch]; });
        for (Index cid = 0; cid < n_elements; cid += Config::block_work_size) {
            block_reduce_ewise_1d_join
                    <Config::block_size, Config::n_elements_per_thread, Config::vector_size, Config::interface>
                    (op, joined_1d, n_elements - cid, reduced, tid);
            joined_1d.for_each([](auto& accessor) { accessor.offset_accessor(Config::block_work_size); });
        }

        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
    }
}
