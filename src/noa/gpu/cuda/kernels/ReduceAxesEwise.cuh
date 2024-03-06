#pragma once

#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

namespace noa::cuda::guts {
    template<typename Config, u32 BlockSizeX, u32 VectorSize>
    struct ReduceAxesEwiseWidthConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 vector_size_x = VectorSize;
        static constexpr u32 n_elements_per_thread_x = max(vector_size_x, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
    };

    template<typename Config, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_width_ewise(Op op, Input input, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto tid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<Index>::from_values(
                blockIdx.z,
                blockIdx.y,
                blockIdx.x * blockDim.y + tid[0],
                tid[1]);
        const bool is_valid_row = gid[2] < shape_hw[0];

        auto input_row = std::move(input).map([&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor);
            else
                return accessor[gid[0]][gid[1]][gid[2]];
        });

        // Initial reduction. Loop until the end of the row is reached.
        for (Index cid = 0; cid < shape_hw[1] and is_valid_row; cid += Config::block_work_size_x) {
            block_reduce_ewise_1d_init
                    <Config::block_size_x, Config::n_elements_per_thread_x, Config::vector_size_x, Config::interface>
                    (op, input_row, shape_hw[1] - cid, reduced, tid[1]);

            // Offset to the next work space.
            input_row.for_each([](auto& accessor) { accessor.offset_accessor(Config::block_work_size_x); });
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Config::block_size];
        Reduced* joined = shared_buffer + tid[0] * Config::block_size_x;
        joined[tid[1]] = reduced;
        block_synchronize();

        // Reduce shared data to one element.
        reduced = block_reduce_shared<Config::interface, Config::block_size_x>(op, joined, tid[1]);
        if (gid[3] == 0 and is_valid_row)
            Config::interface::final(op, reduced, output, gid[0], gid[1], gid[2]);
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesEwiseHeightConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Config, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_height_ewise(Op op, Input input, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto gid = Vec4<Index>::from_values(
                blockIdx.z,
                blockIdx.y,
                threadIdx.y, // one block along the height
                blockIdx.x * Config::block_size_x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape_hw[1];

        // Process every row.
        input.for_each([&](auto& accessor) { accessor.offset_accessor(gid[0], gid[1]); });
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Config::block_size_y)
            Config::interface::init(op, input, reduced, 0, 0, tidy, gid[3]);

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Config::block_size];
        Reduced* joined = shared_buffer + threadIdx.y * Config::block_size_x + threadIdx.x;
        *joined = reduced;
        block_synchronize();

        // Reduce the height of the block.
        #pragma unroll
        for (u32 size = Config::block_size_y; size >= 2; size /= 2) {
            if (threadIdx.y < size / 2)
                Config::interface::join(op, joined[Config::block_size_x * size / 2], *joined);
            block_synchronize();
        }

        if (threadIdx.y == 0 and is_valid_column)
            Config::interface::final(op, *joined, output, gid[0], gid[1], gid[3]);
    }
}
