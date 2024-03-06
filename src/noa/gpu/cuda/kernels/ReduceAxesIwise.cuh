#pragma once

#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

namespace noa::cuda::guts {
    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesIwiseWidthConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<size_t N, typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_width_iwise(Op op, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto tid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<Index>::from_values(
                blockIdx.z, // always 0 if N < 4
                blockIdx.y, // always 0 if N < 3
                blockIdx.x * blockDim.y + tid[0],
                tid[1]);
        const bool is_valid_row = gid[2] < shape_hw[0];

        // Initial reduction. Loop until the end of the row is reached.
        for (Index cid = gid[3]; cid < shape_hw[1] and is_valid_row; cid += Config::block_size_x) {
            if constexpr (N == 4)
                Config::interface::init(op, reduced, gid[0], gid[1], gid[2], cid);
            else if constexpr (N == 3)
                Config::interface::init(op, reduced, gid[1], gid[2], cid);
            else
                static_assert(nt::always_false_v<Config>);
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Config::block_size];
        Reduced* joined = shared_buffer + tid[0] * Config::block_size_x;
        joined[tid[1]] = reduced;
        block_synchronize();

        // Reduce each "shared row" to one element.
        reduced = block_reduce_shared<Config::interface, Config::block_size_x>(op, joined, tid[1]);
        if (gid[3] == 0 and is_valid_row) {
            if constexpr (N == 4)
                Config::interface::final(op, reduced, output, gid[0], gid[1], gid[2]);
            else if constexpr (N == 3)
                Config::interface::final(op, reduced, output, gid[1], gid[2]);
            else
                static_assert(nt::always_false_v<Config>);
        }
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesIwiseHeightConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<size_t N, size_t R, typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_height_iwise(Op op, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto gid = Vec4<Index>::from_values(
                blockIdx.z, // always 0 if N < 4
                blockIdx.y, // always 0 if N < 3
                threadIdx.y, // one block along the height
                blockIdx.x * Config::block_size_x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape_hw[1];

        // Process every row.
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Config::block_size_y) {
            if constexpr (N == 4) {
                if constexpr (R == 0) {
                    Config::interface::init(op, reduced, tidy, gid[0], gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Config::interface::init(op, reduced, gid[0], tidy, gid[1], gid[3]);
                } else if constexpr (R == 2) {
                    Config::interface::init(op, reduced, gid[0], gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false_v<Index>);
                }
            } else if constexpr (N == 3) {
                if constexpr (R == 0) {
                    Config::interface::init(op, reduced, tidy, gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Config::interface::init(op, reduced, gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false_v<Index>);
                }
            } else if constexpr (N == 2 and R == 0) {
                Config::interface::init(op, reduced, tidy, gid[3]);
            } else {
                static_assert(nt::always_false_v<Config>);
            }
        }

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

        if (threadIdx.y == 0 and is_valid_column) {
            if constexpr (N == 4)
                Config::interface::final(op, *joined, output, gid[0], gid[1], gid[3]);
            else if constexpr (N == 3)
                Config::interface::final(op, *joined, output, gid[1], gid[3]);
            else if constexpr (N == 2)
                Config::interface::final(op, *joined, output, gid[3]);
            else
                static_assert(nt::always_false_v<Config>);
        }
    }

    template<typename Config, size_t N> requires (N == 1 or N == 2)
    struct ReduceAxesIwiseBlockConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = max(Constant::WARP_SIZE, Config::block_size);
        static constexpr u32 block_size_x = N == 1 ? block_size : Constant::WARP_SIZE;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Config, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_4d_first(
            Op op,
            Reduced reduced,
            Joined joined, // 2d Accessor of Reduced
            Vec3<Index> shape_dhw,
            Vec2<u32> n_blocks_hw
    ) {
        // Get the position within the 4d span.
        const Index cb = blockIdx.z;
        const Vec2<u32> index = ni::offset2index(blockIdx.x, n_blocks_hw[1]);
        const auto gid = Vec3<Index>::from_values(
                blockIdx.y,
                Config::block_size_y * index[0] + threadIdx.y,
                Config::block_size_x * index[1] + threadIdx.x
        );

        // Traverse the entire 3d span of this batch.
        for (Index cd = gid[0]; cd < shape_dhw[0]; cd += gridDim.y)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Config::block_size_y * n_blocks_hw[0])
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Config::block_size_x * n_blocks_hw[1])
                    Config::interface::init(op, reduced, cb, cd, ch, cw);

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, cb, bid);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_3d_first(
            Op op,
            Reduced reduced,
            Joined joined, // 2d Accessor of Reduced
            Vec2<Index> shape_hw
    ) {
        const Index cd = blockIdx.z;
        const auto gid = Vec2<Index>::from_values(
                Config::block_size_y * blockIdx.y + threadIdx.y,
                Config::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Config::block_size_y * gridDim.y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Config::block_size_x * gridDim.x)
                Config::interface::init(op, reduced, cd, ch, cw);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, cd, bid);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_2d_first(
            Op op,
            Reduced reduced,
            Joined joined,
            Vec1<Index> shape
    ) {
        static_assert(Config::block_size_y == 1);
        const Index ch = blockIdx.z;
        const Index gid = Config::block_size_x * blockIdx.x + threadIdx.x;

        for (Index cw = gid; cw < shape[0]; cw += Config::block_size_x * gridDim.x)
            Config::interface::init(op, reduced, ch, cw);

        const Index tid = threadIdx.x;
        const Index bid = blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, ch, bid);
    }

    template<typename Config, typename Op, typename Index, typename Joined, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_second(
            Op op,
            Joined to_join,
            Index n_to_join,
            Reduced reduced,
            Output output
    ) {
        const Index batch = blockIdx.x;
        const Index tid = threadIdx.x;
        for (Index cid = tid; cid < n_to_join; cid += Config::block_size)
            Config::interface::join(op, to_join(batch, cid), reduced);
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, batch);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_4d_small(Op op, Reduced reduced, Output output, Vec3<Index> shape_dhw) {
        const Index cb = blockIdx.x;
        const auto gid = Vec3<Index>::from_values(0, threadIdx.y, threadIdx.x);

        for (Index cd = gid[0]; cd < shape_dhw[0]; ++cd)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Config::block_size_y)
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Config::block_size_x)
                    Config::interface::init(op, reduced, cb, cd, ch, cw);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, cb);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_3d_small(Op op, Reduced reduced, Output output, Vec2<Index> shape_hw) {
        const Index cd = blockIdx.x;
        const auto gid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);

        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Config::block_size_y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Config::block_size_x)
                Config::interface::init(op, reduced, cd, ch, cw);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, cd);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_axes_iwise_2d_small(Op op, Reduced reduced, Output output, Vec1<Index> shape) {
        static_assert(Config::block_size_y == 1);
        const Index ch = blockIdx.x;
        const Index tid = threadIdx.x;

        for (Index cw = tid; cw < shape[0]; cw += Config::block_size_x)
            Config::interface::init(op, reduced, ch, cw);

        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, ch);
    }
}
