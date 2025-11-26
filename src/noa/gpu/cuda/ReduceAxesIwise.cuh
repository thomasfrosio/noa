#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/ReduceIwise.cuh"

namespace noa::cuda::details {
    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesIwiseWidthBlock {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<size_t N, typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_width_iwise(Op op, Shape2<Index> shape_hw, Reduced reduced, Output output, Vec<u32, N - 2> grid_offset) {
        // Multigrid case.
        auto bid = block_indices<u32, 3>();
        if constexpr (N == 4)
            bid[0] += grid_offset[0];
        if constexpr (N >= 3)
            bid[1] += grid_offset[N - 3];

        // Global indices.
        const auto tid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<Index>::from_values(
            bid[0], // always 0 if N < 4
            bid[1], // always 0 if N < 3
            bid[2] * Block::block_size_y + tid[0],
            tid[1]);
        const bool is_valid_row = gid[2] < shape_hw[0];

        // Initial reduction. Loop until the end of the row is reached.
        for (Index cid = gid[3]; cid < shape_hw[1] and is_valid_row; cid += Block::block_size_x) {
            if constexpr (N == 4)
                Interface::init(op, reduced, gid[0], gid[1], gid[2], cid);
            else if constexpr (N == 3)
                Interface::init(op, reduced, gid[1], gid[2], cid);
            else
                static_assert(nt::always_false<Block>);
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Block::block_size];
        Reduced* joined = shared_buffer + tid[0] * Block::block_size_x;
        joined[tid[1]] = reduced;
        block_synchronize();

        // Reduce each "shared row" to one element.
        reduced = block_reduce_shared<Interface, Block::block_size_x>(op, joined, tid[1]);
        if (gid[3] == 0 and is_valid_row) {
            if constexpr (N == 4)
                Interface::final(op, reduced, output, gid[0], gid[1], gid[2]);
            else if constexpr (N == 3)
                Interface::final(op, reduced, output, gid[1], gid[2]);
            else
                static_assert(nt::always_false<Interface>);
        }
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesIwiseHeightBlock {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<size_t N, size_t R, typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_height_iwise(Op op, Shape2<Index> shape_hw, Reduced reduced, Output output, Vec<u32, N - 2> grid_offset) {
        // Multigrid case.
        auto bid = block_indices<u32, 3>();
        if constexpr (N == 4)
            bid[0] += grid_offset[0];
        if constexpr (N >= 3)
            bid[1] += grid_offset[N - 3];

        // Global indices.
        const auto gid = Vec4<Index>::from_values(
            bid[0], // always 0 if N < 4
            bid[1], // always 0 if N < 3
            threadIdx.y, // one block along the height
            bid[2] * Block::block_size_x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape_hw[1];

        // Process every row.
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Block::block_size_y) {
            if constexpr (N == 4) {
                if constexpr (R == 0) {
                    Interface::init(op, reduced, tidy, gid[0], gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Interface::init(op, reduced, gid[0], tidy, gid[1], gid[3]);
                } else if constexpr (R == 2) {
                    Interface::init(op, reduced, gid[0], gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false<Interface>);
                }
            } else if constexpr (N == 3) {
                if constexpr (R == 0) {
                    Interface::init(op, reduced, tidy, gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Interface::init(op, reduced, gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false<Interface>);
                }
            } else if constexpr (N == 2 and R == 0) {
                Interface::init(op, reduced, tidy, gid[3]);
            } else {
                static_assert(nt::always_false<Interface>);
            }
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Block::block_size];
        Reduced* joined = shared_buffer + threadIdx.y * Block::block_size_x + threadIdx.x;
        *joined = reduced;
        block_synchronize();

        // Reduce the height of the block.
        #pragma unroll
        for (u32 size = Block::block_size_y; size >= 2; size /= 2) {
            if (threadIdx.y < size / 2)
                Interface::join(op, joined[Block::block_size_x * size / 2], *joined);
            block_synchronize();
        }

        if (threadIdx.y == 0 and is_valid_column) {
            if constexpr (N == 4)
                Interface::final(op, *joined, output, gid[0], gid[1], gid[3]);
            else if constexpr (N == 3)
                Interface::final(op, *joined, output, gid[1], gid[3]);
            else if constexpr (N == 2)
                Interface::final(op, *joined, output, gid[3]);
            else
                static_assert(nt::always_false<Interface>);
        }
    }

    template<typename Config, size_t N> requires (N == 1 or N == 2)
    struct ReduceAxesIwiseBlock {
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = max(Constant::WARP_SIZE, Config::block_size);
        static constexpr u32 block_size_x = N == 1 ? block_size : Constant::WARP_SIZE;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_4d_first(
        Op op,
        Reduced reduced,
        Joined joined, // 2d Accessor of Reduced
        Vec3<Index> shape_dhw,
        Vec2<u32> n_blocks_hw,
        u32 offset
    ) {
        // Get the position within the 4d span.
        const Index cb = blockIdx.z + offset;
        const Vec2<u32> index = ni::offset2index(blockIdx.x, n_blocks_hw[1]);
        const auto gid = Vec3<Index>::from_values(
            blockIdx.y,
            Block::block_size_y * index[0] + threadIdx.y,
            Block::block_size_x * index[1] + threadIdx.x
        );

        // Traverse the entire 3d span of this batch.
        for (Index cd = gid[0]; cd < shape_dhw[0]; cd += gridDim.y)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Block::block_size_y * n_blocks_hw[0])
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Block::block_size_x * n_blocks_hw[1])
                    Interface::init(op, reduced, cb, cd, ch, cw);

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_reduce_join<Interface, Block::block_size>(op, reduced, joined, tid, cb, bid);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_3d_first(
        Op op,
        Reduced reduced,
        Joined joined, // 2d Accessor of Reduced
        Vec2<Index> shape_hw,
        u32 offset
    ) {
        const Index cd = blockIdx.z + offset;
        const auto gid = Vec2<Index>::from_values(
            Block::block_size_y * blockIdx.y + threadIdx.y,
            Block::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Block::block_size_y * gridDim.y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Block::block_size_x * gridDim.x)
                Interface::init(op, reduced, cd, ch, cw);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_reduce_join<Interface, Block::block_size>(op, reduced, joined, tid, cd, bid);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Joined>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_2d_first(
        Op op,
        Reduced reduced,
        Joined joined,
        Vec1<Index> shape,
        u32 offset
    ) {
        static_assert(Block::block_size_y == 1);
        const Index ch = blockIdx.z + offset;
        const Index gid = Block::block_size_x * blockIdx.x + threadIdx.x;

        for (Index cw = gid; cw < shape[0]; cw += Block::block_size_x * gridDim.x)
            Interface::init(op, reduced, ch, cw);

        const Index tid = threadIdx.x;
        const Index bid = blockIdx.x;
        block_reduce_join<Interface, Block::block_size>(op, reduced, joined, tid, ch, bid);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Joined, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_second(
        Op op,
        Joined to_join,
        Index n_to_join,
        Reduced reduced,
        Output output
    ) {
        const Index batch = blockIdx.x;
        const Index tid = threadIdx.x;
        for (Index cid = tid; cid < n_to_join; cid += Block::block_size)
            Interface::join(op, to_join(batch, cid), reduced);
        block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, batch);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_4d_small(Op op, Reduced reduced, Output output, Vec3<Index> shape_dhw) {
        const Index cb = blockIdx.x;
        const auto gid = Vec3<Index>::from_values(0, threadIdx.y, threadIdx.x);

        for (Index cd = gid[0]; cd < shape_dhw[0]; ++cd)
            for (Index ch = gid[1]; ch < shape_dhw[1]; ch += Block::block_size_y)
                for (Index cw = gid[2]; cw < shape_dhw[2]; cw += Block::block_size_x)
                    Interface::init(op, reduced, cb, cd, ch, cw);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, cb);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_3d_small(Op op, Reduced reduced, Output output, Vec2<Index> shape_hw) {
        const Index cd = blockIdx.x;
        const auto gid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);

        for (Index ch = gid[0]; ch < shape_hw[0]; ch += Block::block_size_y)
            for (Index cw = gid[1]; cw < shape_hw[1]; cw += Block::block_size_x)
                Interface::init(op, reduced, cd, ch, cw);

        const Index tid = threadIdx.y * Block::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, cd);
    }

    template<typename Block, typename Interface, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_axes_iwise_2d_small(Op op, Reduced reduced, Output output, Vec1<Index> shape) {
        static_assert(Block::block_size_y == 1);
        const Index ch = blockIdx.x;
        const Index tid = threadIdx.x;

        for (Index cw = tid; cw < shape[0]; cw += Block::block_size_x)
            Interface::init(op, reduced, ch, cw);

        block_reduce_join_and_final<Interface, Block::block_size>(op, reduced, output, tid, ch);
    }
}

namespace noa::cuda::details {
    template<typename Config, size_t N>
    auto reduce_axes_iwise_nd_first_config(const Shape<i64, N>& shape) {
        constexpr auto max_grid_size = static_cast<i64>(Config::max_grid_size);
        constexpr auto block_size = Vec3<i64>::from_values(1, Config::block_size_y, Config::block_size_x);
        static_assert(N >= 2 or Config::block_size_y == 1);

        // Set the number of blocks while keeping the total number of blocks under the maximum allowed.
        auto n_blocks = Vec3<i64>::from_value(1);
        const auto shape_whd = shape.flip();
        for (i64 i = 0; i <  static_cast<i64>(N); ++i) {
            const auto n_blocks_allowed = max_grid_size / product(n_blocks);
            n_blocks[2 - i] = min(divide_up(shape_whd[i], block_size[2 - i]), n_blocks_allowed);
        }

        const auto n_blocks_u32 = n_blocks.as<u32>();
        const auto n_threads = dim3(block_size[2], block_size[1], 1);
        const auto n_blocks_dim3 = [&] {
            if constexpr (N == 3)
                return dim3(n_blocks_u32[2] * n_blocks_u32[1], n_blocks_u32[0]);
            else
                return dim3(n_blocks_u32[2], n_blocks_u32[1]);
        }();

        auto config = LaunchConfig{.n_blocks=n_blocks_dim3, .n_threads=n_threads};
        auto n_blocks_yx = Vec{n_blocks_u32[1], n_blocks_u32[2]};
        return make_tuple(config, product(n_blocks_u32), n_blocks_yx);
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_4d(
        const Shape<Index, 4>& input_shape,
        Vec<bool, 4> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Interface = Config::interface;

        if (axes_to_reduce[3]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(2, 3);

            // The width of the output is empty/reduced, remove it.
            constexpr auto TO_3D = nd::AccessorConfig<3>{.filter={0, 1, 2}};
            auto output_3d = nd::reconfig_accessors<TO_3D>(output);
            using Output3D = decltype(output_3d);

            // Block shape.
            u32 n_threads_x = shape_u32[3] > 512 ? 256u : 64u;
            if (not is_multiple_of(Config::block_size, n_threads_x))
                n_threads_x = Constant::WARP_SIZE;
            const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const auto n_threads = dim3(n_threads_x, n_threads_y);

            // Grid shape.
            const auto grid_x = GridX(shape_u32[2], n_threads_y);
            const auto grid_y = GridY(shape_u32[1], 1);
            const auto grid_z = GridZ(shape_u32[0], 1);
            check(grid_x.n_launches() == 1);

            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = n_threads,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    if (n_threads_x == 256) {
                        using Block = ReduceAxesIwiseWidthBlock<Config, 256>;
                        stream.enqueue(
                            reduce_width_iwise<4, Block, Interface, OpDecay, Index, ReducedDecay, Output3D>,
                            config, op, shape_hw, reduced, output_3d, grid_offset
                        );
                    } else {
                        using Block = ReduceAxesIwiseWidthBlock<Config, 64>;
                        stream.enqueue(
                            reduce_width_iwise<4, Block, Interface, OpDecay, Index, ReducedDecay, Output3D>,
                            config, op, shape_hw, reduced, output_3d, grid_offset
                        );
                    }
                }
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = ni::squeeze_left(axes_to_reduce.as<i32>() + 1);
            order = order.filter(0, 1, 3, 2); // move the width back to rightmost

            // Reorder to (X, X, axis_to_reduce, width).
            auto reordered_shape = input_shape.reorder(order);
            auto reordered_output = output.map([&order](auto accessor) {
                accessor.reorder(order);
                return accessor;
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto TO_3D = nd::AccessorConfig<3>{.filter = {0, 1, 3}};
            auto reordered_output_3d = nd::reconfig_accessors<TO_3D>(reordered_output);
            using ReorderedOutput3D = decltype(reordered_output_3d);

            // Block shape.
            constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
            constexpr u32 N_THREADS_Y = max(Config::block_size, N_THREADS_X) / N_THREADS_X;
            const auto n_threads = dim3(N_THREADS_X, N_THREADS_Y);

            // Grid shape.
            const auto grid_x = GridX(reordered_shape[3], N_THREADS_X);
            const auto grid_y = GridY(reordered_shape[1], 1);
            const auto grid_z = GridZ(reordered_shape[0], 1);
            check(grid_x.n_launches() == 1);

            using Block = ReduceAxesIwiseHeightBlock<Config, N_THREADS_X>;
            const auto shape_2d = reordered_shape.template pop_front<2>();

            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = n_threads,
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    if (axes_to_reduce[2]) {
                        stream.enqueue(
                            reduce_height_iwise<4, 2, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d, grid_offset
                        );
                    } else if (axes_to_reduce[1]) {
                        stream.enqueue(
                            reduce_height_iwise<4, 1, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d, grid_offset
                        );
                    } else {
                        stream.enqueue(
                            reduce_height_iwise<4, 0, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput3D>,
                            config, op, shape_2d, reduced, reordered_output_3d, grid_offset
                        );
                    }
                }
            }
        }
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_3d(
        const Shape3<Index>& input_shape,
        Vec3<bool> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Interface = Config::interface;

        if (axes_to_reduce[2]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(1, 2);

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_2d = nd::AccessorConfig<2>{.filter={0, 1}};
            auto output_2d = nd::reconfig_accessors<to_2d>(output);
            using Output2D = decltype(output_2d);

            // Block shape.
            u32 n_threads_x = shape_u32[2] > 512 ? 256u : 64u;
            if (not is_multiple_of(Config::block_size, n_threads_x))
                n_threads_x = Constant::WARP_SIZE;
            const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const auto n_threads = dim3(n_threads_x, n_threads_y);

            // Grid shape.
            const auto grid_x = GridX(shape_u32[1], n_threads_y);
            const auto grid_y = GridY(shape_u32[0], 1);
            check(grid_x.n_launches() == 1);

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                    .n_threads = n_threads,
                };
                const auto grid_offset = Vec{grid_y.offset(y)};
                if (n_threads_x == 256) {
                    using Block = ReduceAxesIwiseWidthBlock<Config, 256>;
                    stream.enqueue(
                        reduce_width_iwise<3, Block, Interface, OpDecay, Index, ReducedDecay, Output2D>,
                        config, op, shape_hw, reduced, output_2d, grid_offset
                    );
                } else {
                    using Block = ReduceAxesIwiseWidthBlock<Config, 64>;
                    stream.enqueue(
                        reduce_width_iwise<3, Block, Interface, OpDecay, Index, ReducedDecay, Output2D>,
                        config, op, shape_hw, reduced, output_2d, grid_offset
                    );
                }
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = ni::squeeze_left(axes_to_reduce.as<i32>() + 1);
            order = order.filter(0, 2, 1); // move the width back to rightmost

            // Reorder to (X, axis_to_reduce, width).
            auto reordered_shape = input_shape.reorder(order);
            auto reordered_output = output.map([&order](auto accessor) {
                accessor.reorder(order);
                return accessor;
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto TO_2D = nd::AccessorConfig<2>{.filter = {0, 2}};
            auto reordered_output_2d = nd::reconfig_accessors<TO_2D>(reordered_output);
            using ReorderedOutput2D = decltype(reordered_output_2d);

            // Block shape.
            constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
            constexpr u32 N_THREADS_Y = max(Config::block_size, N_THREADS_X) / N_THREADS_X;
            const auto n_threads = dim3(N_THREADS_X, N_THREADS_Y);

            // Grid shape.
            const auto grid_x = GridX(reordered_shape[2], N_THREADS_X);
            const auto grid_y = GridY(reordered_shape[0], 1);
            check(grid_x.n_launches() == 1);

            using Block = ReduceAxesIwiseHeightBlock<Config, N_THREADS_X>;
            const auto shape_2d = reordered_shape.template pop_front<1>();

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), 1),
                    .n_threads = n_threads,
                };
                const auto grid_offset = Vec{grid_y.offset(y)};
                if (axes_to_reduce[1]) {
                    stream.enqueue(
                        reduce_height_iwise<3, 1, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput2D>,
                        config, op, shape_2d, reduced, reordered_output_2d, grid_offset
                    );
                } else {
                    stream.enqueue(
                        reduce_height_iwise<3, 0, Block, Interface, OpDecay, Index, ReducedDecay, ReorderedOutput2D>,
                        config, op, shape_2d, reduced, reordered_output_2d, grid_offset
                    );
                }
            }
        }
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_2d(
        const Shape2<Index>& input_shape,
        Vec2<bool> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;

        if (axes_to_reduce[0]) {
            const auto input_shape_u32 = input_shape.template as<u32>();
            constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
            constexpr u32 N_THREADS_Y = max(Config::block_size, N_THREADS_X) / N_THREADS_X;
            const u32 n_blocks_x = divide_up(input_shape_u32[1], N_THREADS_X);
            const auto config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, 1, 1),
                .n_threads = dim3(N_THREADS_X, N_THREADS_Y),
            };

            constexpr auto TO_1D = nd::AccessorConfig<1>{.filter = {1}};
            auto output_1d = nd::reconfig_accessors<TO_1D>(output);
            using Output1D = decltype(output_1d);

            using Block = ReduceAxesIwiseHeightBlock<Config, N_THREADS_X>;
            using Interface = Config::interface;
            stream.enqueue(
                reduce_height_iwise<2, 0, Block, Interface, OpDecay, Index, ReducedDecay, Output1D>,
                config, std::forward<Op>(op), input_shape,
                std::forward<Reduced>(reduced), output_1d, Vec<u32, 0>{}
            );
        } else {
            panic("unreachable");
        }
    }
}

namespace noa::cuda {
    template<bool ZipReduced = false,
             bool ZipOutput = false,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096>
    struct ReduceAxesIwiseConfig {
        static_assert(is_multiple_of(BlockSize, Constant::WARP_SIZE) and BlockSize <= Limits::MAX_THREADS);
        using interface = nd::ReduceIwiseInterface<ZipReduced, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 max_grid_size = MaxGridSize;
    };

    template<typename Config = ReduceAxesIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_pure_nd_or_empty<Output, N>)
    NOA_NOINLINE void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        const auto axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input:shape={}, output:shape={}", input_shape, output_shape);
        } else if (all(axes_to_reduce == false)) {
            panic("No reduction to compute. Got shape input={}, output={}. Please use iwise instead.",
                  input_shape, output_shape);
        }

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce)) { // reduce to one value
            constexpr auto TO_1D = nd::AccessorConfig<1>{.enforce_contiguous = true, .filter = {0}};
            const auto output_1d = nd::reconfig_accessors<TO_1D>(output);
            return reduce_iwise(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, stream);
        }

        if constexpr (N > 1) {
            const auto shape_to_reduce = input_shape.pop_front();
            const auto shape_to_reduce_i64 = shape_to_reduce.template as_safe<i64>();
            const auto batch = safe_cast<u32>(input_shape[0]);

            using OpDecay = std::decay_t<Op>;
            using ReducedDecay = std::decay_t<Reduced>;

            if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per leftmost
                using Block = details::ReduceAxesIwiseBlock<Config, N == 2 ? 1 : 2>;
                using Interface = Config::interface;

                const auto n_elements = shape_to_reduce_i64.n_elements();
                constexpr auto SMALL_THRESHOLD = static_cast<i64>(Config::block_size * 32);
                auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter={0}}>(output);
                using Output1D = decltype(output_1d);

                if (n_elements <= SMALL_THRESHOLD) {
                    const auto n_threads = dim3(Block::block_size_x, Block::block_size_y);
                    const auto config = LaunchConfig{.n_blocks = batch, .n_threads = n_threads};

                    if constexpr (N == 4) {
                        stream.enqueue(
                            details::reduce_axes_iwise_4d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1D>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec
                        );
                    } else if constexpr (N == 3) {
                        stream.enqueue(
                            details::reduce_axes_iwise_3d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1D>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec
                        );
                    } else {
                        stream.enqueue(
                            details::reduce_axes_iwise_2d_small<Block, Interface, OpDecay, Index, ReducedDecay, Output1D>,
                            config, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, shape_to_reduce.vec
                        );
                    }
                } else {
                    auto [config, n_blocks_per_batch, n_blocks_hw] =
                        details::reduce_axes_iwise_nd_first_config<Block>(shape_to_reduce_i64);

                    // Allocate the 2d buffer.
                    using Joined = AccessorRestrictContiguous<ReducedDecay, 2, Index>;
                    auto buffer = AllocatorDevice::allocate_async<ReducedDecay>(n_blocks_per_batch * batch, stream);
                    auto joined = Joined(buffer.get(), Strides<Index, 2>{n_blocks_per_batch, 1});

                    auto grid_z = GridZ(batch, 1);
                    for (u32 z{}; z < grid_z.n_launches(); ++z) {
                        config.n_blocks.z = grid_z.n_blocks(z);
                        if constexpr (N == 4) {
                            stream.enqueue(
                                details::reduce_axes_iwise_4d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec, n_blocks_hw, grid_z.offset(z)
                            );
                        } else if constexpr (N == 3) {
                            stream.enqueue(
                                details::reduce_axes_iwise_3d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec, grid_z.offset(z)
                            );
                        } else {
                            stream.enqueue(
                                details::reduce_axes_iwise_2d_first<Block, Interface, OpDecay, Index, ReducedDecay, Joined>,
                                config, op, reduced, joined, shape_to_reduce.vec, grid_z.offset(z)
                            );
                        }
                    }
                    stream.enqueue(
                        details::reduce_axes_iwise_second<Config, Interface, OpDecay, Index, Joined, ReducedDecay, Output1D>,
                        LaunchConfig{.n_blocks = batch, .n_threads = Config::block_size},
                        std::forward<Op>(op), joined, n_blocks_per_batch, std::forward<Reduced>(reduced), output_1d
                    );
                }
                return;
            }
        }

        const i32 nb_axes_to_reduce = sum(axes_to_reduce.template as<i32>());
        check(nb_axes_to_reduce == 1,
              "Reducing more than one axis at a time is currently limited to a reduction that would "
              "result in one value per batch, i.e. the DHW dimensions should empty after reduction. "
              "Got input_shape={}, output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        if constexpr (N == 4) {
            details::launch_reduce_axes_iwise_4d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream
            );
        } else if constexpr (N == 3) {
            details::launch_reduce_axes_iwise_3d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream
            );
        } else if constexpr (N == 2) {
            details::launch_reduce_axes_iwise_2d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream
            );
        } else {
            panic("unreachable");
        }
    }
}
