#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/utils/Irange.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/ReduceIwise.cuh"

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
                static_assert(nt::always_false<Config>);
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
                static_assert(nt::always_false<Config>);
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
                    static_assert(nt::always_false<Index>);
                }
            } else if constexpr (N == 3) {
                if constexpr (R == 0) {
                    Config::interface::init(op, reduced, tidy, gid[1], gid[3]);
                } else if constexpr (R == 1) {
                    Config::interface::init(op, reduced, gid[1], tidy, gid[3]);
                } else {
                    static_assert(nt::always_false<Index>);
                }
            } else if constexpr (N == 2 and R == 0) {
                Config::interface::init(op, reduced, tidy, gid[3]);
            } else {
                static_assert(nt::always_false<Config>);
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
                static_assert(nt::always_false<Config>);
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

#ifdef NOA_IS_OFFLINE
namespace noa::cuda::guts {
    template<typename Config, size_t N> requires (N >= 2)
    auto reduce_axes_iwise_nd_first_config(const Shape<i32, N>& shape) {
        constexpr auto max_grid_size = static_cast<i32>(Config::max_grid_size);
        constexpr auto block_size = Vec3<i32>::from_values(1, Config::block_size_y, Config::block_size_x);
        static_assert(N > 2 or Config::block_size_y == 1);

        // Set the number of blocks, while keep the total number of blocks within the maximum allowed.
        auto n_blocks = Vec3<i32>::from_value(1);
        const auto shape_whd = shape.flip().pop_back();
        for (i32 i = 0; i <  static_cast<i32>(N - 1); ++i) {
            const auto n_blocks_allowed = max_grid_size / product(n_blocks);
            n_blocks[2 - i] = min(divide_up(shape_whd[i], block_size[2 - i]), n_blocks_allowed);
        }

        const auto n_blocks_u32 = n_blocks.as<u32>();
        const auto n_threads = dim3(block_size[2], block_size[1], 1);
        const auto n_blocks_dim3 = [&] {
            if constexpr (N == 4)
                return dim3(n_blocks_u32[2] * n_blocks_u32[1], n_blocks_u32[0], static_cast<u32>(shape[0]));
            else
                return dim3(n_blocks_u32[2], n_blocks_u32[1], static_cast<u32>(shape[0]));
        }();

        auto launch_config = LaunchConfig{.n_blocks=n_blocks_dim3, .n_threads=n_threads};
        auto n_blocks_yx = Vec{n_blocks_u32[1], n_blocks_u32[2]};
        return make_tuple(launch_config, product(n_blocks_u32), n_blocks_yx);
    }

    template<typename Config, typename Op, typename Reduced, typename Output, typename Index>
    void launch_reduce_axes_iwise_4d(
        const Shape4<Index>& input_shape,
        Vec4<bool> axes_to_reduce,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;

        if (axes_to_reduce[3]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(2, 3);
            u32 n_threads_x = shape_u32[3] > 512 ? 256u : 64u;
            if (not is_multiple_of(Config::block_size, n_threads_x))
                n_threads_x = Constant::WARP_SIZE;
            const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(shape_u32[2], n_threads_y);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, shape_u32[1], shape_u32[0]),
                .n_threads = dim3(n_threads_x, n_threads_y),
            };

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 2}};
            auto output_3d = ng::reconfig_accessors<to_3d>(output);
            using output_3d_t = decltype(output_3d);

            if (n_threads_x == 256) {
                using kernel_config = ReduceAxesIwiseWidthConfig<Config, 256>;
                stream.enqueue(
                    reduce_width_iwise<4, kernel_config, op_t, Index, reduced_t, output_3d_t>,
                    launch_config, std::forward<Op>(op), shape_hw, std::forward<Reduced>(reduced), output_3d
                );
            } else {
                using kernel_config = ReduceAxesIwiseWidthConfig<Config, 64>;
                stream.enqueue(
                    reduce_width_iwise<4, kernel_config, op_t, Index, reduced_t, output_3d_t>,
                    launch_config, std::forward<Op>(op), shape_hw, std::forward<Reduced>(reduced), output_3d
                );
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = ni::squeeze_left(axes_to_reduce.template as<i32>() + 1);
            order = order.filter(0, 1, 3, 2); // move the width back to rightmost

            // Reorder to (X, X, axis_to_reduce, width).
            auto reordered_shape = input_shape.reorder(order);
            auto reordered_output = output.map([&order](auto accessor) {
                accessor.reorder(order);
                return accessor;
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto to_3d = ng::AccessorConfig<3>{.filter = {0, 1, 3}};
            auto reordered_output_3d = ng::reconfig_accessors<to_3d>(reordered_output);
            using reordered_output_3d_t = decltype(reordered_output_3d);

            // Launch config.
            const auto reordered_shape_u32 = reordered_shape.template as<u32>();
            constexpr u32 n_threads_x = Constant::WARP_SIZE;
            constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(reordered_shape_u32[3], n_threads_x);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, reordered_shape_u32[1], reordered_shape_u32[0]),
                .n_threads = dim3(n_threads_x, n_threads_y),
            };

            using kernel_config = ReduceAxesIwiseHeightConfig<Config, n_threads_x>;
            auto shape_2d = reordered_shape.template pop_front<2>();
            if (axes_to_reduce[2]) {
                stream.enqueue(
                    reduce_height_iwise<4, 2, kernel_config, op_t, Index, reduced_t, reordered_output_3d_t>,
                    launch_config, std::forward<Op>(op), shape_2d,
                    std::forward<Reduced>(reduced), reordered_output_3d);
            } else if (axes_to_reduce[1]) {
                stream.enqueue(
                    reduce_height_iwise<4, 1, kernel_config, op_t, Index, reduced_t, reordered_output_3d_t>,
                    launch_config, std::forward<Op>(op), shape_2d,
                    std::forward<Reduced>(reduced), reordered_output_3d);
            } else {
                stream.enqueue(
                    reduce_height_iwise<4, 0, kernel_config, op_t, Index, reduced_t, reordered_output_3d_t>,
                    launch_config, std::forward<Op>(op), shape_2d,
                    std::forward<Reduced>(reduced), reordered_output_3d);
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
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;

        if (axes_to_reduce[2]) {
            const auto shape_u32 = input_shape.template as_safe<u32>();
            const auto shape_hw = input_shape.filter(1, 2);
            u32 n_threads_x = shape_u32[2] > 512 ? 256u : 64u;
            if (not is_multiple_of(Config::block_size, n_threads_x))
                n_threads_x = Constant::WARP_SIZE;
            const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(shape_u32[1], n_threads_y);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, shape_u32[0]),
                .n_threads = dim3(n_threads_x, n_threads_y),
            };

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_2d = ng::AccessorConfig<2>{.filter={0, 1}};
            auto output_2d = ng::reconfig_accessors<to_2d>(output);
            using output_2d_t = decltype(output_2d);

            if (n_threads_x == 256) {
                using kernel_config = ReduceAxesIwiseWidthConfig<Config, 256>;
                stream.enqueue(
                    reduce_width_iwise<3, kernel_config, op_t, Index, reduced_t, output_2d_t>,
                    launch_config, std::forward<Op>(op), shape_hw, std::forward<Reduced>(reduced), output_2d
                );
            } else {
                using kernel_config = ReduceAxesIwiseWidthConfig<Config, 64>;
                stream.enqueue(
                    reduce_width_iwise<3, kernel_config, op_t, Index, reduced_t, output_2d_t>,
                    launch_config, std::forward<Op>(op), shape_hw, std::forward<Reduced>(reduced), output_2d
                );
            }
        } else {
            // The kernel needs the axis to reduce at the "height" position.
            // The width should still be at the rightmost dimension.
            auto order = ni::squeeze_left(axes_to_reduce.template as<i32>() + 1);
            order = order.filter(0, 2, 1); // move the width back to rightmost

            // Reorder to (X, axis_to_reduce, width).
            auto reordered_shape = input_shape.reorder(order);
            auto reordered_output = output.map([&order](auto accessor) {
                accessor.reorder(order);
                return accessor;
            });

            // Remove the empty/reduced axis from the output.
            constexpr auto to_2d = ng::AccessorConfig<2>{.filter = {0, 2}};
            auto reordered_output_2d = ng::reconfig_accessors<to_2d>(reordered_output);
            using reordered_output_2d_t = decltype(reordered_output_2d);

            // Launch config.
            const auto reordered_shape_u32 = reordered_shape.template as<u32>();
            constexpr u32 n_threads_x = Constant::WARP_SIZE;
            constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(reordered_shape_u32[2], n_threads_x);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, reordered_shape_u32[0], 1),
                .n_threads = dim3(n_threads_x, n_threads_y),
            };

            using kernel_config = ReduceAxesIwiseHeightConfig<Config, n_threads_x>;
            auto shape_2d = reordered_shape.template pop_front<1>();
            if (axes_to_reduce[1]) {
                stream.enqueue(
                    reduce_height_iwise<3, 1, kernel_config, op_t, Index, reduced_t, reordered_output_2d_t>,
                    launch_config, std::forward<Op>(op), shape_2d,
                    std::forward<Reduced>(reduced), reordered_output_2d);
            } else {
                stream.enqueue(
                    reduce_height_iwise<3, 0, kernel_config, op_t, Index, reduced_t, reordered_output_2d_t>,
                    launch_config, std::forward<Op>(op), shape_2d,
                    std::forward<Reduced>(reduced), reordered_output_2d);
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
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;

        if (axes_to_reduce[0]) {
            const auto input_shape_u32 = input_shape.template as<u32>();
            constexpr u32 n_threads_x = Constant::WARP_SIZE;
            constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(input_shape_u32[1], n_threads_x);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x, 1, 1),
                .n_threads = dim3(n_threads_x, n_threads_y),
            };

            constexpr auto to_1d = ng::AccessorConfig<1>{.filter = {1}};
            auto output_1d = ng::reconfig_accessors<to_1d>(output);
            using output_1d_t = decltype(output_1d);

            using kernel_config = ReduceAxesIwiseHeightConfig<Config, n_threads_x>;
            stream.enqueue(
                reduce_height_iwise<2, 0, kernel_config, op_t, Index, reduced_t, output_1d_t>,
                launch_config, std::forward<Op>(op), input_shape,
                std::forward<Reduced>(reduced), output_1d
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
        using interface = ng::ReduceIwiseInterface<ZipReduced, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 max_grid_size = MaxGridSize;
    };

    template<typename Config = ReduceAxesIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value<std::decay_t<Reduced>> and
              nt::tuple_of_accessor_pure_nd_or_empty<Output, N>)
    constexpr void reduce_axes_iwise(
        const Shape<Index, N>& input_shape,
        const Shape<Index, N>& output_shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;

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
            constexpr auto to_1d = ng::AccessorConfig<1>{.enforce_contiguous = true, .filter = {0}};
            const auto output_1d = ng::reconfig_accessors<to_1d>(output);
            return reduce_iwise(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, stream);
        }

        if constexpr (N > 1) {
            const auto shape_to_reduce = input_shape.pop_front();
            const auto n_batch = static_cast<u32>(input_shape[0]);

            if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per leftmost
                const auto n_elements = shape_to_reduce.template as<i64>().n_elements();
                constexpr auto SMALL_THRESHOLD = static_cast<i64>(Config::block_size * 32);
                using config_t = guts::ReduceAxesIwiseBlockConfig<Config, N == 2 ? 1 : 2>;
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter={0}}>(output);
                using output_1d_t = decltype(output_1d);

                if (n_elements <= SMALL_THRESHOLD) {
                    const auto n_threads = dim3(config_t::block_size_x, config_t::block_size_y);
                    const auto launch_config = LaunchConfig{.n_blocks = n_batch, .n_threads = n_threads};

                    if constexpr (N == 4) {
                        stream.enqueue(
                            guts::reduce_axes_iwise_4d_small<config_t, op_t, Index, reduced_t, output_1d_t>,
                            launch_config, std::forward<Op>(op), std::forward<Reduced>(reduced),
                            output_1d, shape_to_reduce.vec);
                    } else if constexpr (N == 3) {
                        stream.enqueue(
                            guts::reduce_axes_iwise_3d_small<config_t, op_t, Index, reduced_t, output_1d_t>,
                            launch_config, std::forward<Op>(op), std::forward<Reduced>(reduced),
                            output_1d, shape_to_reduce.vec);
                    } else {
                        stream.enqueue(
                            guts::reduce_axes_iwise_2d_small<config_t, op_t, Index, reduced_t, output_1d_t>,
                            launch_config, std::forward<Op>(op), std::forward<Reduced>(reduced),
                            output_1d, shape_to_reduce.vec);
                    }
                } else {
                    const auto shape_i32 = input_shape.template as_safe<i32>();
                    auto [launch_config, n_blocks_per_batch, n_blocks_hw] =
                        guts::reduce_axes_iwise_nd_first_config<config_t>(shape_i32);

                    // Allocate the 2d buffer.
                    using joined_t = AccessorRestrictContiguous<reduced_t, 2, Index>;
                    auto buffer = AllocatorDevice<reduced_t>::allocate_async(n_blocks_per_batch * n_batch, stream);
                    auto joined = joined_t(buffer.get(), Strides2<Index>{n_blocks_per_batch, 1});

                    if constexpr (N == 4) {
                        stream.enqueue(
                            guts::reduce_axes_iwise_4d_first<config_t, op_t, Index, reduced_t, joined_t>,
                            launch_config, op, reduced, joined, shape_to_reduce.vec, n_blocks_hw);
                    } else if constexpr (N == 3) {
                        stream.enqueue(
                            guts::reduce_axes_iwise_3d_first<config_t, op_t, Index, reduced_t, joined_t>,
                            launch_config, op, reduced, joined, shape_to_reduce.vec);
                    } else {
                        stream.enqueue(
                            guts::reduce_axes_iwise_2d_first<config_t, op_t, Index, reduced_t, joined_t>,
                            launch_config, op, reduced, joined, shape_to_reduce.vec);
                    }
                    stream.enqueue(
                        guts::reduce_axes_iwise_second<Config, op_t, Index, joined_t, reduced_t, output_1d_t>,
                        LaunchConfig{.n_blocks = n_batch, .n_threads = Config::block_size},
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
            guts::launch_reduce_axes_iwise_4d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream);
        } else if constexpr (N == 3) {
            guts::launch_reduce_axes_iwise_3d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream);
        } else if constexpr (N == 2) {
            guts::launch_reduce_axes_iwise_2d<Config>(
                input_shape, axes_to_reduce, std::forward<Op>(op),
                std::forward<Reduced>(reduced), output, stream);
        } else {
            panic("unreachable");
        }
    }
}
#endif
