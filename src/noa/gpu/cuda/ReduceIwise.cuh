#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

namespace noa::cuda::guts {
    template<typename Config>
    struct ReduceIwise2dBlockConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = max(Constant::WARP_SIZE, Config::block_size);
        static constexpr u32 block_size_x = Constant::WARP_SIZE;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Config>
    struct ReduceIwise1dBlockConfig {
        using interface = Config::interface;
        static constexpr u32 max_grid_size = Config::max_grid_size;
        static constexpr u32 block_size = Config::block_size;
    };
}

namespace noa::cuda::guts {
    template<typename Config, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_4d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec4<Index> shape,
        Vec2<u32> n_blocks_hw
    ) {
        // Get the position within the 4d span.
        const Vec2<u32> index = ni::offset2index(blockIdx.x, n_blocks_hw[1]);
        const auto gid = Vec4<Index>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + threadIdx.y,
            Config::block_size_x * index[1] + threadIdx.x
        );

        // Traverse the entire 4d span.
        for (Index cb = gid[0]; cb < shape[0]; cb += gridDim.z)
            for (Index cd = gid[1]; cd < shape[1]; cd += gridDim.y)
                for (Index ch = gid[2]; ch < shape[2]; ch += Config::block_size_y * n_blocks_hw[0])
                    for (Index cw = gid[3]; cw < shape[3]; cw += Config::block_size_x * n_blocks_hw[1])
                        Config::interface::init(op, reduced, cb, cd, ch, cw);

        // Reduce to one value per block.
        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_3d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec3<Index> shape
    ) {
        const auto gid = Vec3<Index>::from_values(
            blockIdx.z,
            Config::block_size_y * blockIdx.y + threadIdx.y,
            Config::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index cd = gid[0]; cd < shape[0]; cd += gridDim.z)
            for (Index ch = gid[1]; ch < shape[1]; ch += Config::block_size_y * gridDim.y)
                for (Index cw = gid[2]; cw < shape[2]; cw += Config::block_size_x * gridDim.x)
                    Config::interface::init(op, reduced, cd, ch, cw);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_2d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec2<Index> shape
    ) {
        const auto gid = Vec2<Index>::from_values(
            Config::block_size_y * blockIdx.y + threadIdx.y,
            Config::block_size_x * blockIdx.x + threadIdx.x
        );

        for (Index ch = gid[0]; ch < shape[0]; ch += Config::block_size_y * gridDim.y)
            for (Index cw = gid[1]; cw < shape[1]; cw += Config::block_size_x * gridDim.x)
                Config::interface::init(op, reduced, ch, cw);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        const Index bid = blockIdx.y * gridDim.x + blockIdx.x;
        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, bid);
    }

    template<typename Config, typename Op, typename Index, typename Reduced>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_1d_first(
        Op op,
        Reduced reduced,
        Reduced* joined,
        Vec1<Index> shape
    ) {
        const Index tid = threadIdx.x;
        const Index bid = blockIdx.x;
        const auto gid = Vec1<Index>::from_values(Config::block_size * bid + tid);

        for (Index cw = gid[0]; cw < shape[0]; cw += Config::block_size * gridDim.x)
            Config::interface::init(op, reduced, cw);

        block_reduce_join<Config::interface, Config::block_size>(op, reduced, joined, tid, bid);
    }

    // One 1d block to finish joining the reduced values and compute the final output.
    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_second(
        Op op,
        Reduced* to_join,
        Index n_to_join,
        Reduced reduced,
        Output output
    ) {
        const Index tid = threadIdx.x;
        for (Index cid = tid; cid < n_to_join; cid += Config::block_size)
            Config::interface::join(op, to_join[cid], reduced);
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, 0);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_4d_small(Op op, Reduced reduced, Output output, Vec4<Index> shape) {
        const auto gid = Vec4<Index>::from_values(0, 0, threadIdx.y, threadIdx.x);

        for (Index i = gid[0]; i < shape[0]; ++i)
            for (Index j = gid[1]; j < shape[1]; ++j)
                for (Index k = gid[2]; k < shape[2]; k += Config::block_size_y)
                    for (Index l = gid[3]; l < shape[3]; l += Config::block_size_x)
                        Config::interface::init(op, reduced, i, j, k, l);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, 0);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_3d_small(Op op, Reduced reduced, Output output, Vec3<Index> shape) {
        const auto gid = Vec3<Index>::from_values(0, threadIdx.y, threadIdx.x);

        for (Index i = gid[0]; i < shape[0]; ++i)
            for (Index j = gid[1]; j < shape[1]; j += Config::block_size_y)
                for (Index k = gid[2]; k < shape[2]; k += Config::block_size_x)
                    Config::interface::init(op, reduced, i, j, k);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, 0);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_2d_small(Op op, Reduced reduced, Output output, Vec2<Index> shape) {
        const auto gid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);

        for (Index i = gid[0]; i < shape[0]; i += Config::block_size_y)
            for (Index j = gid[1]; j < shape[1]; j += Config::block_size_x)
                Config::interface::init(op, reduced, i, j);

        const Index tid = threadIdx.y * Config::block_size_x + threadIdx.x;
        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, tid, 0);
    }

    template<typename Config, typename Op, typename Index, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_iwise_1d_small(Op op, Reduced reduced, Output output, Vec1<Index> shape) {
        const auto gid = Vec1<Index>::from_values(threadIdx.x);

        for (Index i = gid[0]; i < shape[0]; i += Config::block_size)
            Config::interface::init(op, reduced, i);

        block_reduce_join_and_final<Config::interface, Config::block_size>(op, reduced, output, gid[0], 0);
    }
}

#ifdef NOA_IS_OFFLINE
namespace noa::cuda::guts {
    template<typename Config, size_t N>
    auto reduce_iwise_nd_first_config(const Shape<i32, N>& shape) {
        constexpr auto max_grid_size = static_cast<i32>(Config::max_grid_size);
        constexpr auto block_size = [] {
            if constexpr (N > 1) // 2d blocks
                return Vec4<i32>::from_values(1, 1, Config::block_size_y, Config::block_size_x);
            else // 1d blocks
                return Vec4<i32>::from_values(1, 1, 1, Config::block_size);
        }();

        // Set the number of blocks, while keep the total number of blocks within the maximum allowed.
        auto n_blocks = Vec4<i32>::from_value(1);
        const auto shape_whdb = shape.flip();
        for (i32 i = 0; i <  static_cast<i32>(N); ++i) {
            const auto n_blocks_allowed = max_grid_size / product(n_blocks);
            n_blocks[3 - i] = min(divide_up(shape_whdb[i], block_size[3 - i]), n_blocks_allowed);
        }

        const auto n_blocks_u32 = n_blocks.as<u32>();
        const auto n_threads = dim3(block_size[3], block_size[2], 1);
        const auto n_blocks_dim3 = [&] {
            if constexpr (N == 4)
                return dim3(n_blocks_u32[3] * n_blocks_u32[2], n_blocks_u32[1], n_blocks_u32[0]);
            else
                return dim3(n_blocks_u32[3], n_blocks_u32[2], n_blocks_u32[1]);
        }();

        auto launch_config = LaunchConfig{.n_blocks=n_blocks_dim3, .n_threads=n_threads};
        auto n_blocks_yx = Vec{n_blocks_u32[2], n_blocks_u32[3]};
        return make_tuple(launch_config, product(n_blocks_u32), n_blocks_yx);
    }
}

namespace noa::cuda {
    template<bool ZipReduced = false,
             bool ZipOutput = false,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096>
    struct ReduceIwiseConfig {
        static_assert(is_multiple_of(BlockSize, Constant::WARP_SIZE) and BlockSize <= Limits::MAX_THREADS);
        using interface = ng::ReduceIwiseInterface<ZipReduced, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 max_grid_size = MaxGridSize;
    };

    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::tuple_of_accessor_value<Reduced> and
              nt::tuple_of_accessor_pure<Output> and
              nt::tuple_of_accessor_nd<Output, 1>)
    constexpr void reduce_iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Reduced&& reduced,
        Output& output,
        Stream& stream
    ) {
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;
        using output_t = std::decay_t<Output>;

        const auto n_elements = shape.template as<i64>().elements();
        constexpr auto SMALL_THRESHOLD = static_cast<i64>(Config::block_size * 32);

        if (n_elements <= SMALL_THRESHOLD) {
            if constexpr (N == 1) {
                using config_t = guts::ReduceIwise1dBlockConfig<Config>;
                stream.enqueue(
                    guts::reduce_iwise_1d_small<config_t, op_t, Index, reduced_t, output_t>,
                    LaunchConfig{.n_blocks = 1, .n_threads = dim3(config_t::block_size)},
                    std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 2) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                    guts::reduce_iwise_2d_small<config_t, op_t, Index, reduced_t, output_t>,
                    LaunchConfig{.n_blocks = 1, .n_threads = dim3(config_t::block_size_x, config_t::block_size_y)},
                    std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 3) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                    guts::reduce_iwise_3d_small<config_t, op_t, Index, reduced_t, output_t>,
                    LaunchConfig{.n_blocks = 1, .n_threads = dim3(config_t::block_size_x, config_t::block_size_y)},
                    std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 4) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                    guts::reduce_iwise_4d_small<config_t, op_t, Index, reduced_t, output_t>,
                    LaunchConfig{.n_blocks = 1, .n_threads = dim3(config_t::block_size_x, config_t::block_size_y)},
                    std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);
            } else {
                static_assert(nt::always_false<Op>);
            }
        } else {
            const auto shape_i32 = shape.template as_safe<i32>();
            using buffer_t = AllocatorDevice<reduced_t>::unique_type;
            buffer_t joined;
            Index n_joined;
            auto allocate_joined = [&](u32 n) {
                n_joined = static_cast<Index>(n);
                joined = AllocatorDevice<reduced_t>::allocate_async(static_cast<i64>(n_joined), stream);
            };

            // First kernel.
            if constexpr (N == 1) {
                using config_t = guts::ReduceIwise1dBlockConfig<Config>;
                auto [launch_config, n_blocks, _] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_1d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               op, reduced, joined.get(), shape.vec);

            } else if constexpr (N == 2) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, _] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_2d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               op, reduced, joined.get(), shape.vec);

            } else if constexpr (N == 3) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, _] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_3d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               op, reduced, joined.get(), shape.vec);

            } else if constexpr (N == 4) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, n_blocks_hw] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_4d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               op, reduced, joined.get(), shape.vec, n_blocks_hw);

            } else {
                static_assert(nt::always_false<Op>);
            }

            // Second kernel.
            stream.enqueue(
                guts::reduce_iwise_second<Config, op_t, Index, reduced_t, output_t>,
                LaunchConfig{.n_blocks = 1, .n_threads = Config::block_size},
                std::forward<Op>(op), joined.get(), n_joined, std::forward<Reduced>(reduced), output
            );
        }
    }
}
#endif
