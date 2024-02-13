#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Irange.hpp"
#include "noa/gpu/cuda/kernels/ReduceIwise.cuh"
#include "noa/gpu/cuda/AllocatorDevice.hpp"

namespace noa::cuda::guts {
    template<typename Config>
    auto reduce_iwise_4d_first_config(const Shape4<u32>& shape) -> Tuple<LaunchConfig, u32, Vec2<u32>> {
        u32 n_blocks_left = Config::max_grid_size;
        Vec4<u32> n_blocks{};
        Vec4<u32> block_size{Config::block_size_x, Config::block_size_y, 1, 1};
        for (auto i: irange(4)) {
            n_blocks[i] = min(divide_up(shape[i], block_size[i]), n_blocks_left);
            n_blocks_left = max(1u, n_blocks_left - n_blocks[i]);
        }
        auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks[3] * n_blocks[2], n_blocks[1], n_blocks[0]),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
        };
        return {launch_config, product(n_blocks), {n_blocks[2], n_blocks[3]}};
    }

    template<typename Config>
    auto reduce_iwise_3d_first_config(const Shape3<u32>& shape) -> Tuple<LaunchConfig, u32> {
        u32 n_blocks_left = Config::max_grid_size;
        Vec3<u32> n_blocks{};
        Vec3<u32> block_size{Config::block_size_x, Config::block_size_y, 1};
        for (auto i: irange(3)) {
            n_blocks[i] = min(divide_up(shape[i], block_size[i]), n_blocks_left);
            n_blocks_left = max(1u, n_blocks_left - n_blocks[i]);
        }
        auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks[2], n_blocks[1], n_blocks[0]),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
        };
        return {launch_config, product(n_blocks)};
    }

    template<typename Config>
    auto reduce_iwise_2d_first_config(const Shape2<u32>& shape) -> Tuple<LaunchConfig, u32> {
        u32 n_blocks_left = Config::max_grid_size;
        Vec2<u32> n_blocks{};
        Vec2<u32> block_size{Config::block_size_x, Config::block_size_y};
        for (auto i: irange(2)) {
            n_blocks[i] = min(divide_up(shape[i], block_size[i]), n_blocks_left);
            n_blocks_left = max(1u, n_blocks_left - n_blocks[i]);
        }
        auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks[1], n_blocks[0]),
                .n_threads=dim3(Config::block_size_x, Config::block_size_y),
        };
        return {launch_config, product(n_blocks)};
    }

    template<typename Config>
    auto reduce_iwise_1d_first_config(const Shape1<u32>& shape) -> Tuple<LaunchConfig, u32> {
        u32 n_blocks_left = Config::max_grid_size;
        const u32 n_blocks = min(divide_up(shape[0], Config::block_size), n_blocks_left);
        auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks),
                .n_threads=dim3(Config::block_size_x),
        };
        return {launch_config, n_blocks};
    }
}

namespace noa::cuda {
    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::is_tuple_of_accessor_value_v<Reduced> and nt::is_tuple_of_accessor_pure_v<Output>)
    constexpr void reduce_iwise(
            const Shape<Index, N>& shape,
            Op&& op,
            Reduced&& reduced,
            Output& output,
            Stream& stream
    )  {
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
                        LaunchConfig{.n_blocks=1, .n_threads=dim3(config_t::block_size)},
                        std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 2) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                        guts::reduce_iwise_2d_small<config_t, op_t, Index, reduced_t, output_t>,
                        LaunchConfig{.n_blocks=1, .n_threads=dim3(config_t::block_size_x, config_t::block_size_y)},
                        std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 3) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                        guts::reduce_iwise_3d_small<config_t, op_t, Index, reduced_t, output_t>,
                        LaunchConfig{.n_blocks=1, .n_threads=dim3(config_t::block_size_x, config_t::block_size_y)},
                        std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);

            } else if constexpr (N == 4) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                stream.enqueue(
                        guts::reduce_iwise_4d_small<config_t, op_t, Index, reduced_t, output_t>,
                        LaunchConfig{.n_blocks=1, .n_threads=dim3(config_t::block_size_x, config_t::block_size_y)},
                        std::forward<Op>(op), std::forward<Reduced>(reduced), output, shape.vec);
            } else {
                static_assert(nt::always_false_v<Op>);
            }
        } else {
            const auto shape_u32 = shape.template as_safe<u32>();
            using buffer_t = AllocatorDevice<Reduced>::unique_type;
            buffer_t joined;
            Index n_joined;
            auto allocate_joined = [&](u32 n) {
                n_joined = n;
                joined = AllocatorDevice<Reduced>::allocate_async(n_joined, stream);
            };

            // First kernel.
            if constexpr (N == 1) {
                using config_t = guts::ReduceIwise1dBlockConfig<Config>;
                auto [launch_config, n_blocks] = guts::reduce_iwise_1d_first_config<config_t>(shape_u32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_1d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 2) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks] = guts::reduce_iwise_2d_first_config<config_t>(shape_u32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_2d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 3) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks] = guts::reduce_iwise_3d_first_config<config_t>(shape_u32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_3d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 4) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, n_blocks_hw] = guts::reduce_iwise_4d_first_config<config_t>(shape_u32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_4d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec, n_blocks_hw);

            } else {
                static_assert(nt::always_false_v<Op>);
            }

            // Second kernel.
            stream.enqueue(
                    guts::reduce_iwise_second<Config, reduced_t, output_t, Index, op_t>,
                    LaunchConfig{.n_blocks=1, .n_threads=Config::block_size},
                    std::forward<Op>(op), joined.get(), n_joined, std::forward<Reduced>(reduced), output
            );
        }
    }
}
#endif
