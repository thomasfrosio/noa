#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Irange.hpp"
#include "noa/gpu/cuda/kernels/ReduceIwise.cuh"
#include "noa/gpu/cuda/AllocatorDevice.hpp"

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
    template<typename Config = ReduceIwiseConfig<>,
             typename Op, typename Reduced, typename Output, typename Index, size_t N>
    requires (nt::is_tuple_of_accessor_value_v<Reduced> and
              nt::is_tuple_of_accessor_pure_v<Output> and
              nt::is_tuple_of_accessor_ndim_v<1, Output>)
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
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 2) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, _] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_2d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 3) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, _] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_3d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec);

            } else if constexpr (N == 4) {
                using config_t = guts::ReduceIwise2dBlockConfig<Config>;
                auto [launch_config, n_blocks, n_blocks_hw] = guts::reduce_iwise_nd_first_config<config_t>(shape_i32);
                allocate_joined(n_blocks);
                stream.enqueue(guts::reduce_iwise_4d_first<config_t, op_t, Index, reduced_t>, launch_config,
                               std::forward<Op>(op), reduced, joined.get(), shape.vec, n_blocks_hw);

            } else {
                static_assert(nt::always_false_v<Op>);
            }

            // Second kernel.
            stream.enqueue(
                    guts::reduce_iwise_second<Config, op_t, Index, reduced_t, output_t>,
                    LaunchConfig{.n_blocks=1, .n_threads=Config::block_size},
                    std::forward<Op>(op), joined.get(), n_joined, std::forward<Reduced>(reduced), output
            );
        }
    }
}
#endif
