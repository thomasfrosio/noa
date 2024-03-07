#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/utils/Irange.hpp"
#include "noa/gpu/cuda/kernels/ReduceAxesIwise.cuh"
#include "noa/gpu/cuda/AllocatorDevice.hpp"
#include "noa/gpu/cuda/ReduceIwise.cuh"

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
                    .n_blocks=dim3(n_blocks_x, shape_u32[1], shape_u32[0]),
                    .n_threads=dim3(n_threads_x, n_threads_y),
            };

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 2}};
            auto output_3d = ng::reconfig_accessors<to_3d>(output);
            using output_3d_t = decltype(output_3d);

            if (n_threads_x == 256) {
                using kernel_config = guts::ReduceAxesIwiseWidthConfig<Config, 256>;
                stream.enqueue(
                        guts::reduce_width_iwise<4, kernel_config, op_t, Index, reduced_t, output_3d_t>,
                        launch_config, std::forward<Op>(op), shape_hw, reduced, output_3d
                );
            } else {
                using kernel_config = guts::ReduceAxesIwiseWidthConfig<Config, 64>;
                stream.enqueue(
                        guts::reduce_width_iwise<4, kernel_config, op_t, Index, reduced_t, output_3d_t>,
                        launch_config, std::forward<Op>(op), shape_hw, reduced, output_3d
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
            constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 3}};
            auto reordered_output_3d = ng::reconfig_accessors<to_3d>(reordered_output);
            using reordered_output_3d_t = decltype(reordered_output_3d);

            // Launch config.
            const auto reordered_shape_u32 = reordered_shape.template as<u32>();
            constexpr u32 n_threads_x = Constant::WARP_SIZE;
            constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(reordered_shape_u32[3], n_threads_x);
            const auto launch_config = LaunchConfig{
                    .n_blocks=dim3(n_blocks_x, reordered_shape_u32[1], reordered_shape_u32[0]),
                    .n_threads=dim3(n_threads_x, n_threads_y),
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
                    .n_blocks=dim3(n_blocks_x, shape_u32[0]),
                    .n_threads=dim3(n_threads_x, n_threads_y),
            };

            // The width of the output is empty/reduced, remove it.
            constexpr auto to_2d = ng::AccessorConfig<2>{.filter={0, 1}};
            auto output_2d = ng::reconfig_accessors<to_2d>(output);
            using output_2d_t = decltype(output_2d);

            if (n_threads_x == 256) {
                using kernel_config = guts::ReduceAxesIwiseWidthConfig<Config, 256>;
                stream.enqueue(
                        guts::reduce_width_iwise<3, kernel_config, op_t, Index, reduced_t, output_2d_t>,
                        launch_config, std::forward<Op>(op), shape_hw, reduced, output_2d
                );
            } else {
                using kernel_config = guts::ReduceAxesIwiseWidthConfig<Config, 64>;
                stream.enqueue(
                        guts::reduce_width_iwise<3, kernel_config, op_t, Index, reduced_t, output_2d_t>,
                        launch_config, std::forward<Op>(op), shape_hw, reduced, output_2d
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
            constexpr auto to_2d = ng::AccessorConfig<2>{.filter={0, 2}};
            auto reordered_output_2d = ng::reconfig_accessors<to_2d>(reordered_output);
            using reordered_output_2d_t = decltype(reordered_output_2d);

            // Launch config.
            const auto reordered_shape_u32 = reordered_shape.template as<u32>();
            constexpr u32 n_threads_x = Constant::WARP_SIZE;
            constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
            const u32 n_blocks_x = divide_up(reordered_shape_u32[2], n_threads_x);
            const auto launch_config = LaunchConfig{
                    .n_blocks=dim3(n_blocks_x, reordered_shape_u32[0], 1),
                    .n_threads=dim3(n_threads_x, n_threads_y),
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
                    .n_blocks=dim3(n_blocks_x, 1, 1),
                    .n_threads=dim3(n_threads_x, n_threads_y),
            };

            constexpr auto to_1d = ng::AccessorConfig<1>{.filter={1}};
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
    requires (nt::is_tuple_of_accessor_value_v<Reduced> and
              nt::is_tuple_of_accessor_pure_v<Output> and
              nt::is_tuple_of_accessor_ndim_v<N, Output>)
    constexpr void reduce_axes_iwise(
            const Shape<Index, N>& input_shape,
            const Shape<Index, N>& output_shape,
            Op&& op,
            Reduced&& reduced,
            Output& output,
            Stream& stream
    )  {
        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;

        const auto axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input:shape={}, output:shape={}", input_shape, output_shape);
        }

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce)) { // reduce to one value
            constexpr auto to_1d = ng::AccessorConfig<1>{.enforce_contiguous=true, .filter={0}};
            const auto output_1d = ng::reconfig_accessors<to_1d>(output);
            return reduce_iwise(input_shape, std::forward<Op>(op), std::forward<Reduced>(reduced), output_1d, stream);
        } else if (all(axes_to_reduce == false)) {
            return; // nothing to reduce
        }

        if constexpr (N > 1) {
            const auto shape_to_reduce = input_shape.pop_front();
            const auto n_batch = static_cast<u32>(input_shape[0]);

            if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value per leftmost
                const auto n_elements = shape_to_reduce.template as<i64>().elements();
                constexpr auto SMALL_THRESHOLD = static_cast<i64>(Config::block_size * 32);
                using config_t = guts::ReduceAxesIwiseBlockConfig<Config, N == 2 ? 1 : 2>;
                auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter={0}}>(output);
                using output_1d_t = decltype(output_1d);

                if (n_elements <= SMALL_THRESHOLD) {
                    const auto n_threads = dim3(config_t::block_size_x, config_t::block_size_y);
                    const auto launch_config = LaunchConfig{.n_blocks=n_batch, .n_threads=n_threads};

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
                                launch_config, std::forward<Op>(op), reduced, joined, shape_to_reduce.vec, n_blocks_hw);
                    } else if constexpr (N == 3) {
                        stream.enqueue(
                                guts::reduce_axes_iwise_3d_first<config_t, op_t, Index, reduced_t, joined_t>,
                                launch_config, std::forward<Op>(op), reduced, joined, shape_to_reduce.vec);
                    } else {
                        stream.enqueue(
                                guts::reduce_axes_iwise_2d_first<config_t, op_t, Index, reduced_t, joined_t>,
                                launch_config, std::forward<Op>(op), reduced, joined, shape_to_reduce.vec);
                    }
                    stream.enqueue(
                            guts::reduce_axes_iwise_second<Config, op_t, Index, joined_t, reduced_t, output_1d_t>,
                            LaunchConfig{.n_blocks=n_batch, .n_threads=Config::block_size},
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
