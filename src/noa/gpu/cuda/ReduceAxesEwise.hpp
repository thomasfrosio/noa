#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/utils/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/ReduceAxesEwise.cuh"
#include "noa/gpu/cuda/ReduceEwise.hpp"

namespace noa::cuda::guts {
    template<typename Index>
    constexpr auto get_reduced_axes(
            const Shape4<Index>& input_shape,
            const Shape4<Index>& output_shape
    ) -> Vec4<bool> {
        const auto axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input={}, output={}", input_shape, output_shape);
        }
        return axes_to_reduce;
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_width(
            const Shape4<Index>& shape,
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Stream& stream
    ) {
        const auto shape_u32 = shape.template as_safe<u32>();
        const auto shape_hw = shape.filter(2, 3);
        const u32 n_threads_x = shape_u32[3] > 512 ? 256u : 64u;
        const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const u32 n_blocks_x = divide_up(shape_u32[2], n_threads_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x, shape_u32[1], shape_u32[0]),
                .n_threads=dim3(n_threads_x, n_threads_y),
        };

        // Load-vectorize every row.
        u32 input_vector_size{1};
        if constexpr (ng::are_accessors_const<Input>() and std::decay_t<Input>::SIZE <= 4) {
            input_vector_size = maximum_vector_size(
                    input, Config::n_elements_per_thread, n_threads_x, shape_u32.pop_back());
        }

        // The width of the output is empty/reduced, remove it.
        constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 2}};
        auto output_3d = ng::reconfig_accessors<to_3d>(std::forward<Output>(output));

        using op_t = std::decay_t<Op>;
        using input_t = std::decay_t<Input>;
        using reduced_t = std::decay_t<Reduced>;
        using output_3d_t = decltype(output_3d);

        if (input_vector_size > 1) {
            constexpr auto to_contiguous = ng::AccessorConfig<0>{.enforce_contiguous=true};
            auto input_contiguous = ng::reconfig_accessors<to_contiguous>(std::forward<Input>(input));
            using input_contig_t = decltype(input_contiguous);

            if (n_threads_x == 256) {
                if (input_vector_size == 2) {
                    using kernel_config = ReduceAxesEwiseWidthConfig<Config, 256, 2>;
                    stream.enqueue(
                            reduce_width_ewise<kernel_config, op_t, Index, input_contig_t, reduced_t, output_3d_t>,
                            launch_config, std::forward<Op>(op), std::move(input_contiguous), shape_hw, reduced, output_3d
                    );
                } else {
                    using kernel_config = ReduceAxesEwiseWidthConfig<Config, 256, 4>;
                    stream.enqueue(
                            reduce_width_ewise<kernel_config, op_t, Index, input_contig_t, reduced_t, output_3d_t>,
                            launch_config, std::forward<Op>(op), std::move(input_contiguous), shape_hw, reduced, output_3d
                    );
                }
            } else {
                if (input_vector_size == 2) {
                    using kernel_config = ReduceAxesEwiseWidthConfig<Config, 64, 2>;
                    stream.enqueue(
                            reduce_width_ewise<kernel_config, op_t, Index, input_contig_t, reduced_t, output_3d_t>,
                            launch_config, std::forward<Op>(op), std::move(input_contiguous), shape_hw, reduced, output_3d
                    );
                } else {
                    using kernel_config = ReduceAxesEwiseWidthConfig<Config, 64, 4>;
                    stream.enqueue(
                            reduce_width_ewise<kernel_config, op_t, Index, input_contig_t, reduced_t, output_3d_t>,
                            launch_config, std::forward<Op>(op), std::move(input_contiguous), shape_hw, reduced, output_3d
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                using kernel_config = ReduceAxesEwiseWidthConfig<Config, 256, 1>;
                stream.enqueue(
                        reduce_width_ewise<kernel_config, op_t, Index, input_t, reduced_t, output_3d_t>,
                        launch_config, std::forward<Op>(op), std::forward<Input>(input), shape_hw, reduced, output_3d
                );
            } else {
                using kernel_config = ReduceAxesEwiseWidthConfig<Config, 64, 1>;
                stream.enqueue(
                        guts::reduce_width_ewise<kernel_config, op_t, Index, input_t, reduced_t, output_3d_t>,
                        launch_config, std::forward<Op>(op), std::forward<Input>(input), shape_hw, reduced, output_3d
                );
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_bdh(
            const Shape4<Index>& shape,
            const Vec4<bool>& axes_to_reduce,
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Stream& stream
    ) {
        // First copy|move the input and output since they'll need to be reordered.
        auto input_ = std::forward<Input>(input);
        auto output_ = std::forward<Output>(output);

        // The kernel needs the axis to reduce at the "height" position.
        // The width should still be at the rightmost dimension.
        auto order = ni::squeeze_left(axes_to_reduce.template as<i32>() + 1);
        order = order.filter(0, 1, 3, 2); // move the width back to rightmost

        // Reorder to (X, X, axis_to_reduce, width).
        auto reordered_shape = shape.reorder(order);
        input_.for_each([&order](auto& accessor) { accessor.reorder(order); });
        output_.for_each([&order](auto& accessor) { accessor.reorder(order); });

        // Remove the empty/reduced axis from the output.
        constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 3}};
        auto output_3d = ng::reconfig_accessors<to_3d>(std::move(output_));

        // Launch config.
        const auto reordered_shape_u32 = reordered_shape.template as<u32>();
        constexpr u32 n_threads_x = Constant::WARP_SIZE;
        constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const u32 n_blocks_x = divide_up(reordered_shape_u32[3], n_threads_x);
        const auto launch_config = LaunchConfig{
            .n_blocks=dim3(n_blocks_x, reordered_shape_u32[1], reordered_shape_u32[0]),
            .n_threads=dim3(n_threads_x, n_threads_y),
        };

        using op_t = std::decay_t<Op>;
        using input_t = std::decay_t<Input>;
        using reduced_t = std::decay_t<Reduced>;
        using output_3d_t = decltype(output_3d);
        using kernel_config = ReduceAxesEwiseHeightConfig<Config, n_threads_x>;
        stream.enqueue(
                reduce_height_ewise<kernel_config, op_t, Index, input_t, reduced_t, output_3d_t>,
                launch_config, std::forward<Op>(op), std::move(input_), reordered_shape.template pop_front<2>(),
                std::forward<Reduced>(reduced), std::move(output_3d)
        );
    }
}

namespace noa::cuda {
    template<typename Config = ReduceEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    requires (nt::is_tuple_of_accessor_v<Input> and
              nt::is_tuple_of_accessor_pure_v<Output> and
              nt::are_tuple_of_accessor_ndim_v<4, Input, Output> and
              not nt::is_tuple_of_accessor_value_v<Input> and // at least one varray
              nt::is_tuple_of_accessor_value_v<Reduced>)
    constexpr void reduce_axes_ewise(
            const Shape4<Index>& input_shape,
            const Shape4<Index>& output_shape,
            Op&& op,
            Input&& input,
            Reduced&& reduced,
            Output&& output,
            Stream& stream
    ) {
        const Vec axes_to_reduce = guts::get_reduced_axes(input_shape, output_shape);

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value or one value per batch
            const auto n_batches = output_shape[0];
            const auto input_i64 = input_shape.template as<i64>();
            const bool reduce_all = axes_empty_or_to_reduce[0];
            const auto n_elements_to_reduce = safe_cast<Index>(
                    reduce_all ? input_i64.elements() : input_i64.pop_front().elements());

            Vec4<bool> is_contiguous = ni::is_contiguous(input, input_shape);
            if (not reduce_all)
                is_contiguous[0] = true;

            auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter={0}}>(std::forward<Output>(output));

            constexpr auto SMALL_THRESHOLD = Config::n_elements_per_thread * Config::block_size * 4;
            if (all(is_contiguous.pop_back())) {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    guts::launch_reduce_ewise_small_2d<Config>(
                            std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                            std::move(output_1d), Shape2<Index>{n_batches, n_elements_to_reduce}, stream);
                } else {
                    guts::launch_reduce_ewise_large_2d<Config>(
                            std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                            std::move(output_1d), Shape2<Index>{n_batches, n_elements_to_reduce}, stream);
                }
            } else {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    guts::launch_reduce_ewise_small_4d<Config>(
                            std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                            std::move(output_1d), input_shape, true, stream);
                } else {
                    guts::launch_reduce_ewise_large_4d<Config>(
                            std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                            std::move(output_1d), input_shape, true, stream);
                }
            }
            return;
        }

        const i32 nb_axes_to_reduce = sum(axes_to_reduce.template as<i32>());
        check(nb_axes_to_reduce == 1,
              "Reducing more than one axis at a time is currently limited to a reduction that would "
              "result in one value per batch, i.e. the DHW dimensions should empty after reduction. "
              "Got input_shape={}, output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        if (axes_to_reduce[3]) {
            guts::launch_reduce_ewise_width<Config>(
                    input_shape,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    stream);
        } else {
            guts::launch_reduce_ewise_bdh<Config>(
                    input_shape, axes_to_reduce,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    stream);
        }
    }
}
#endif
