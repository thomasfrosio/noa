#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/gpu/cuda/kernels/Ewise.cuh"
#include "noa/gpu/cuda/Pointers.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda {
    template<typename Config = EwiseConfig<>,
             typename Input, typename Output, typename Index, typename Op>
    requires (nt::is_tuple_of_accessor_or_empty_v<Input> and
              (nt::is_empty_tuple_v<Output> or nt::is_tuple_of_accessor_pure_v<Output>))
    void ewise(
            const Shape4<Index>& shape,
            Op&& op,
            Input&& input,
            Output&& output,
            Stream& stream
    ) {
        const Vec4<bool> is_contiguous =
                ni::is_contiguous(input, shape) and
                ni::is_contiguous(output, shape);

        using op_t = std::decay_t<Op>;

        if (is_contiguous[1] and is_contiguous[2]) { // 2d-like
            // Keep batches separated in a different grid.y if they're not contiguous.
            const auto batch = is_contiguous[0] ? 1u : static_cast<u32>(shape[0]);
            const auto shape_i64 = shape.template as<i64>();
            const auto n_elements = safe_cast<Index>(
                    is_contiguous[0] ? shape_i64.elements() : shape_i64.pop_front().elements());

            // Only vectorize if the inputs are not modified.
            u32 vector_size{1};
            if constexpr (ng::are_accessors_const<Input>()) {
                vector_size = min(
                        maximum_vector_size(input, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1}),
                        maximum_vector_size(output, Config::n_elements_per_thread, Config::block_size, Shape3<Index>{shape[0], 1, 1})
                );
            }

            // 1d|2d grid of 1d blocks.
            const auto block_work_size = static_cast<Index>(
                    Config::block_size * max(vector_size, Config::n_elements_per_thread));
            const auto launch_config = LaunchConfig{
                .n_blocks=dim3(static_cast<u32>(divide_up(n_elements, block_work_size)), batch, 1u),
                .n_threads=dim3(Config::block_size, 1u, 1u),
            };

            if (Config::n_elements_per_thread == 1 or vector_size == 1) {
                constexpr auto accessor_config_2d = ng::AccessorConfig<2>{
                        .enforce_contiguous=false,
                        .enforce_restrict=false,
                        .filter={0, 3},
                };
                auto input_2d = ng::reconfig_accessors<accessor_config_2d>(std::forward<Input>(input));
                auto output_2d = ng::reconfig_accessors<accessor_config_2d>(std::forward<Output>(output));
                using input_t = decltype(input_2d);
                using output_t = decltype(output_2d);
                return stream.enqueue(
                        guts::ewise_2d<guts::EwiseConfig1dBlock<Config, 1>, op_t, input_t, output_t, Index>,
                        launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);

            } else {
                constexpr auto accessor_config_2d = ng::AccessorConfig<2>{
                        .enforce_contiguous=true,
                        .enforce_restrict=false,
                        .filter={0, 3},
                };
                auto input_2d = ng::reconfig_accessors<accessor_config_2d>(std::forward<Input>(input));
                auto output_2d = ng::reconfig_accessors<accessor_config_2d>(std::forward<Output>(output));
                using input_t = decltype(input_2d);
                using output_t = decltype(output_2d);

                if (vector_size == 2) {
                    return stream.enqueue(
                            guts::ewise_2d_vectorized<guts::EwiseConfig1dBlock<Config, 2>, op_t, input_t, output_t, Index>,
                            launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);
                } else if (vector_size == 4) {
                    return stream.enqueue(
                            guts::ewise_2d_vectorized<guts::EwiseConfig1dBlock<Config, 4>, op_t, input_t, output_t, Index>,
                            launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);
                } else {
                    return stream.enqueue(
                            guts::ewise_2d_vectorized<guts::EwiseConfig1dBlock<Config, 8>, op_t, input_t, output_t, Index>,
                            launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);
                }
            }
        } else {
            using kernel_config = guts::EwiseConfig2dBlock<Config>;
            using input_t = std::decay_t<Input>;
            using output_t = std::decay_t<Output>;

            const auto shape_u32 = shape.template as<u32>();
            const u32 n_blocks_x = divide_up(shape_u32[3], kernel_config::block_work_size_x);
            const u32 n_blocks_y = divide_up(shape_u32[2], kernel_config::block_work_size_y);
            const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, shape_u32[1], shape_u32[0]),
                .n_threads=dim3(kernel_config::block_size_x, kernel_config::block_size_y, 1u),
            };
            stream.enqueue(
                    guts::ewise_4d<kernel_config, op_t, input_t, output_t, Index>,
                    launch_config,
                    std::forward<Op>(op),
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    shape.filter(2, 3), n_blocks_x);
        }
    }
}
#endif
