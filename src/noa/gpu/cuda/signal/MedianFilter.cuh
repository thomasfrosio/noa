#pragma once

#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"
#include "noa/gpu/cuda/kernels/MedianFilter.cuh"

// The current implementations only supports small squared windows. This allows to:
//  1)  Load the windows for all threads in a block in shared memory. This is useful because windows overlap.
//  2)  The exchange search can be done on the per thread registers. Only about half of the window needs to
//      be on the registers at a single time. The rest stays in shared memory. This also requires the indexing
//      to be constant, i.e. the window size should be a template argument.
// TODO Maybe look at other implementations for larger windows (e.g. with textures)?

namespace noa::cuda::signal {
    struct MedianFilterConfig {
        static constexpr i32 block_size_x = 16;
        static constexpr i32 block_size_y = 16;
        static constexpr i32 block_size = block_size_x * block_size_y;
    };

    template<typename T, typename U, typename I>
    void median_filter_1d(
            const T* input, const Strides4<I>& input_strides,
            U* output, const Strides4<I>& output_strides,
            const Shape4<i64>& shape, Border border_mode, i64 window_size, Stream& stream
    ) {
        using config_t = MedianFilterConfig;
        const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
        const i32 n_blocks_x = divide_up(shape_2d[1], config_t::block_size_x);
        const i32 n_blocks_y = divide_up(shape_2d[0], config_t::block_size_y);
        const auto launch_config = LaunchConfig{
            .n_blocks=dim3(static_cast<u32>(n_blocks_x * n_blocks_y),
                           static_cast<u32>(shape[1]),
                           static_cast<u32>(shape[0])),
            .n_threads=dim3(config_t::block_size_x, config_t::block_size_y),
        };

        using input_t = AccessorRestrict<const T, 4, I>;
        using output_t = AccessorRestrict<U, 4, I>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 3> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 3>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 5:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 5> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 5>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 7:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 7> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 7>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 9:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 9> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 9>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 11:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 11> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 11>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 13:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 13> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 13>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 15:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 15> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 15>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 17:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 17> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 17>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 19:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 19> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 19>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 21:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_1d<config_t, input_t, output_t, Border::REFLECT, 21> :
                        guts::median_filter_1d<config_t, input_t, output_t, Border::ZERO, 21>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            default:
                panic("Unsupported window size. It should be an odd number from 1 to 21, got {}", window_size);
        }
    }

    template<typename T, typename U, typename I>
    void median_filter_2d(
            const T* input, Strides4<I> input_strides,
            U* output, Strides4<I> output_strides,
            Shape4<i64> shape, Border border_mode, i64 window_size, Stream& stream
    ) {
        const auto order_2d = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (any(order_2d != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        using config_t = MedianFilterConfig;
        const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
        const i32 n_blocks_x = divide_up(shape_2d[1], config_t::block_size_x);
        const i32 n_blocks_y = divide_up(shape_2d[0], config_t::block_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(static_cast<u32>(n_blocks_x * n_blocks_y),
                               static_cast<u32>(shape[1]),
                               static_cast<u32>(shape[0])),
                .n_threads=dim3(config_t::block_size_x, config_t::block_size_y),
        };

        using input_t = AccessorRestrict<const T, 4, I>;
        using output_t = AccessorRestrict<U, 4, I>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_2d<config_t, input_t, output_t, Border::REFLECT, 3> :
                        guts::median_filter_2d<config_t, input_t, output_t, Border::ZERO, 3>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 5:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_2d<config_t, input_t, output_t, Border::REFLECT, 5> :
                        guts::median_filter_2d<config_t, input_t, output_t, Border::ZERO, 5>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 7:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_2d<config_t, input_t, output_t, Border::REFLECT, 7> :
                        guts::median_filter_2d<config_t, input_t, output_t, Border::ZERO, 7>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 9:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_2d<config_t, input_t, output_t, Border::REFLECT, 9> :
                        guts::median_filter_2d<config_t, input_t, output_t, Border::ZERO, 9>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            case 11:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_2d<config_t, input_t, output_t, Border::REFLECT, 11> :
                        guts::median_filter_2d<config_t, input_t, output_t, Border::ZERO, 11>,
                        launch_config, input_accessor, output_accessor, shape_2d, n_blocks_x);
            default:
                panic("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
    }

    template<typename T, typename U, typename I>
    void median_filter_3d(
            const T* input, Strides4<I> input_strides,
            U* output, Strides4<I> output_strides,
            Shape4<i64> shape, Border border_mode, i64 window_size, Stream& stream
    ) {
        const auto order_3d = ni::order(output_strides.pop_front(), shape.pop_front());
        if (any(order_3d != Vec3<i64>{0, 1, 2})) {
            const auto order = (order_3d + 1).push_front(0);
            input_strides = input_strides.reorder(order);
            output_strides = output_strides.reorder(order);
            shape = shape.reorder(order);
        }

        using config_t = MedianFilterConfig;
        const auto shape_3d = shape.pop_front().as_safe<i32>();
        const i32 n_blocks_x = divide_up(shape_3d[2], config_t::block_size_x);
        const i32 n_blocks_y = divide_up(shape_3d[1], config_t::block_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(static_cast<u32>(n_blocks_x * n_blocks_y),
                               static_cast<u32>(shape_3d[0]),
                               static_cast<u32>(shape[0])),
                .n_threads=dim3(config_t::block_size_x, config_t::block_size_y),
        };

        using input_t = AccessorRestrict<const T, 4, I>;
        using output_t = AccessorRestrict<U, 4, I>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        switch (window_size) {
            case 3:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_3d<config_t, input_t, output_t, Border::REFLECT, 3> :
                        guts::median_filter_3d<config_t, input_t, output_t, Border::ZERO, 3>,
                        launch_config, input_accessor, output_accessor, shape_3d, n_blocks_x);
            case 5:
                return stream.enqueue(
                        border_mode == Border::REFLECT ?
                        guts::median_filter_3d<config_t, input_t, output_t, Border::REFLECT, 5> :
                        guts::median_filter_3d<config_t, input_t, output_t, Border::ZERO, 5>,
                        launch_config, input_accessor, output_accessor, shape_3d, n_blocks_x);
            default:
                panic("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
        }
    }
}
