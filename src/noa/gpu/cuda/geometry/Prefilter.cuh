#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/kernels/CubicBSplinePrefilter.cuh"

namespace noa::cuda::geometry::guts {
    template<typename T>
    void cubic_bspline_prefilter_1d(
            const T* input, const Strides2<u32>& input_strides,
            T* output, const Strides2<u32>& output_strides,
            const Shape2<u32>& shape, noa::cuda::Stream& stream
    ) {
        // Each thread processes an entire batch.
        // This has the same problem as the 2d/3d case, memory reads/writes are not coalesced.
        const u32 n_threads = next_multiple_of(shape[0], 32u);
        const u32 n_blocks = divide_up(shape[0], n_threads);
        const auto config = noa::cuda::LaunchConfig{n_blocks, n_threads};

        if (input == output) {
            stream.enqueue(
                    guts::cubic_bspline_prefilter_1d_x_inplace<T>,
                    config, output, output_strides, shape);
        } else {
            stream.enqueue(
                    guts::cubic_bspline_prefilter_1d_x<T>,
                    config, input, input_strides, output, output_strides, shape);
        }
    }

    template<typename T>
    void cubic_bspline_prefilter_2d(
            const T* input, const Strides3<u32>& input_strides,
            T* output, const Strides3<u32>& output_strides,
            const Shape3<u32>& shape, noa::cuda::Stream& stream
    ) {
        // Each thread processes an entire line. The line is first x, then y.
        const u32 n_threads_x = shape[1] <= 32u ? 32u : 64u;
        const u32 n_threads_y = shape[2] <= 32u ? 32u : 64u;
        const dim3 n_blocks_x(divide_up(shape[1], n_threads_x), shape[0]);
        const dim3 n_blocks_y(divide_up(shape[2], n_threads_y), shape[0]);
        const auto config_x = LaunchConfig{n_blocks_x, n_threads_x};
        const auto config_y = LaunchConfig{n_blocks_y, n_threads_y};

        if (input == output) {
            const auto accessor = AccessorU32<T, 3>(output, output_strides);
            stream.enqueue(
                    guts::cubic_bspline_prefilter_2d_x_inplace<T>,
                    config_x, accessor, shape.pop_front());
        } else {
            const auto input_accessor = AccessorRestrictU32<const T, 3>(input, input_strides);
            const auto output_accessor = AccessorRestrictU32<T, 3>(output, output_strides);
            stream.enqueue(
                    guts::cubic_bspline_prefilter_2d_x<T>,
                    config_x, input_accessor, output_accessor, shape.pop_front());
        }
        stream.enqueue(
                guts::cubic_bspline_prefilter_2d_y<T>,
                config_y, output, output_strides, shape.pop_front());
    }

    template<typename T>
    void cubic_bspline_prefilter_3d(
            const T* input, const Strides4<u32>& input_strides,
            T* output, const Strides4<u32>& output_strides,
            const Shape4<u32>& shape, noa::cuda::Stream& stream
    ) {
        // Determine the optimal block dimensions:
        auto get_launch_config_3d = [](u32 batch, u32 dim0, u32 dim1) {
            const u32 n_threads_x = dim0 <= 32u ? 32u : 64u; // either 32 or 64 threads in the first dimension
            const u32 n_threads_y = min(next_multiple_of(dim1, 32u), 512u / n_threads_x); // 2d block up to 512 threads
            return LaunchConfig{
                    .n_blocks=dim3(divide_up(dim0, n_threads_x), divide_up(dim1, n_threads_y), batch),
                    .n_threads=dim3(n_threads_x, n_threads_y, 1),
            };
        };

        auto launch_config = get_launch_config_3d(shape[0], shape[1], shape[2]);
        if (input == output) {
            const auto accessor = AccessorU32<T, 4>(output, output_strides);
            stream.enqueue(
                    guts::cubic_bspline_prefilter_3d_x_inplace<T>,
                    launch_config, accessor, shape.filter(1, 2, 3));
        } else {
            const auto input_accessor = AccessorRestrictU32<const T, 4>(input, input_strides);
            const auto output_accessor = AccessorRestrictU32<T, 4>(output, output_strides);
            stream.enqueue(
                    guts::cubic_bspline_prefilter_3d_x<T>,
                    launch_config, input_accessor, output_accessor, shape.filter(1, 2, 3));
        }

        launch_config = get_launch_config_3d(shape[0], shape[3], shape[1]);
        stream.enqueue(
                guts::cubic_bspline_prefilter_3d_y<T>,
                launch_config, output, output_strides, shape.filter(1, 2, 3));

        launch_config = get_launch_config_3d(shape[0], shape[3], shape[2]);
        stream.enqueue(
                guts::cubic_bspline_prefilter_3d_z<T>,
                launch_config, output, output_strides, shape.filter(1, 2, 3));
    }
}

namespace noa::cuda::geometry {
    template<typename Value>
    void cubic_bspline_prefilter(
            const Value* input, Strides4<i64> input_strides,
            Value* output, Strides4<i64> output_strides,
            Shape4<i64> shape, Stream& stream
    ) {
        // Reorder to rightmost.
        const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
        if (any(order != Vec3<i64>{0, 1, 2})) {
            const auto order_4d = (order + 1).push_front(0);
            input_strides = input_strides.reorder(order_4d);
            output_strides = output_strides.reorder(order_4d);
            shape = shape.reorder(order_4d);
        }

        const auto ndim = shape.ndim();
        if (ndim == 3) {
            guts::cubic_bspline_prefilter_3d<Value>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    shape.as_safe<u32>(), stream);
        } else if (ndim == 2) {
            guts::cubic_bspline_prefilter_2d<Value>(
                    input, input_strides.filter(0, 2, 3).as_safe<u32>(),
                    output, output_strides.filter(0, 2, 3).as_safe<u32>(),
                    shape.filter(0, 2, 3).as_safe<u32>(), stream);
        } else {
            guts::cubic_bspline_prefilter_1d<Value>(
                    input, input_strides.filter(0, 3).as_safe<u32>(),
                    output, output_strides.filter(0, 3).as_safe<u32>(),
                    shape.filter(0, 3).as_safe<u32>(), stream);
        }
    }
}
