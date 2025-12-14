#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Interpolation.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// FIXME The implementation requires a single thread to go through the each line. Obviously, this leeds to poor
//       performance. We should try to optimize it at some point.

namespace noa::cuda::details {
    template<typename T>
    __global__ void cubic_bspline_prefilter_1d_x_inplace(T* input, Strides<u32, 2> strides, Shape<u32, 2> shape) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * strides[0];
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_1d_x(
        const T* __restrict__ input, Strides<u32, 2> input_strides,
        T* __restrict__ output, Strides<u32, 2> output_strides,
        Shape<u32, 2> shape
    ) {
        // process lines in x-direction
        const u32 batch = blockIdx.x * blockDim.x + threadIdx.x;
        if (batch >= shape[0])
            return;
        input += batch * input_strides[0];
        output += batch * output_strides[0];
        nd::BSplinePrefilter1d<T, i32>::filter(
            input, input_strides[1], output, output_strides[1], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_x_inplace(Accessor<T, 3, u32> input, Shape<u32, 2> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y]; // blockIdx.y == batch
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(
            input_1d.get(), input_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_x(
        AccessorRestrict<const T, 3, u32> input,
        AccessorRestrict<T, 3, u32> output,
        Shape<u32, 2> shape
    ) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= shape[0])
            return;
        const auto input_1d = input[blockIdx.y][y];
        const auto output_1d = output[blockIdx.y][y];
        nd::BSplinePrefilter1d<T, i32>::filter(
            input_1d.get(), input_1d.template stride<0>(),
            output.get(), output_1d.template stride<0>(), shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_2d_y(T* input, Strides<u32, 3> strides, Shape<u32, 2> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= shape[1])
            return;
        input += blockIdx.y * strides[0] + x * strides[2];
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[0]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_x_inplace(Accessor<T, 4, u32> input, Shape<u32, 3> shape) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] or y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(
            input_1d.get(), input_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_x(
        AccessorRestrict<const T, 4, u32> input,
        AccessorRestrict<T, 4, u32> output,
        Shape<u32, 3> shape
    ) {
        // process lines in x-direction
        const u32 y = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] or y >= shape[1])
            return;
        const auto input_1d = input[blockIdx.z][z][y];
        const auto output_1d = output[blockIdx.z][z][y];
        nd::BSplinePrefilter1d<T, i32>::filter(
            input_1d.get(), input_1d.template stride<0>(),
            output_1d.get(), output_1d.template stride<0>(), shape[2]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_y(T* input, Strides<u32, 4> strides, Shape<u32, 3> shape) {
        // process lines in y-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 z = blockIdx.y * blockDim.y + threadIdx.y;
        if (z >= shape[0] or x >= shape[2])
            return;
        input += ni::offset_at(strides, blockIdx.z, z) + x * strides[3];
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[2], shape[1]);
    }

    template<typename T>
    __global__ void cubic_bspline_prefilter_3d_z(T* input, Strides<u32, 4> strides, Shape<u32, 3> shape) {
        // process lines in z-direction
        const u32 x = blockIdx.x * blockDim.x + threadIdx.x;
        const u32 y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= shape[1] or x >= shape[2])
            return;
        input += blockIdx.z * strides[0] + y * strides[2] + x * strides[3];
        nd::BSplinePrefilter1d<T, i32>::filter_inplace(input, strides[1], shape[0]);
    }
}

namespace noa::cuda::details {
    template<typename T>
    void cubic_bspline_prefilter_1d(
        const T* input, const Strides<u32, 2>& input_strides,
        T* output, const Strides<u32, 2>& output_strides,
        const Shape<u32, 2>& shape, Stream& stream
    ) {
        // Each thread processes an entire batch.
        // This has the same problem as the 2d/3d case, memory reads/writes are not coalesced.
        const u32 n_threads = next_multiple_of(shape[0], 32u);
        const u32 n_blocks = divide_up(shape[0], n_threads);
        const auto config = LaunchConfig{n_blocks, n_threads};

        if (input == output) {
            stream.enqueue(
                details::cubic_bspline_prefilter_1d_x_inplace<T>,
                config, output, output_strides, shape);
        } else {
            stream.enqueue(
                details::cubic_bspline_prefilter_1d_x<T>,
                config, input, input_strides, output, output_strides, shape);
        }
    }

    template<typename T>
    void cubic_bspline_prefilter_2d(
        const T* input, const Strides<u32, 3>& input_strides,
        T* output, const Strides<u32, 3>& output_strides,
        const Shape<u32, 3>& shape, Stream& stream
    ) {
        // Each thread processes an entire line. The line is first x, then y.
        const u32 n_threads_x = shape[1] <= 128u ? 32u : 64u;
        const u32 n_threads_y = shape[2] <= 128u ? 32u : 64u;
        const dim3 n_blocks_x(divide_up(shape[1], n_threads_x), shape[0]);
        const dim3 n_blocks_y(divide_up(shape[2], n_threads_y), shape[0]);
        const auto config_x = LaunchConfig{n_blocks_x, n_threads_x};
        const auto config_y = LaunchConfig{n_blocks_y, n_threads_y};

        if (input == output) {
            const auto accessor = AccessorU32<T, 3>(output, output_strides);
            stream.enqueue(
                details::cubic_bspline_prefilter_2d_x_inplace<T>,
                config_x, accessor, shape.pop_front());
        } else {
            const auto input_accessor = AccessorRestrictU32<const T, 3>(input, input_strides);
            const auto output_accessor = AccessorRestrictU32<T, 3>(output, output_strides);
            stream.enqueue(
                details::cubic_bspline_prefilter_2d_x<T>,
                config_x, input_accessor, output_accessor, shape.pop_front());
        }
        stream.enqueue(
            details::cubic_bspline_prefilter_2d_y<T>,
            config_y, output, output_strides, shape.pop_front());
    }

    template<typename T>
    void cubic_bspline_prefilter_3d(
        const T* input, const Strides<u32, 4>& input_strides,
        T* output, const Strides<u32, 4>& output_strides,
        const Shape<u32, 4>& shape, Stream& stream
    ) {
        // Determine the optimal block dimensions:
        auto get_launch_config_3d = [](u32 batch, u32 dim0, u32 dim1) {
            const u32 n_threads_x = dim1 <= 128u ? 32u : 64u; // either 32 or 64 threads in the first dimension
            const u32 n_threads_y = min(next_multiple_of(dim0, 32u), 512u / n_threads_x); // 2d block up to 512 threads
            return LaunchConfig{
                .n_blocks = dim3(divide_up(dim1, n_threads_x), divide_up(dim0, n_threads_y), batch),
                .n_threads = dim3(n_threads_x, n_threads_y, 1),
            };
        };

        auto shape_3d = shape.filter(1, 2, 3);
        auto launch_config = get_launch_config_3d(shape[0], shape[1], shape[2]);
        if (input == output) {
            const auto accessor = AccessorU32<T, 4>(output, output_strides);
            stream.enqueue(
                details::cubic_bspline_prefilter_3d_x_inplace<T>,
                launch_config, accessor, shape_3d);
        } else {
            const auto input_accessor = AccessorRestrictU32<const T, 4>(input, input_strides);
            const auto output_accessor = AccessorRestrictU32<T, 4>(output, output_strides);
            stream.enqueue(
                details::cubic_bspline_prefilter_3d_x<T>,
                launch_config, input_accessor, output_accessor, shape_3d);
        }

        launch_config = get_launch_config_3d(shape[0], shape[1], shape[3]);
        stream.enqueue(
            details::cubic_bspline_prefilter_3d_y<T>,
            launch_config, output, output_strides, shape_3d);

        launch_config = get_launch_config_3d(shape[0], shape[2], shape[3]);
        stream.enqueue(
            details::cubic_bspline_prefilter_3d_z<T>,
            launch_config, output, output_strides, shape_3d);
    }
}

namespace noa::cuda {
    template<typename Value>
    void cubic_bspline_prefilter(
        const Value* input, Strides4 input_strides,
        Value* output, Strides4 output_strides,
        Shape4 shape, Stream& stream
    ) {
        // Reorder to rightmost.
        const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
        if (order != Vec<isize, 3>{0, 1, 2}) {
            const auto order_4d = (order + 1).push_front(0);
            input_strides = input_strides.reorder(order_4d);
            output_strides = output_strides.reorder(order_4d);
            shape = shape.reorder(order_4d);
        }

        const auto ndim = shape.ndim();
        if (ndim == 3) {
            details::cubic_bspline_prefilter_3d<Value>(
                input, input_strides.as_safe<u32>(),
                output, output_strides.as_safe<u32>(),
                shape.as_safe<u32>(), stream);
        } else if (ndim == 2) {
            details::cubic_bspline_prefilter_2d<Value>(
                input, input_strides.filter(0, 2, 3).as_safe<u32>(),
                output, output_strides.filter(0, 2, 3).as_safe<u32>(),
                shape.filter(0, 2, 3).as_safe<u32>(), stream);
        } else {
            details::cubic_bspline_prefilter_1d<Value>(
                input, input_strides.filter(0, 3).as_safe<u32>(),
                output, output_strides.filter(0, 3).as_safe<u32>(),
                shape.filter(0, 3).as_safe<u32>(), stream);
        }
    }
}
