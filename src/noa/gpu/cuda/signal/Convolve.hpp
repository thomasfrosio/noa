#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/AllocatorDevice.hpp"
#include "noa/gpu/cuda/kernels/Convolve.cuh"

namespace noa::cuda::signal::guts {
    template<typename T, typename U, typename V>
    void launch_convolve_separable_x(
            const T* input, const Strides4<u32>& input_strides,
            U* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const V* filter, u32 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using config = ConvolveConfig;
        const u32 n_blocks_x = divide_up(shape[3], config::block_size_x);
        const u32 n_blocks_y = divide_up(shape[2], config::block_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, shape[1], shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y),
                .n_bytes_of_shared_memory=
                (config::block_size_x + filter_size - 1) * config::block_size_y * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_x<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(2, 3).as<i32>(), filter_size, n_blocks_x);
    }

    template<typename T, typename U, typename V>
    void launch_convolve_separable_y(
            const T* input, const Strides4<u32>& input_strides,
            U* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const V* filter, u32 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using config = ConvolveConfig;
        const u32 n_blocks_x = divide_up(shape[3], config::block_size_x);
        const u32 n_blocks_y = divide_up(shape[2], config::block_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, shape[1], shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y),
                .n_bytes_of_shared_memory=
                config::block_size_x * (config::block_size_y + filter_size - 1) * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_y<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(2, 3).as<i32>(), filter_size, n_blocks_x);
    }

    template<typename T, typename U, typename V>
    void launch_convolve_separable_z(
            const T* input, const Strides4<u32>& input_strides,
            U* output, const Strides4<u32>& output_strides, const Shape4<u32>& shape,
            const V* filter, u32 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using config = ConvolveConfig;
        const u32 n_blocks_x = divide_up(shape[3], config::block_size_x);
        const u32 n_blocks_z = divide_up(shape[1], config::block_size_y);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_z, shape[2], shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y),
                .n_bytes_of_shared_memory=
                config::block_size_x * (config::block_size_y + filter_size - 1) * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_z<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(1, 3).as<i32>(), filter_size, n_blocks_x);
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename V>
    void convolve(
            const T* input, Strides4<i64> input_strides,
            U* output, Strides4<i64> output_strides, const Shape4<i64>& shape,
            const V* filter, const Shape3<i64>& filter_shape, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using config = ConvolveConfig;
        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;

        const auto n_dimensions_to_convolve = sum(filter_shape > 1);
        const auto ndim = filter_shape.ndim();
        if (n_dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                guts::launch_convolve_separable_z(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[0], stream);
            } else if (filter_shape[1] > 1) {
                guts::launch_convolve_separable_y(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[1], stream);
            } else {
                guts::launch_convolve_separable_x(
                        input, input_strides.as_safe<u32>(),
                        output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                        filter, filter_shape[2], stream);
            }
        } else if (ndim == 2) {
            const auto filter_shape_2d = filter_shape.pop_front();
            const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
            const u32 n_blocks_x = divide_up(static_cast<u32>(shape_2d[1]), config::block_size_x);
            const u32 n_blocks_y = divide_up(static_cast<u32>(shape_2d[0]), config::block_size_y);
            const auto launch_config = LaunchConfig{
                    .n_blocks=dim3(n_blocks_x * n_blocks_y, shape[1], shape[0]),
                    .n_threads=dim3(config::block_size_x, config::block_size_y),
                    .n_bytes_of_shared_memory=
                    (config::block_size_x + filter_shape_2d[1] - 1) *
                    (config::block_size_y + filter_shape_2d[0] - 1) * sizeof(V),
            };
            stream.enqueue(convolve_2d<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                           launch_config,
                           input_accessor_t(input, input_strides.as_safe<u32>()),
                           output_accessor_t(output, output_strides.as_safe<u32>()),
                           filter_accessor_t(filter), shape_2d, filter_shape_2d.as_safe<i32>(), n_blocks_x);

        } else if (ndim == 3) {
            const auto shape_3d = shape.pop_front().as_safe<i32>();
            const u32 n_blocks_x = divide_up(static_cast<u32>(shape_3d[2]), config::block_size_x);
            const u32 n_blocks_y = divide_up(static_cast<u32>(shape_3d[1]), config::block_size_y);
            auto launch_config = LaunchConfig{
                    .n_blocks=dim3(n_blocks_x * n_blocks_y, static_cast<u32>(shape[1]), static_cast<u32>(shape[0])),
                    .n_threads=dim3(config::block_size_x, config::block_size_y),
            };

            if (all(filter_shape == 5)) {
                stream.enqueue(
                        convolve_3d_square<config, input_accessor_t, output_accessor_t, filter_accessor_t, 5>,
                        launch_config,
                        input_accessor_t(input, input_strides.as_safe<u32>()),
                        output_accessor_t(output, output_strides.as_safe<u32>()),
                        filter_accessor_t(filter), shape_3d, n_blocks_x);

            } else if (all(filter_shape == 3)) {
                stream.enqueue(
                        convolve_3d_square<config, input_accessor_t, output_accessor_t, filter_accessor_t, 3>,
                        launch_config,
                        input_accessor_t(input, input_strides.as_safe<u32>()),
                        output_accessor_t(output, output_strides.as_safe<u32>()),
                        filter_accessor_t(filter), shape_3d, n_blocks_x);
            } else {
                launch_config.n_bytes_of_shared_memory =
                        (config::block_size_x + filter_shape[2] - 1) *
                        (config::block_size_y + filter_shape[1] - 1) *
                        filter_shape[0] * sizeof(V);
                stream.enqueue(convolve_3d<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                               launch_config,
                               input_accessor_t(input, input_strides.as_safe<u32>()),
                               output_accessor_t(output, output_strides.as_safe<u32>()),
                               filter_accessor_t(filter), shape_3d,
                               filter_shape.as_safe<i32>(), n_blocks_x);
            }
        } else if (all(filter_shape == 1)) {
            T filter_value;
            copy(filter, &filter_value, 1, stream);

            auto order = ni::order(output_strides, shape);
            if (any(order != Vec4<i64>{0, 1, 2, 3})) {
                input_strides = ni::reorder(input_strides, order);
                output_strides = ni::reorder(output_strides, order);
            }
            const auto input_accessor = input_accessor_t(input, input_strides.as_safe<u32>());
            const auto output_accessor = output_accessor_t(output, output_strides.as_safe<u32>());
            const auto value = AccessorValue<T>(static_cast<T>(filter[0]));
            return ewise(shape, Multiply{}, make_tuple(input, value), make_tuple(output), stream);
        } else {
            panic("unreachable");
        }
    }

    template<typename T, typename U, typename V> requires nt::are_real_v<T, U, V>
    void convolve_separable(
            const T* input, const Strides4<i64>& input_strides,
            U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const V* filter_depth, i64 filter_depth_size,
            const V* filter_height, i64 filter_height_size,
            const V* filter_width, i64 filter_width_size,
            V* tmp, Strides4<i64> tmp_strides, Stream& stream
    ) {
        const auto u_input_strides = input_strides.as_safe<u32>();
        const auto u_output_strides = output_strides.as_safe<u32>();
        const auto u_shape = shape.as_safe<u32>();

        if (filter_depth_size <= 0)
            filter_depth = nullptr;
        if (filter_height_size <= 0)
            filter_height = nullptr;
        if (filter_width_size <= 0)
            filter_width = nullptr;

        // Allocate temp buffer if necessary.
        i32 count = 0;
        if (filter_depth)
            count += 1;
        if (filter_height)
            count += 1;
        if (filter_width)
            count += 1;
        typename AllocatorDevice<V>::unique_type buffer{};
        if (not tmp and count > 1) {
            buffer = AllocatorDevice<V>::allocate_async(shape.elements(), stream);
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }
        const auto u_tmp_strides = tmp_strides.as_safe<u32>();

        if (filter_depth and filter_height and filter_width) {
            guts::launch_convolve_separable_z(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            guts::launch_convolve_separable_y(
                    output, u_output_strides, tmp, u_tmp_strides, u_shape,
                    filter_height, filter_height_size, stream);
            guts::launch_convolve_separable_x(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_depth and filter_height) {
            guts::launch_convolve_separable_z(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            guts::launch_convolve_separable_y(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_height, filter_height_size, stream);

        } else if (filter_depth and filter_width) {
            guts::launch_convolve_separable_z(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
            guts::launch_convolve_separable_x(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_height and filter_width) {
            guts::launch_convolve_separable_y(
                    input, u_input_strides, tmp, u_tmp_strides, u_shape,
                    filter_height, filter_height_size, stream);
            guts::launch_convolve_separable_x(
                    tmp, u_tmp_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);

        } else if (filter_depth) {
            guts::launch_convolve_separable_z(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_depth, filter_depth_size, stream);
        } else if (filter_height) {
            guts::launch_convolve_separable_y(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_height, filter_height_size, stream);
        } else if (filter_width) {
            guts::launch_convolve_separable_x(
                    input, u_input_strides, output, u_output_strides, u_shape,
                    filter_width, filter_width_size, stream);
        }
    }
}
