#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/indexing/Layout.hpp"
#include "noa/gpu/cuda/kernels/Permute.cuh"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda::guts {
    struct PermuteConfig {
        static constexpr u32 tile_size = Constant::WARP_SIZE;
        static constexpr u32 block_size = 256;
        static constexpr u32 block_size_x = tile_size;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename T>
    void permute_0132(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        using config = PermuteConfig;
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
                is_multiple_of(shape_2d[0], config::tile_size) and
                is_multiple_of(shape_2d[1], config::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };
        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue(permute_0132_<config, true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0132_<config, false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }

    template<typename T>
    void permute_0213(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        using config = PermuteConfig;
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
                is_multiple_of(shape_2d[0], config::tile_size) and
                is_multiple_of(shape_2d[1], config::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0213_<config, true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0213_<config, false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }


    template<typename T>
    void permute_0312(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        using config = PermuteConfig;
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
                is_multiple_of(shape_2d[0], config::tile_size) and
                is_multiple_of(shape_2d[1], config::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0312_<config, true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0312_<config, false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }

    template<typename T>
    void permute_0231(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        using config = PermuteConfig;
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(1, 3);
        const bool are_multiple_tile =
                is_multiple_of(shape_2d[0], config::tile_size) and
                is_multiple_of(shape_2d[1], config::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, u_shape[2], u_shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>().filter(0, 2, 1, 3)); // Y -> Z'
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue(permute_0231_<config, true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0231_<config, false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }
    template<typename T>
    void permute_0321(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        using config = PermuteConfig;
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(1, 3);
        const bool are_multiple_tile =
                is_multiple_of(shape_2d[0], config::tile_size) and
                is_multiple_of(shape_2d[1], config::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, u_shape[2], u_shape[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>().filter(0, 2, 1, 3));
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0321_<config, true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0321_<config, false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }

    template<typename T>
    void permute_0132_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        check(shape[3] == shape[2],
              "For a \"0132\" in-place permutation, shape[2] should be equal to shape[3]. Got shape={}", shape);

        using config = PermuteConfig;
        const auto shape_u32 = shape.as_safe<u32>();
        const bool is_multiple_tile = is_multiple_of(shape_u32[3], config::tile_size);
        const auto accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        const u32 n_blocks_x = divide_up(shape_u32[3], config::tile_size); // blocks_y == blocks_x
        const auto launch_config = LaunchConfig{
            .n_blocks=dim3(n_blocks_x * n_blocks_x, shape_u32[1], shape_u32[0]), // about less than half will be idle blocks...
            .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        if (is_multiple_tile)
            stream.enqueue(permute_0132_inplace_<config, true, T>, launch_config, accessor, shape_u32[3], n_blocks_x);
        else
            stream.enqueue(permute_0132_inplace_<config, false, T>, launch_config, accessor, shape_u32[3], n_blocks_x);
    }

    template<typename T>
    void permute_0213_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        check(shape[1] == shape[2],
              "For a \"0213\" in-place permutation, shape[1] should be equal to shape[2]. Got shape={}", shape);

        using config = PermuteConfig;
        const auto shape_u32 = shape.as_safe<u32>();
        const auto shape_2d = shape_u32.filter(2, 3);
        const bool is_multiple_tile = is_multiple_of(shape_2d[0], config::tile_size);
        const auto accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        const u32 n_blocks_x = divide_up(shape_2d[1], config::tile_size);
        const u32 n_blocks_y = divide_up(shape_2d[0], config::tile_size);
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_y, shape_u32[1], shape_u32[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        if (is_multiple_tile)
            stream.enqueue(permute_0213_inplace_<config, true, T>, launch_config, accessor, shape_2d, n_blocks_x);
        else
            stream.enqueue(permute_0213_inplace_<config, false, T>, launch_config, accessor, shape_2d, n_blocks_x);
    }

    template<typename T>
    void permute_0321_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
        check(shape[1] == shape[3],
              "For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got shape={}", shape);

        using config = PermuteConfig;
        const auto shape_u32 = shape.as_safe<u32>();
        const bool is_multiple_tile = is_multiple_of(shape_u32[1], config::tile_size);
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        const u32 n_blocks_x = divide_up(shape_u32[1], config::tile_size); // blocks_z == blocks_x
        const auto launch_config = LaunchConfig{
                .n_blocks=dim3(n_blocks_x * n_blocks_x, shape_u32[2], shape_u32[0]),
                .n_threads=dim3(config::block_size_x, config::block_size_y, 1),
        };

        if (is_multiple_tile) {
            stream.enqueue(permute_0321_inplace_<config, true, T>, launch_config,
                           output_accessor, shape_u32[1], n_blocks_x);
        } else {
            stream.enqueue(permute_0321_inplace_<config, false, T>, launch_config,
                           output_accessor, shape_u32[1], n_blocks_x);
        }
    }
}

namespace noa::cuda {
    template<typename T>
    void permute_copy(
            const T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            T* output, const Strides4<i64>& output_strides,
            const Vec4<i64>& permutation, Stream& stream
    ) {
        const auto idx = permutation[0] * 1000 +
                         permutation[1] * 100 +
                         permutation[2] * 10 +
                         permutation[3];
        if (input == output) {
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return guts::permute_0213_inplace(output, output_strides, input_shape, stream);
                case 132:
                    return guts::permute_0132_inplace(output, output_strides, input_shape, stream);
                case 321:
                    return guts::permute_0321_inplace(output, output_strides, input_shape, stream);
                default:
                    panic("The in-place permutation {} is not supported", permutation);
            }
        } else {
            switch (idx) {
                case 123:
                    return copy(input, input_strides, output, output_strides, input_shape, stream);
                case 213:
                    return guts::permute_0213(input, input_strides, output, output_strides, input_shape, stream);
                case 132:
                    return guts::permute_0132(input, input_strides, output, output_strides, input_shape, stream);
                case 312:
                    return guts::permute_0312(input, input_strides, output, output_strides, input_shape, stream);
                case 231:
                    return guts::permute_0231(input, input_strides, output, output_strides, input_shape, stream);
                case 321:
                    return guts::permute_0321(input, input_strides, output, output_strides, input_shape, stream);
                default:
                    // Much slower...
                    const auto output_shape = ni::reorder(input_shape, permutation);
                    const auto input_strides_permuted = ni::reorder(input_strides, permutation);
                    copy(input, input_strides_permuted, output, output_strides, output_shape, stream);
            }
        }
    }
}
#endif
