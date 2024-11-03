#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::guts {
    struct ConvolveConfig {
        static constexpr u32 block_size_x = 16;
        static constexpr u32 block_size_y = 16;
        static constexpr u32 block_size = block_size_x * block_size_y;
    };

    // TODO Add constant memory with per-device stream lock to store the filter?

    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_1d(
        Input input, Output output, Filter filter,
        Shape2<i32> shape, i32 filter_size, u32 blocks_x
    ) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        if (gid[2] < shape[0]) {
            const i32 PADDING = filter_size - 1;
            const i32 HALO = PADDING / 2;
            const auto SHARED_LEN = static_cast<i32>(Config::block_size_x) + PADDING;
            shared += tid[0] * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            const auto input_1d = input[gid[0]][gid[1]][gid[2]];
            for (i32 lx = tid[1], gx = gid[3];
                 lx < SHARED_LEN;
                 lx += Config::block_size_x, gx += Config::block_size_x) {
                const i32 idx = gx - HALO;
                shared[lx] = (idx >= 0 and idx < shape[1]) ?
                             static_cast<filter_value_t>(input_1d[idx]) : filter_value_t{};
            }
            block_synchronize();

            if (gid[3] < shape[1]) {
                // Weighted sum.
                filter_value_t result{0};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_2d(
        Input input, Output output, Filter filter,
        Shape2<i32> shape, Shape2<i32> filter_shape, u32 blocks_x
    ) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        const auto OFFSET = static_cast<i32>(Config::block_size_x);
        const auto PADDING = Vec2<i32>::from_vec(filter_shape.vec - 1);
        const auto HALO = PADDING / 2;
        const auto SHARED_LEN = OFFSET + PADDING;

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Load to shared memory. Loop to take into account padding.
        const auto input_2d = input[gid[0]][gid[1]];
        for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            const i32 i_y = gy - HALO[0];
            const bool is_in_y = i_y >= 0 and i_y < shape[0];
            for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                const i32 i_x = gx - HALO[1];
                const bool is_in_x = i_x >= 0 and i_x < shape[1];
                shared[ly * SHARED_LEN[1] + lx] = is_in_y and is_in_x ?
                                                  static_cast<filter_value_t>(input_2d(i_y, i_x)) : filter_value_t{};
            }
        }
        block_synchronize();

        if (gid[2] < shape[0] and gid[3] < shape[1]) {
            // Weighted sum.
            filter_value_t result{};
            for (i32 y = 0; y < filter_shape[0]; ++y)
                for (i32 x = 0; x < filter_shape[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * filter[y * filter_shape[1] + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    template<typename Config, typename Input, typename Output, typename Filter, i32 FILTER_LEN>
    __global__ __launch_bounds__(ConvolveConfig::block_size)
    void convolve_3d_square(
        Input input, Output output, Filter filter,
        Shape3<i32> shape, u32 blocks_x
    ) {
        static_assert(is_odd(FILTER_LEN));
        constexpr i32 PADDING = FILTER_LEN - 1; // assume odd
        constexpr i32 HALO = FILTER_LEN / 2;
        constexpr auto SHARED_SHAPE = Shape3<i32>::from_values(
            FILTER_LEN, Config::block_size_y + PADDING, Config::block_size_x + PADDING);
        constexpr auto SHARED_SIZE = SHARED_SHAPE.n_elements();

        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        __shared__ filter_value_t shared[SHARED_SIZE];

        // Load shared memory. Loop to take into account padding.
        const auto input_3d = input[gid[0]];
        for (i32 lz = 0, gz = gid[1]; lz < SHARED_SHAPE[0]; ++lz, ++gz) {
            const i32 i_z = gz - HALO;
            const i32 tmp_z = lz * SHARED_SHAPE[1] * SHARED_SHAPE[2];
            const bool is_in_z = i_z >= 0 and i_z < shape[0];
            for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[1]; ly += Config::block_size_y, gy += Config::block_size_y) {
                const i32 i_y = gy - HALO;
                const i32 tmp = tmp_z + ly * SHARED_SHAPE[2];
                const bool is_in_y = i_y >= 0 and i_y < shape[1];
                for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[2]; lx += Config::block_size_x, gx += Config::block_size_x) {
                    const i32 i_x = gx - HALO;
                    const bool is_in_x = i_x >= 0 and i_x < shape[2];
                    shared[tmp + lx] = (is_in_z and is_in_y and is_in_x) ?
                                       static_cast<filter_value_t>(input_3d(i_z, i_y, i_x)) : filter_value_t{};
                }
            }
        }
        block_synchronize();

        if (gid[2] < shape[1] and gid[3] < shape[2]) {
            // Weighted sum.
            filter_value_t result{0};
            for (i32 z = 0; z < FILTER_LEN; ++z)
                for (i32 y = 0; y < FILTER_LEN; ++y)
                    for (i32 x = 0; x < FILTER_LEN; ++x)
                        result += shared[(z * SHARED_SHAPE[1] + tid[0] + y) * SHARED_SHAPE[2] + tid[1] + x] *
                                  filter[(z * FILTER_LEN + y) * FILTER_LEN + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_3d(
        Input input, Output output, Filter filter,
        Shape3<i32> shape, Shape3<i32> filter_length, u32 blocks_x
    ) {
        const auto padding = filter_length.vec - 1; // assume odd
        const auto halo = padding / 2;
        const auto shared_shape = Vec3<i32>::from_values(
            filter_length[0],
            Config::block_size_y + padding[1],
            Config::block_size_x + padding[2]);

        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Load shared memory. Loop to take into account padding.
        const auto input_3d = input[gid[0]];
        for (i32 lz = 0, gz = gid[1]; lz < shared_shape[0]; ++lz, ++gz) {
            const i32 i_z = gz - halo[0];
            const i32 tmp_z = lz * shared_shape[1] * shared_shape[2];
            const bool is_in_z = i_z >= 0 and i_z < shape[0];
            for (i32 ly = tid[0], gy = gid[2]; ly < shared_shape[1]; ly += Config::block_size_y, gy += Config::block_size_y) {
                const i32 i_y = gy - halo[1];
                const i32 tmp = tmp_z + ly * shared_shape[2];
                const bool is_in_y = i_y >= 0 and i_y < shape[1];
                for (i32 lx = tid[1], gx = gid[3]; lx < shared_shape[2]; lx += Config::block_size_x, gx += Config::block_size_x) {
                    const i32 i_x = gx - halo[2];
                    const bool is_in_x = i_x >= 0 and i_x < shape[2];
                    shared[tmp + lx] = (is_in_z and is_in_y and is_in_x) ?
                                       static_cast<filter_value_t>(input_3d(i_z, i_y, i_x)) : filter_value_t{0};
                }
            }
        }
        block_synchronize();

        if (gid[2] < shape[1] and gid[3] < shape[2]) {
            // Weighted sum.
            filter_value_t result{};
            for (i32 z = 0; z < filter_length[0]; ++z)
                for (i32 y = 0; y < filter_length[1]; ++y)
                    for (i32 x = 0; x < filter_length[2]; ++x)
                        result += shared[(z * shared_shape[1] + tid[0] + y) * shared_shape[2] + tid[1] + x] *
                                  filter[(z * filter_length[1] + y) * filter_length[2] + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    // TODO This is identical to the convolve1_ kernel.
    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_separable_x(
        Input input, Output output, Filter filter,
        Shape2<i32> shape_yx, i32 filter_size, u32 blocks_x
    ) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Filter along x.
        const auto input_x = input[gid[0]][gid[1]][gid[2]];
        if (gid[2] < shape_yx[0]) {
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len = static_cast<i32>(Config::block_size_x) + padding;
            auto row = shared + tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (i32 lx = tid[1], gx = gid[3]; lx < shared_len; lx += Config::block_size_x, gx += Config::block_size_x) {
                const i32 i_x = gx - halo;
                row[lx] = i_x >= 0 and i_x < shape_yx[1] ? static_cast<filter_value_t>(input_x[i_x]) : filter_value_t{};
            }
            block_synchronize();

            if (gid[3] < shape_yx[1]) {
                // Weighted sum.
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += row[tid[1] + idx] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_separable_y(
        Input input, Output output, Filter filter,
        Shape2<i32> shape_yx, i32 filter_size, u32 blocks_x
    ) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            blockIdx.y,
            Config::block_size_y * index[0] + tid[0],
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Filter along y.
        if (gid[3] < shape_yx[1]) {
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len_y = static_cast<i32>(Config::block_size_y) + padding;

            const auto input_yx = input[gid[0]][gid[1]];
            for (i32 ly = tid[0], gy = gid[2]; ly < shared_len_y; ly += Config::block_size_y, gy += Config::block_size_y) {
                const i32 i_y = gy - halo;
                shared[ly * Config::block_size_x + tid[1]] =
                    i_y >= 0 and i_y < shape_yx[0] ?
                    static_cast<filter_value_t>(input_yx(i_y, gid[3])) : filter_value_t{};
            }
            block_synchronize();

            if (gid[2] < shape_yx[0]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * Config::block_size_x + tid[1]] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<typename Config, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Config::block_size)
    void convolve_separable_z(
        Input input, Output output, Filter filter,
        Shape2<i32> shape_zx, i32 filter_size, u32 blocks_x
    ) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<i32>::from_values(
            blockIdx.z,
            Config::block_size_y * index[0] + tid[0],
            blockIdx.y,
            Config::block_size_x * index[1] + tid[1]);

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        if (gid[3] < shape_zx[1]) {
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len_z = static_cast<i32>(Config::block_size_y) + padding;

            const auto input_3d = input[gid[0]];
            for (i32 lz = tid[0], gz = gid[1]; lz < shared_len_z; lz += Config::block_size_y, gz += Config::block_size_y) {
                const i32 i_z = gz - halo;
                shared[lz * Config::block_size_x + tid[1]] =
                    i_z >= 0 and i_z < shape_zx[0] ?
                    static_cast<filter_value_t>(input_3d(i_z, gid[2], gid[3])) : filter_value_t{};
            }
            block_synchronize();

            // Weighted sum.
            if (gid[1] < shape_zx[0]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * Config::block_size_x + tid[1]] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }
}

#ifdef NOA_IS_OFFLINE
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
            .n_blocks = dim3(n_blocks_x * n_blocks_y, shape[1], shape[0]),
            .n_threads = dim3(config::block_size_x, config::block_size_y),
            .n_bytes_of_shared_memory =
            (config::block_size_x + filter_size - 1) * config::block_size_y * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_x<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(2, 3).as<i32>(),
                       static_cast<i32>(filter_size), n_blocks_x);
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
            .n_blocks = dim3(n_blocks_x * n_blocks_y, shape[1], shape[0]),
            .n_threads = dim3(config::block_size_x, config::block_size_y),
            .n_bytes_of_shared_memory =
            config::block_size_x * (config::block_size_y + filter_size - 1) * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_y<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(2, 3).as<i32>(),
                       static_cast<i32>(filter_size), n_blocks_x);
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
            .n_blocks = dim3(n_blocks_x * n_blocks_z, shape[2], shape[0]),
            .n_threads = dim3(config::block_size_x, config::block_size_y),
            .n_bytes_of_shared_memory =
            config::block_size_x * (config::block_size_y + filter_size - 1) * static_cast<u32>(sizeof(V)),
        };

        using input_accessor_t = AccessorRestrictU32<const T, 4>;
        using output_accessor_t = AccessorRestrictU32<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        stream.enqueue(convolve_separable_z<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                       launch_config,
                       input_accessor_t(input, input_strides.as_safe<u32>()),
                       output_accessor_t(output, output_strides.as_safe<u32>()),
                       filter_accessor_t(filter), shape.filter(1, 3).as<i32>(),
                       static_cast<i32>(filter_size), n_blocks_x);
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
                    filter, static_cast<u32>(filter_shape[0]), stream);
            } else if (filter_shape[1] > 1) {
                guts::launch_convolve_separable_y(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                    filter, static_cast<u32>(filter_shape[1]), stream);
            } else {
                guts::launch_convolve_separable_x(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(), shape.as_safe<u32>(),
                    filter, static_cast<u32>(filter_shape[2]), stream);
            }
        } else if (ndim == 2) {
            const auto filter_shape_2d = filter_shape.pop_front().as<i32>();
            const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
            const u32 n_blocks_x = divide_up(static_cast<u32>(shape_2d[1]), config::block_size_x);
            const u32 n_blocks_y = divide_up(static_cast<u32>(shape_2d[0]), config::block_size_y);
            const auto launch_config = LaunchConfig{
                    .n_blocks=dim3(n_blocks_x * n_blocks_y, static_cast<u32>(shape[1]), static_cast<u32>(shape[0])),
                    .n_threads=dim3(config::block_size_x, config::block_size_y),
                    .n_bytes_of_shared_memory=
                        (config::block_size_x + static_cast<u32>(filter_shape_2d[1]) - 1) *
                        (config::block_size_y + static_cast<u32>(filter_shape_2d[0]) - 1) * sizeof(V),
            };
            stream.enqueue(convolve_2d<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                           launch_config,
                           input_accessor_t(input, input_strides.as_safe<u32>()),
                           output_accessor_t(output, output_strides.as_safe<u32>()),
                           filter_accessor_t(filter), shape_2d, filter_shape_2d, n_blocks_x);

        } else if (ndim == 3) {
            const auto shape_3d = shape.pop_front().as_safe<i32>();
            const u32 n_blocks_x = divide_up(static_cast<u32>(shape_3d[2]), config::block_size_x);
            const u32 n_blocks_y = divide_up(static_cast<u32>(shape_3d[1]), config::block_size_y);
            auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x * n_blocks_y, static_cast<u32>(shape[1]), static_cast<u32>(shape[0])),
                .n_threads = dim3(config::block_size_x, config::block_size_y),
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
                    (config::block_size_x + static_cast<u32>(filter_shape[2]) - 1) *
                    (config::block_size_y + static_cast<u32>(filter_shape[1]) - 1) *
                    static_cast<u32>(filter_shape[0]) * sizeof(V);
                stream.enqueue(convolve_3d<config, input_accessor_t, output_accessor_t, filter_accessor_t>,
                               launch_config,
                               input_accessor_t(input, input_strides.as_safe<u32>()),
                               output_accessor_t(output, output_strides.as_safe<u32>()),
                               filter_accessor_t(filter), shape_3d,
                               filter_shape.as<i32>(), n_blocks_x);
            }
        } else if (all(filter_shape == 1)) {
            V filter_value;
            copy(filter, &filter_value, 1, stream);

            auto order = ni::order(output_strides, shape);
            if (vany(NotEqual{}, order, Vec{0, 1, 2, 3})) {
                input_strides = ni::reorder(input_strides, order);
                output_strides = ni::reorder(output_strides, order);
            }
            const auto input_accessor = input_accessor_t(input, input_strides.as_safe<u32>());
            const auto output_accessor = output_accessor_t(output, output_strides.as_safe<u32>());
            const auto value = AccessorValue<T>(static_cast<T>(filter_value));
            return ewise(shape.as_safe<i32>(), Multiply{},
                         make_tuple(input_accessor, value),
                         make_tuple(output_accessor),
                         stream);
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
        const auto u_filter_depth_size = static_cast<u32>(filter_depth_size);
        const auto u_filter_height_size = static_cast<u32>(filter_height_size);
        const auto u_filter_width_size = static_cast<u32>(filter_width_size);

        if (u_filter_depth_size <= 0)
            filter_depth = nullptr;
        if (u_filter_height_size <= 0)
            filter_height = nullptr;
        if (u_filter_width_size <= 0)
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
            buffer = AllocatorDevice<V>::allocate_async(shape.n_elements(), stream);
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }
        const auto u_tmp_strides = tmp_strides.as_safe<u32>();

        if (filter_depth and filter_height and filter_width) {
            guts::launch_convolve_separable_z(
                input, u_input_strides, output, u_output_strides, u_shape,
                filter_depth, u_filter_depth_size, stream);
            guts::launch_convolve_separable_y(
                output, u_output_strides, tmp, u_tmp_strides, u_shape,
                filter_height, u_filter_height_size, stream);
            guts::launch_convolve_separable_x(
                tmp, u_tmp_strides, output, u_output_strides, u_shape,
                filter_width, u_filter_width_size, stream);

        } else if (filter_depth and filter_height) {
            guts::launch_convolve_separable_z(
                input, u_input_strides, tmp, u_tmp_strides, u_shape,
                filter_depth, u_filter_depth_size, stream);
            guts::launch_convolve_separable_y(
                tmp, u_tmp_strides, output, u_output_strides, u_shape,
                filter_height, u_filter_height_size, stream);

        } else if (filter_depth and filter_width) {
            guts::launch_convolve_separable_z(
                input, u_input_strides, tmp, u_tmp_strides, u_shape,
                filter_depth, u_filter_depth_size, stream);
            guts::launch_convolve_separable_x(
                tmp, u_tmp_strides, output, u_output_strides, u_shape,
                filter_width, u_filter_width_size, stream);

        } else if (filter_height and filter_width) {
            guts::launch_convolve_separable_y(
                input, u_input_strides, tmp, u_tmp_strides, u_shape,
                filter_height, u_filter_height_size, stream);
            guts::launch_convolve_separable_x(
                tmp, u_tmp_strides, output, u_output_strides, u_shape,
                filter_width, u_filter_width_size, stream);

        } else if (filter_depth) {
            guts::launch_convolve_separable_z(
                input, u_input_strides, output, u_output_strides, u_shape,
                filter_depth, u_filter_depth_size, stream);
        } else if (filter_height) {
            guts::launch_convolve_separable_y(
                input, u_input_strides, output, u_output_strides, u_shape,
                filter_height, u_filter_height_size, stream);
        } else if (filter_width) {
            guts::launch_convolve_separable_x(
                input, u_input_strides, output, u_output_strides, u_shape,
                filter_width, u_filter_width_size, stream);
        }
    }
}
#endif
