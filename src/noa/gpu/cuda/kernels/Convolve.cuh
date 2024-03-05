#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

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
        constexpr auto SHARED_SIZE = SHARED_SHAPE.elements();

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
            shared += tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (i32 lx = tid[1], gx = gid[3]; lx < shared_len; lx += Config::block_size_x, gx += Config::block_size_x) {
                const i32 i_x = gx - halo;
                shared[lx] = i_x >= 0 and i_x < shape_yx[1] ?
                             static_cast<filter_value_t>(input_x[i_x]) : filter_value_t{};
            }
            block_synchronize();

            if (gid[3] < shape_yx[1]) {
                // Weighted sum.
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * filter[idx];
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
