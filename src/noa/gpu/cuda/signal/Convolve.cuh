#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::guts {
    using ConvolveBlock = StaticBlock<16, 16, 1>;

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_1d(
        Input input, Output output, Filter filter,
        Shape2<i64> shape, i32 filter_size,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_1d = input[gid[0]][gid[1]][gid[2]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        if (gid[2] < shape[0]) {
            const i32 PADDING = filter_size - 1;
            const i32 HALO = PADDING / 2;
            const auto SHARED_LEN = static_cast<i32>(Block::block_size_x) + PADDING;
            shared += tid[0] * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            i32 lx = tid[1];
            i64 gx = gid[3];
            for (; lx < SHARED_LEN; lx += Block::block_size_x, gx += Block::block_size_x) {
                const i64 ix = gx - HALO;

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (ix >= 0 and ix < shape[1])
                        value = static_cast<filter_value_t>(input_1d[ix]);
                } else {
                    const auto ix_reflected = ni::index_at<Border::REFLECT>(ix, shape[1]);
                    value = static_cast<filter_value_t>(input_1d[ix_reflected]);
                }
                shared[lx] = value;
            }
            block_synchronize();

            if (gid[3] < shape[1]) {
                filter_value_t result{0};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_2d(
        Input input, Output output, Filter filter,
        Shape2<i64> shape, Shape2<i32> filter_shape,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_2d = input[gid[0]][gid[1]];

        const auto OFFSET = static_cast<i32>(Block::block_size_x);
        const auto PADDING = Vec2<i32>::from_vec(filter_shape.vec - 1);
        const auto HALO = PADDING / 2;
        const auto SHARED_LEN = OFFSET + PADDING;

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Load to shared memory. Loop to take into account padding.
        i32 ly, lx;
        i64 gy, gx;
        for (ly = tid[0], gy = gid[2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            for (lx = tid[1], gx = gid[3]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                const i64 iy = gy - HALO[0];
                const i64 ix = gx - HALO[1];

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (iy >= 0 and iy < shape[0] and ix >= 0 and ix < shape[1])
                        value = static_cast<filter_value_t>(input_2d(iy, ix));
                } else {
                    const auto iy_reflected = ni::index_at<Border::REFLECT>(iy, shape[0]);
                    const auto ix_reflected = ni::index_at<Border::REFLECT>(ix, shape[1]);
                    value = static_cast<filter_value_t>(input_2d(iy_reflected, ix_reflected));
                }
                shared[ly * SHARED_LEN[1] + lx] = value;
            }
        }
        block_synchronize();

        if (gid[2] < shape[0] and gid[3] < shape[1]) {
            filter_value_t result{};
            for (i32 y = 0; y < filter_shape[0]; ++y)
                for (i32 x = 0; x < filter_shape[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * filter[y * filter_shape[1] + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter, i32 FILTER_LEN>
    __global__ __launch_bounds__(ConvolveBlock::block_size)
    void convolve_3d_square(
        Input input, Output output, Filter filter, Shape3<i64> shape,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        static_assert(is_odd(FILTER_LEN));
        constexpr i32 PADDING = FILTER_LEN - 1; // assume odd
        constexpr i32 HALO = FILTER_LEN / 2;
        constexpr auto SHARED_SHAPE = Shape<i32, 3>::from_values(
            FILTER_LEN,
            Block::block_size_y + PADDING,
            Block::block_size_x + PADDING
        );
        constexpr auto SHARED_SIZE = SHARED_SHAPE.n_elements();

        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_3d = input[gid[0]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        __shared__ filter_value_t shared[SHARED_SIZE];

        // Load shared memory. Loop to take into account padding.
        // i32 lz, ly, lx;
        // i64 gz, gy, gx;
        // for (lz = 0, gz = gid[1]; lz < SHARED_SHAPE[0]; ++lz, ++gz) {
        //     for (ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[1]; ly += Block::block_size_y, gy += Block::block_size_y) {
        //         for (lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[2]; lx += Block::block_size_x, gx += Block::block_size_x) {
        //             // const i64 iz = gz - HALO;
        //             // const i64 iy = gy - HALO;
        //             // const i64 ix = gx - HALO;
        //             const i64 i_z = gz - HALO;
        //             const i64 i_y = gy - HALO;
        //             const i64 i_x = gx - HALO;
        //
        //             filter_value_t value{};
        //             if constexpr (BORDER_ZERO) {
        //                 if (ni::is_inbound(shape, i_z, i_y, i_x))
        //                     value = static_cast<filter_value_t>(input_3d(i_z, i_y, i_x));
        //             } else {
        //                 const auto iz_ = ni::index_at<Border::REFLECT>(i_z, shape[0]);
        //                 const auto iy_ = ni::index_at<Border::REFLECT>(i_y, shape[1]);
        //                 const auto ix_ = ni::index_at<Border::REFLECT>(i_x, shape[2]);
        //                 value = static_cast<filter_value_t>(input_3d(iz_, iy_, ix_));
        //             }
        //             shared[(lz * SHARED_SHAPE[1] + ly) * SHARED_SHAPE[2] + lx] = value;
        //         }
        //     }
        // }

        i32 lz, ly, lx;
        i64 gz, gy, gx;
        for (lz = 0, gz = gid[1]; lz < SHARED_SHAPE[0]; ++lz, ++gz) {
            for (ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[1]; ly += Block::block_size_y, gy += Block::block_size_y) {
                for (lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[2]; lx += Block::block_size_x, gx += Block::block_size_x) {
                    const i64 iz = gz - HALO;
                    const i64 iy = gy - HALO;
                    const i64 ix = gx - HALO;

                    filter_value_t value{};
                    if constexpr (BORDER_ZERO) {
                        if (ni::is_inbound(shape, iz, iy, ix))
                            value = static_cast<filter_value_t>(input_3d(iz, iy, ix));
                    } else {
                        const auto iz_reflected = ni::index_at<Border::REFLECT>(iz, shape[0]);
                        const auto iy_reflected = ni::index_at<Border::REFLECT>(iy, shape[1]);
                        const auto ix_reflected = ni::index_at<Border::REFLECT>(ix, shape[2]);
                        value = static_cast<filter_value_t>(input_3d(iz_reflected, iy_reflected, ix_reflected));
                    }
                    shared[(lz * SHARED_SHAPE[1] + ly) * SHARED_SHAPE[2] + lx] = value;
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

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_3d(
        Input input, Output output, Filter filter,
        Shape3<i64> shape, Shape3<i32> filter_length,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto padding = filter_length.vec - 1; // assume odd
        const auto halo = padding / 2;
        const auto shared_shape = Vec3<i32>::from_values(
            filter_length[0],
            Block::block_size_y + padding[1],
            Block::block_size_x + padding[2]);

        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_3d = input[gid[0]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Load shared memory. Loop to take into account padding.
        i32 lz, ly, lx;
        i64 gz, gy, gx;
        for (lz = 0, gz = gid[1]; lz < shared_shape[0]; ++lz, ++gz) {
            for (ly = tid[0], gy = gid[2]; ly < shared_shape[1]; ly += Block::block_size_y, gy += Block::block_size_y) {
                for (lx = tid[1], gx = gid[3]; lx < shared_shape[2]; lx += Block::block_size_x, gx += Block::block_size_x) {
                    const i64 iz = gz - halo[0];
                    const i64 iy = gy - halo[1];
                    const i64 ix = gx - halo[2];

                    filter_value_t value{};
                    if constexpr (BORDER_ZERO) {
                        if (ni::is_inbound(shape, iz, iy, ix))
                            value = static_cast<filter_value_t>(input_3d(iz, iy, ix));
                    } else {
                        const auto iz_reflected = ni::index_at<Border::REFLECT>(iz, shape[0]);
                        const auto iy_reflected = ni::index_at<Border::REFLECT>(iy, shape[1]);
                        const auto ix_reflected = ni::index_at<Border::REFLECT>(ix, shape[2]);
                        value = static_cast<filter_value_t>(input_3d(iz_reflected, iy_reflected, ix_reflected));
                    }
                    shared[(lz * shared_shape[1] + ly) * shared_shape[2] + lx] = value;
                }
            }
        }
        block_synchronize();

        if (gid[2] < shape[1] and gid[3] < shape[2]) {
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
    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_separable_x(
        Input input, Output output, Filter filter,
        Shape2<i64> shape_yx, i32 filter_size,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_x = input[gid[0]][gid[1]][gid[2]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Filter along x.
        if (gid[2] < shape_yx[0]) {
            constexpr i32 BLOCK_SIZE_X = static_cast<i32>(Block::block_size_x);
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len = BLOCK_SIZE_X + padding;
            auto row = shared + tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            i32 lx = tid[1];
            i64 gx = gid[3];
            for (; lx < shared_len; lx += BLOCK_SIZE_X, gx += BLOCK_SIZE_X) {
                const i64 ix = gx - halo;
                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (ix >= 0 and ix < shape_yx[1])
                        value = static_cast<filter_value_t>(input_x[ix]);
                } else {
                    const auto ix_reflected = ni::index_at<Border::REFLECT>(ix, shape_yx[1]);
                    value = static_cast<filter_value_t>(input_x[ix_reflected]);
                }
                row[lx] = value;
            }
            block_synchronize();

            if (gid[3] < shape_yx[1]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += row[tid[1] + idx] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_separable_y(
        Input input, Output output, Filter filter,
        Shape2<i64> shape_yx, i32 filter_size,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy);
        const auto tid = thread_indices<i32, 2>();
        const auto input_yx = input[gid[0]][gid[1]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        // Filter along y.
        if (gid[3] < shape_yx[1]) {
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len_y = static_cast<i32>(Block::block_size_y) + padding;

            i32 ly = tid[0];
            i64 gy = gid[2];
            for (; ly < shared_len_y; ly += Block::block_size_y, gy += Block::block_size_y) {
                const i64 iy = gy - halo;

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (iy >= 0 and iy < shape_yx[0])
                        value = static_cast<filter_value_t>(input_yx(iy, gid[3]));
                } else {
                    const auto iy_reflected = ni::index_at<Border::REFLECT>(iy, shape_yx[0]);
                    value = static_cast<filter_value_t>(input_yx(iy_reflected, gid[3]));
                }
                shared[ly * Block::block_size_x + tid[1]] = value;
            }
            block_synchronize();

            if (gid[2] < shape_yx[0]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * Block::block_size_x + tid[1]] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<bool BORDER_ZERO, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_separable_z(
        Input input, Output output, Filter filter,
        Shape2<i64> shape_zx, i32 filter_size,
        Vec<u32, 2> block_offset_zy, u32 n_blocks_x
    ) {
        const auto gid = global_indices_4d<i64, Block>(n_blocks_x, block_offset_zy).filter(0, 2, 1, 3);
        const auto tid = thread_indices<i32, 2>();
        const auto input_3d = input[gid[0]];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        extern __shared__ filter_value_t shared[];

        if (gid[3] < shape_zx[1]) {
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len_z = static_cast<i32>(Block::block_size_y) + padding;

            i32 lz = tid[0];
            i64 gz = gid[1];
            for (; lz < shared_len_z; lz += Block::block_size_y, gz += Block::block_size_y) {
                const i64 iz = gz - halo;

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (iz >= 0 and iz < shape_zx[0])
                        value = static_cast<filter_value_t>(input_3d(iz, gid[2], gid[3]));
                } else {
                    const auto iz_reflected = ni::index_at<Border::REFLECT>(iz, shape_zx[0]);
                    value = static_cast<filter_value_t>(input_3d(iz_reflected, gid[2], gid[3]));
                }
                shared[lz * Block::block_size_x + tid[1]] = value;
            }
            block_synchronize();

            if (gid[1] < shape_zx[0]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * Block::block_size_x + tid[1]] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }
}

namespace noa::cuda::signal::guts {
    template<Border BORDER, typename T, typename U, typename V>
    void launch_convolve_separable_x(
        const T* input, const Strides4<i64>& input_strides,
        U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
        const V* filter, i64 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using input_t = AccessorRestrictI64<const T, 4>;
        using output_t = AccessorRestrictI64<U, 4>;
        using filter_t = AccessorRestrictContiguousI32<const V, 1>;

        const auto grid_x = GridXY(shape[3], shape[2], ConvolveBlock::block_size_x, ConvolveBlock::block_size_y);
        const auto grid_y = GridY(shape[1], 1);
        const auto grid_z = GridY(shape[0], 1);
        check(grid_x.n_launches() == 1);

        for (u32 z{}; z < grid_z.n_launches(); ++z) {
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                    .n_threads = dim3(ConvolveBlock::block_size_x, ConvolveBlock::block_size_y),
                    .n_bytes_of_shared_memory =
                        (ConvolveBlock::block_size_x + static_cast<u32>(filter_size) - 1) *
                        ConvolveBlock::block_size_y * static_cast<u32>(sizeof(V)),
                };
                const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                constexpr bool BORDER_ZERO = BORDER == Border::ZERO;
                stream.enqueue(
                    convolve_separable_x<BORDER_ZERO, ConvolveBlock, input_t, output_t, filter_t>,
                    config, input_t(input, input_strides), output_t(output, output_strides), filter_t(filter),
                    shape.filter(2, 3), static_cast<i32>(filter_size), grid_offset, grid_x.n_blocks_x()
                );
            }
        }
    }

    template<Border BORDER, typename T, typename U, typename V>
    void launch_convolve_separable_y(
        const T* input, const Strides4<i64>& input_strides,
        U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
        const V* filter, i64 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using input_t = AccessorRestrictI64<const T, 4>;
        using output_t = AccessorRestrictI64<U, 4>;
        using filter_t = AccessorRestrictContiguousI32<const V, 1>;

        const auto grid_x = GridXY(shape[3], shape[2], ConvolveBlock::block_size_x, ConvolveBlock::block_size_y);
        const auto grid_y = GridY(shape[1], 1);
        const auto grid_z = GridY(shape[0], 1);
        check(grid_x.n_launches() == 1);

        for (u32 z{}; z < grid_z.n_launches(); ++z) {
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                    .n_threads = dim3(ConvolveBlock::block_size_x, ConvolveBlock::block_size_y),
                    .n_bytes_of_shared_memory =
                        (ConvolveBlock::block_size_y + static_cast<u32>(filter_size) - 1) *
                        ConvolveBlock::block_size_x  * static_cast<u32>(sizeof(V)),
                };
                const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                constexpr bool BORDER_ZERO = BORDER == Border::ZERO;
                stream.enqueue(
                    convolve_separable_y<BORDER_ZERO, ConvolveBlock, input_t, output_t, filter_t>,
                    config, input_t(input, input_strides), output_t(output, output_strides), filter_t(filter),
                    shape.filter(2, 3), static_cast<i32>(filter_size), grid_offset, grid_x.n_blocks_x()
                );
            }
        }
    }

    template<Border BORDER, typename T, typename U, typename V>
    void launch_convolve_separable_z(
        const T* input, const Strides4<i64>& input_strides,
        U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
        const V* filter, i64 filter_size, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using input_t = AccessorRestrictI64<const T, 4>;
        using output_t = AccessorRestrictI64<U, 4>;
        using filter_t = AccessorRestrictContiguousI32<const V, 1>;

        const auto grid_x = GridXY(shape[3], shape[1], ConvolveBlock::block_size_x, ConvolveBlock::block_size_y);
        const auto grid_y = GridY(shape[2], 1);
        const auto grid_z = GridY(shape[0], 1);
        check(grid_x.n_launches() == 1);

        for (u32 z{}; z < grid_z.n_launches(); ++z) {
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                    .n_threads = dim3(ConvolveBlock::block_size_x, ConvolveBlock::block_size_y),
                    .n_bytes_of_shared_memory =
                        (ConvolveBlock::block_size_y + static_cast<u32>(filter_size) - 1) *
                        ConvolveBlock::block_size_x * static_cast<u32>(sizeof(V)),
                };
                const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                constexpr bool BORDER_ZERO = BORDER == Border::ZERO;
                stream.enqueue(
                    convolve_separable_z<BORDER_ZERO, ConvolveBlock, input_t, output_t, filter_t>,
                    config, input_t(input, input_strides), output_t(output, output_strides), filter_t(filter),
                    shape.filter(1, 3), static_cast<i32>(filter_size), grid_offset, grid_x.n_blocks_x()
                );
            }
        }
    }
}

namespace noa::cuda::signal {
    template<Border BORDER, typename T, typename U, typename V>
    void convolve(
        const T* input, Strides4<i64> input_strides,
        U* output, Strides4<i64> output_strides, const Shape4<i64>& shape,
        const V* filter, const Shape3<i64>& filter_shape, Stream& stream
    ) {
        using namespace noa::cuda::guts;
        using input_accessor_t = AccessorRestrictI64<const T, 4>;
        using output_accessor_t = AccessorRestrictI64<U, 4>;
        using filter_accessor_t = AccessorRestrictContiguousI32<const V, 1>;
        constexpr bool BORDER_ZERO = BORDER == Border::ZERO;

        const auto n_dimensions_to_convolve = sum(filter_shape > 1);
        const auto ndim = filter_shape.ndim();
        if (n_dimensions_to_convolve == 1) {
            if (filter_shape[0] > 1) {
                guts::launch_convolve_separable_z<BORDER>(
                    input, input_strides, output, output_strides, shape, filter, filter_shape[0], stream
                );
            } else if (filter_shape[1] > 1) {
                guts::launch_convolve_separable_y<BORDER>(
                    input, input_strides, output, output_strides, shape, filter, filter_shape[1], stream
                );
            } else {
                guts::launch_convolve_separable_x<BORDER>(
                    input, input_strides, output, output_strides, shape, filter, filter_shape[2], stream
                );
            }
        } else if (ndim == 2) {
            const auto filter_shape_2d = filter_shape.pop_front().as<i32>();
            const auto shape_2d = shape.filter(2, 3);

            const auto grid_x = GridXY(shape[3], shape[2], ConvolveBlock::block_size_x, ConvolveBlock::block_size_y);
            const auto grid_y = GridY(shape[1], 1);
            const auto grid_z = GridZ(shape[0], 1);
            check(grid_x.n_launches() == 1);

            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(ConvolveBlock::block_size_x, ConvolveBlock::block_size_y),
                        .n_bytes_of_shared_memory =
                            (ConvolveBlock::block_size_x + static_cast<u32>(filter_shape_2d[1]) - 1) *
                            (ConvolveBlock::block_size_y + static_cast<u32>(filter_shape_2d[0]) - 1) * sizeof(V),
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    stream.enqueue(
                        convolve_2d<BORDER_ZERO, ConvolveBlock, input_accessor_t, output_accessor_t, filter_accessor_t>,
                        config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                        filter_accessor_t(filter), shape_2d, filter_shape_2d, grid_offset, grid_x.n_blocks_x()
                    );
                }
            }
        } else if (ndim == 3) {
            const auto grid_x = GridXY(shape[3], shape[2], ConvolveBlock::block_size_x, ConvolveBlock::block_size_y);
            const auto grid_y = GridY(shape[1], 1);
            const auto grid_z = GridZ(shape[0], 1);
            check(grid_x.n_launches() == 1);

            const auto shape_3d = shape.pop_front();
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(ConvolveBlock::block_size_x, ConvolveBlock::block_size_y),
                    };
                    const auto grid_offset = Vec{grid_z.offset(z), grid_y.offset(y)};
                    if (all(filter_shape == 5)) {
                        stream.enqueue(
                            convolve_3d_square<BORDER_ZERO, ConvolveBlock, input_accessor_t, output_accessor_t, filter_accessor_t, 5>,
                            config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                            filter_accessor_t(filter), shape_3d, grid_offset, grid_x.n_blocks_x()
                        );
                    } else if (all(filter_shape == 3)) {
                        stream.enqueue(
                            convolve_3d_square<BORDER_ZERO, ConvolveBlock, input_accessor_t, output_accessor_t, filter_accessor_t, 3>,
                            config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                            filter_accessor_t(filter), shape_3d, grid_offset, grid_x.n_blocks_x()
                        );
                    } else {
                        config.n_bytes_of_shared_memory =
                            (ConvolveBlock::block_size_x + static_cast<u32>(filter_shape[2]) - 1) *
                            (ConvolveBlock::block_size_y + static_cast<u32>(filter_shape[1]) - 1) *
                            static_cast<u32>(filter_shape[0]) * sizeof(V);
                        stream.enqueue(
                            convolve_3d<BORDER_ZERO, ConvolveBlock, input_accessor_t, output_accessor_t, filter_accessor_t>,
                            config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                            filter_accessor_t(filter), shape_3d,
                            filter_shape.as<i32>(), grid_offset, grid_x.n_blocks_x()
                        );
                    }
                }
            }
        } else if (all(filter_shape == 1)) {
            V filter_value;
            copy(filter, &filter_value, 1, stream);

            auto order = ni::order(output_strides, shape);
            if (vany(NotEqual{}, order, Vec{0, 1, 2, 3})) {
                input_strides = ni::reorder(input_strides, order);
                output_strides = ni::reorder(output_strides, order);
            }
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);
            const auto value = AccessorValue<T>(static_cast<T>(filter_value));
            return ewise(shape, Multiply{}, make_tuple(input_accessor, value), make_tuple(output_accessor), stream);
        } else {
            panic("unreachable");
        }
    }

    template<Border BORDER, typename T, typename U, typename V>
    void convolve_separable(
        const T* input, const Strides4<i64>& input_strides,
        U* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
        const V* filter_depth, i64 filter_depth_size,
        const V* filter_height, i64 filter_height_size,
        const V* filter_width, i64 filter_width_size,
        V* tmp, Strides4<i64> tmp_strides, Stream& stream
    ) {
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
            buffer = AllocatorDevice<V>::allocate_async(shape.n_elements(), stream);
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }

        if (filter_depth and filter_height and filter_width) {
            guts::launch_convolve_separable_z<BORDER>(
                input, input_strides, output, output_strides, shape, filter_depth, filter_depth_size, stream
            );
            guts::launch_convolve_separable_y<BORDER>(
                output, output_strides, tmp, tmp_strides, shape, filter_height, filter_height_size, stream
            );
            guts::launch_convolve_separable_x<BORDER>(
                tmp, tmp_strides, output, output_strides, shape, filter_width, filter_width_size, stream
            );
        } else if (filter_depth and filter_height) {
            guts::launch_convolve_separable_z<BORDER>(
                input, input_strides, tmp, tmp_strides, shape, filter_depth, filter_depth_size, stream
            );
            guts::launch_convolve_separable_y<BORDER>(
                tmp, tmp_strides, output, output_strides, shape, filter_height, filter_height_size, stream
            );
        } else if (filter_depth and filter_width) {
            guts::launch_convolve_separable_z<BORDER>(
                input, input_strides, tmp, tmp_strides, shape, filter_depth, filter_depth_size, stream
            );
            guts::launch_convolve_separable_x<BORDER>(
                tmp, tmp_strides, output, output_strides, shape, filter_width, filter_width_size, stream
            );
        } else if (filter_height and filter_width) {
            guts::launch_convolve_separable_y<BORDER>(
                input, input_strides, tmp, tmp_strides, shape, filter_height, filter_height_size, stream
            );
            guts::launch_convolve_separable_x<BORDER>(
                tmp, tmp_strides, output, output_strides, shape, filter_width, filter_width_size, stream
            );
        } else if (filter_depth) {
            guts::launch_convolve_separable_z<BORDER>(
                input, input_strides, output, output_strides, shape, filter_depth, filter_depth_size, stream
            );
        } else if (filter_height) {
            guts::launch_convolve_separable_y<BORDER>(
                input, input_strides, output, output_strides, shape, filter_height, filter_height_size, stream
            );
        } else if (filter_width) {
            guts::launch_convolve_separable_x<BORDER>(
                input, input_strides, output, output_strides, shape, filter_width, filter_width_size, stream
            );
        }
    }
}
