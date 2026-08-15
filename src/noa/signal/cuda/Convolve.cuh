#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/Copy.cuh"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/Utils.cuh"

namespace noa::signal::cuda::details {
    using ConvolveBlock1D = noa::cuda::StaticBlock<noa::cuda::Constant::WARP_SIZE * 4, 1, 1>;
    using ConvolveBlock2D = noa::cuda::StaticBlock<noa::cuda::Constant::WARP_SIZE / 2, noa::cuda::Constant::WARP_SIZE / 2, 1>;
    template<usize N> using ConvolveBlockND = std::conditional_t<N == 1, ConvolveBlock1D, ConvolveBlock2D>;

    template<bool BORDER_ZERO, usize N, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_2d(
        Input input, Output output, Filter filter,
        Shape2 shape, Shape<i32, 2> filter_shape,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        const auto gid = noa::cuda::details::global_indices<isize, N, Block>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_2d = input[gid.template pop_front<2>()];

        const auto OFFSET = static_cast<i32>(Block::block_size_x);
        const auto PADDING = Vec<i32, 2>::from_vec(filter_shape.vec - 1);
        const auto HALO = PADDING / 2;
        const auto SHARED_LEN = OFFSET + PADDING;

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        auto* shared = noa::cuda::details::dynamic_shared_memory_pointer<filter_value_t>();

        // Load to shared memory. Loop to take into account padding.
        i32 ly, lx;
        isize gy, gx;
        for (ly = tid[0], gy = gid[N - 2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            for (lx = tid[1], gx = gid[N - 1]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                const isize iy = gy - HALO[0];
                const isize ix = gx - HALO[1];

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (iy >= 0 and iy < shape[0] and ix >= 0 and ix < shape[1])
                        value = static_cast<filter_value_t>(input_2d(iy, ix));
                } else {
                    const auto iy_reflected = index_at<Border::REFLECT>(shape[0], iy);
                    const auto ix_reflected = index_at<Border::REFLECT>(shape[1], ix);
                    value = static_cast<filter_value_t>(input_2d(iy_reflected, ix_reflected));
                }
                shared[ly * SHARED_LEN[1] + lx] = value;
            }
        }
        noa::cuda::details::block_synchronize();

        if (gid[N - 2] < shape[0] and gid[N - 1] < shape[1]) {
            filter_value_t result{};
            for (i32 y = 0; y < filter_shape[0]; ++y)
                for (i32 x = 0; x < filter_shape[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * filter[y * filter_shape[1] + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    template<bool BORDER_ZERO, usize N, typename Block, typename Input, typename Output, typename Filter, i32 FILTER_LEN>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_3d_square(
        Input input, Output output, Filter filter, Shape3 shape,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
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

        const auto gid = noa::cuda::details::global_indices<isize, N, Block>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_3d = input[gid.template pop_front<3>()];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        __shared__ filter_value_t shared[SHARED_SIZE];

        // Load shared memory. Loop to take into account padding.
        i32 lz, ly, lx;
        isize gz, gy, gx;
        for (lz = 0, gz = gid[N - 3]; lz < SHARED_SHAPE[0]; ++lz, ++gz) {
            for (ly = tid[0], gy = gid[N - 2]; ly < SHARED_SHAPE[1]; ly += Block::block_size_y, gy += Block::block_size_y) {
                for (lx = tid[1], gx = gid[N - 1]; lx < SHARED_SHAPE[2]; lx += Block::block_size_x, gx += Block::block_size_x) {
                    const isize iz = gz - HALO;
                    const isize iy = gy - HALO;
                    const isize ix = gx - HALO;

                    filter_value_t value{};
                    if constexpr (BORDER_ZERO) {
                        if (is_inbound(shape, iz, iy, ix))
                            value = static_cast<filter_value_t>(input_3d(iz, iy, ix));
                    } else {
                        const auto iz_reflected = index_at<Border::REFLECT>(shape[0], iz);
                        const auto iy_reflected = index_at<Border::REFLECT>(shape[1], iy);
                        const auto ix_reflected = index_at<Border::REFLECT>(shape[2], ix);
                        value = static_cast<filter_value_t>(input_3d(iz_reflected, iy_reflected, ix_reflected));
                    }
                    shared[(lz * SHARED_SHAPE[1] + ly) * SHARED_SHAPE[2] + lx] = value;
                }
            }
        }
        noa::cuda::details::block_synchronize();

        if (gid[N - 2] < shape[1] and gid[N - 1] < shape[2]) {
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

    template<bool BORDER_ZERO, usize N, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_3d(
        Input input, Output output, Filter filter,
        Shape3 shape, Shape<i32, 3> filter_length,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        const auto padding = filter_length.vec - 1; // assume odd
        const auto halo = padding / 2;
        const auto shared_shape = Vec<i32, 3>::from_values(
            filter_length[0],
            Block::block_size_y + padding[1],
            Block::block_size_x + padding[2]);

        const auto gid = noa::cuda::details::global_indices<isize, N, Block>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_3d = input[gid.template pop_front<3>()];

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        auto* shared = noa::cuda::details::dynamic_shared_memory_pointer<filter_value_t>();

        // Load shared memory. Loop to take into account padding.
        i32 lz, ly, lx;
        isize gz, gy, gx;
        for (lz = 0, gz = gid[N - 1]; lz < shared_shape[0]; ++lz, ++gz) {
            for (ly = tid[0], gy = gid[N - 2]; ly < shared_shape[1]; ly += Block::block_size_y, gy += Block::block_size_y) {
                for (lx = tid[1], gx = gid[N - 1]; lx < shared_shape[2]; lx += Block::block_size_x, gx += Block::block_size_x) {
                    const isize iz = gz - halo[0];
                    const isize iy = gy - halo[1];
                    const isize ix = gx - halo[2];

                    filter_value_t value{};
                    if constexpr (BORDER_ZERO) {
                        if (is_inbound(shape, iz, iy, ix))
                            value = static_cast<filter_value_t>(input_3d(iz, iy, ix));
                    } else {
                        const auto iz_reflected = index_at<Border::REFLECT>(shape[0], iz);
                        const auto iy_reflected = index_at<Border::REFLECT>(shape[1], iy);
                        const auto ix_reflected = index_at<Border::REFLECT>(shape[2], ix);
                        value = static_cast<filter_value_t>(input_3d(iz_reflected, iy_reflected, ix_reflected));
                    }
                    shared[(lz * shared_shape[1] + ly) * shared_shape[2] + lx] = value;
                }
            }
        }
        noa::cuda::details::block_synchronize();

        if (gid[N - 2] < shape[1] and gid[N - 1] < shape[2]) {
            filter_value_t result{};
            for (i32 z = 0; z < filter_length[0]; ++z)
                for (i32 y = 0; y < filter_length[1]; ++y)
                    for (i32 x = 0; x < filter_length[2]; ++x)
                        result += shared[(z * shared_shape[1] + tid[0] + y) * shared_shape[2] + tid[1] + x] *
                                  filter[(z * filter_length[1] + y) * filter_length[2] + x];
            output(gid) = static_cast<output_value_t>(result);
        }
    }

    template<bool BORDER_ZERO, usize N, usize R, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_separable_x(
        Input input, Output output, Filter filter,
        Shape<isize, R> shape_yx, // ((y,)x)
        i32 filter_size,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        const auto gid = noa::cuda::details::global_indices<isize, N, Block>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        auto* shared = noa::cuda::details::dynamic_shared_memory_pointer<filter_value_t>();

        bool is_valid_row;
        if constexpr (N == 1)
            is_valid_row = true;
        else
            is_valid_row = gid[N - 2] < shape_yx[0];

        // Filter along x.
        if (is_valid_row) {
            const auto input_x = input[gid.pop_back()];

            constexpr i32 BLOCK_SIZE_X = static_cast<i32>(Block::block_size_x);
            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len = BLOCK_SIZE_X + padding;
            auto row = shared + tid[0] * shared_len; // focus on current row

            // Load shared memory. Loop to take into account padding.
            i32 lx = tid[1];
            isize gx = gid[N - 1];
            for (; lx < shared_len; lx += BLOCK_SIZE_X, gx += BLOCK_SIZE_X) {
                const isize ix = gx - halo;
                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (ix >= 0 and ix < shape_yx[R - 1])
                        value = static_cast<filter_value_t>(input_x[ix]);
                } else {
                    const auto ix_reflected = index_at<Border::REFLECT>(shape_yx[R - 1], ix);
                    value = static_cast<filter_value_t>(input_x[ix_reflected]);
                }
                row[lx] = value;
            }
            noa::cuda::details::block_synchronize();

            if (gid[N - 1] < shape_yx[R - 1]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += row[tid[1] + idx] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<bool BORDER_ZERO, usize N, typename Block, typename Input, typename Output, typename Filter>
    __global__ __launch_bounds__(Block::block_size)
    void convolve_separable_y(
        Input input, Output output, Filter filter,
        Shape2 shape_yx, i32 filter_size,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        const auto gid = noa::cuda::details::global_indices<isize, N, Block>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();

        using output_value_t = nt::value_type_t<Output>;
        using filter_value_t = nt::mutable_value_type_t<Filter>;
        auto* shared = noa::cuda::details::dynamic_shared_memory_pointer<filter_value_t>();

        // Filter along y.
        if (gid[N - 1] < shape_yx[1]) {
            const auto input_yx = input[gid.template pop_back<2>()];

            const i32 padding = filter_size - 1;
            const i32 halo = padding / 2;
            const i32 shared_len_y = static_cast<i32>(Block::block_size_y) + padding;

            i32 ly = tid[0];
            isize gy = gid[N - 2];
            for (; ly < shared_len_y; ly += Block::block_size_y, gy += Block::block_size_y) {
                const isize iy = gy - halo;

                filter_value_t value{};
                if constexpr (BORDER_ZERO) {
                    if (iy >= 0 and iy < shape_yx[0])
                        value = static_cast<filter_value_t>(input_yx(iy, gid[N - 1]));
                } else {
                    const auto iy_reflected = index_at<Border::REFLECT>(shape_yx[0], iy);
                    value = static_cast<filter_value_t>(input_yx(iy_reflected, gid[N - 1]));
                }
                shared[ly * Block::block_size_x + tid[1]] = value;
            }
            noa::cuda::details::block_synchronize();

            if (gid[N - 2] < shape_yx[0]) {
                filter_value_t result{};
                for (i32 idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid[0] + idx) * Block::block_size_x + tid[1]] * filter[idx];
                output(gid) = static_cast<output_value_t>(result);
            }
        }
    }

    template<usize I, Border BORDER, typename T, typename U, typename V, usize N>
    void launch_convolve_separable(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides, Shape<isize, N> shape,
        const V* filter, isize filter_size, noa::cuda::Stream& stream
    ) {
        using namespace noa::signal::cuda::details;
        using input_t = AccessorRestrict<const T, N, isize>;
        using output_t = AccessorRestrict<U, N, isize>;
        using filter_t = AccessorRestrictContiguous<const V, 1, i32>;
        auto iaccessor = input_t(input, input_strides);
        auto oaccessor = output_t(output, output_strides);
        if constexpr (N >= 3 and I == N - 3) {
            shape = shape.swap(N - 3, N - 2);
            nd::permute_accessors(Vec<u32, N>::arange().swap(N - 3, N - 2), iaccessor, oaccessor);
        }

        // To simplify things, treat 1D has 2D with a 1D block.
        using Block = ConvolveBlockND<N>;
        auto block_shape = Shape<isize, N>::from_value(1);
        if constexpr (N >= 2)
            block_shape[N - 2] = static_cast<isize>(Block::block_size_y);
        block_shape[N - 1] = static_cast<isize>(Block::block_size_x);
        const auto grid = noa::cuda::GridND(shape, block_shape);
        check(grid.n_launches() == 1);

        usize n_bytes_of_shared_memory{};
        if constexpr (I == N - 1) {
            n_bytes_of_shared_memory =
                (Block::block_size_x + static_cast<u32>(filter_size) - 1) *
                 Block::block_size_y * static_cast<u32>(sizeof(V));
        } else {
            n_bytes_of_shared_memory =
                (Block::block_size_y + static_cast<u32>(filter_size) - 1) *
                 Block::block_size_x * static_cast<u32>(sizeof(V));
        }

        constexpr usize R = N == 1 ? 1 : 2;
        Shape<isize, R> shape_hw;
        if constexpr (N >= 2)
            shape_hw = shape.filter(N - 2, N - 1);
        else
            shape_hw = shape.filter(N - 1);

        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().pop_back();
        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = noa::cuda::LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                    .n_bytes_of_shared_memory = n_bytes_of_shared_memory,
                };
                const auto grid_offset = grid.block_offset_for_launch(z, y).template pop_front<(N <= 2 ? 3 - N : 0)>();
                constexpr bool BORDER_ZERO = BORDER == Border::ZERO;
                if constexpr (I == N - 1) {
                    stream.enqueue(
                        convolve_separable_x<BORDER_ZERO, N, R, Block, input_t, output_t, filter_t>,
                        config, iaccessor, oaccessor, filter_t(filter),
                        shape_hw, static_cast<i32>(filter_size), grid_offset, grid_fused_shape_in_x_unbatched
                    );
                } else {
                    stream.enqueue(
                        convolve_separable_y<BORDER_ZERO, N, Block, input_t, output_t, filter_t>,
                        config, iaccessor, oaccessor, filter_t(filter),
                        shape_hw, static_cast<i32>(filter_size), grid_offset, grid_fused_shape_in_x_unbatched
                    );
                }
            }
        }
    }
}

namespace noa::signal::cuda {
    template<Border BORDER, typename T, typename U, typename V, usize N, usize R>
    void convolve(
        const T* input, Strides<isize, N> input_strides,
        U* output, Strides<isize, N> output_strides, const Shape<isize, N>& shape,
        const V* filter, const Shape<isize, R>& filter_shape, noa::cuda::Stream& stream
    ) {
        using namespace noa::signal::cuda::details;
        using input_accessor_t = AccessorRestrict<const T, N, isize>;
        using output_accessor_t = AccessorRestrict<U, N, isize>;
        using filter_accessor_t = AccessorRestrictContiguous<const V, 1, i32>;
        constexpr bool BORDER_ZERO = BORDER == Border::ZERO;

        const auto n_dimensions_to_convolve = sum(filter_shape.cmp_gt(1));
        if (n_dimensions_to_convolve == 1) {
            if constexpr (R == 3) {
                if (filter_shape[R - 3] > 1) {
                    details::launch_convolve_separable<N - 3, BORDER>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[R - 3], stream
                    );
                    return;
                }
            }
            if constexpr (R >= 2) {
                if (filter_shape[R - 2] > 1) {
                    details::launch_convolve_separable<N - 2, BORDER>(
                        input, input_strides, output, output_strides, shape,
                        filter, filter_shape[R - 2], stream
                    );
                    return;
                }
            }
            details::launch_convolve_separable<N - 1, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter, filter_shape[R - 1], stream
            );
            return;
        }

        if constexpr (R >= 2) {
            const auto rank = filter_shape.rank(); // 2 or 3

            auto block_shape = Shape<isize, N>::from_value(1);
            block_shape[N - 2] = static_cast<isize>(ConvolveBlock2D::block_size_y);
            block_shape[N - 1] = static_cast<isize>(ConvolveBlock2D::block_size_x);
            const auto grid = noa::cuda::GridND(shape, block_shape);
            check(grid.n_launches() == 1);

            const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().pop_back();
            const auto filter_shape_i32 = filter_shape.template as<i32>();

            for (u32 z{}; z < grid.n_launches_z(); ++z) {
                for (u32 y{}; y < grid.n_launches_y(); ++y) {
                    auto config = noa::cuda::LaunchConfig{
                        .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                        .n_threads = dim3(ConvolveBlock2D::block_size_x, ConvolveBlock2D::block_size_y, 1),
                    };
                    const auto grid_offset = grid.block_offset_for_launch(z, y).template pop_front<(N <= 2 ? 3 - N : 0)>();
                    if (rank == 2) {
                        const auto shape_hw = shape.filter(N - 2, N - 1);
                        config.n_bytes_of_shared_memory =
                            (ConvolveBlock2D::block_size_x + static_cast<u32>(filter_shape[R - 1]) - 1) *
                            (ConvolveBlock2D::block_size_y + static_cast<u32>(filter_shape[R - 2]) - 1);
                        stream.enqueue(
                            convolve_2d<BORDER_ZERO, ConvolveBlock2D, input_accessor_t, output_accessor_t, filter_accessor_t>,
                            config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                            filter_accessor_t(filter), shape_hw, filter_shape_i32.pop_front(), grid_offset, grid_fused_shape_in_x_unbatched
                        );
                    }
                    if constexpr (R == 3) {
                        if (rank == 3) {
                            const auto shape_bhw = shape.filter(N - 3, N - 2, N - 1);
                            if (filter_shape == 5) {
                                stream.enqueue(
                                    convolve_3d_square<BORDER_ZERO, ConvolveBlock2D, input_accessor_t, output_accessor_t, filter_accessor_t, 5>,
                                    config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                                    filter_accessor_t(filter), shape_bhw, grid_offset, grid_fused_shape_in_x_unbatched
                                );
                            } else if (filter_shape == 3) {
                                stream.enqueue(
                                    convolve_3d_square<BORDER_ZERO, ConvolveBlock2D, input_accessor_t, output_accessor_t, filter_accessor_t, 3>,
                                    config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                                    filter_accessor_t(filter), shape_bhw, grid_offset, grid_fused_shape_in_x_unbatched
                                );
                            } else {
                                config.n_bytes_of_shared_memory =
                                    (ConvolveBlock2D::block_size_x + static_cast<u32>(filter_shape[R - 1]) - 1) *
                                    (ConvolveBlock2D::block_size_y + static_cast<u32>(filter_shape[R - 2]) - 1) *
                                    static_cast<u32>(filter_shape[R - 3]) * sizeof(V);
                                stream.enqueue(
                                    convolve_3d<BORDER_ZERO, ConvolveBlock2D, input_accessor_t, output_accessor_t, filter_accessor_t>,
                                    config, input_accessor_t(input, input_strides), output_accessor_t(output, output_strides),
                                    filter_accessor_t(filter), shape_bhw,
                                    filter_shape_i32, grid_offset, grid_fused_shape_in_x_unbatched
                                );
                            }
                        }
                    }
                }
            }
        }
        panic("unreachable");
    }

    template<Border BORDER, typename T, typename U, typename V, usize N>
    void convolve_separable(
        const T* input, const Strides<isize, N>& input_strides,
        U* output, const Strides<isize, N>& output_strides, const Shape<isize, N>& shape,
        const V* filter_depth, isize filter_depth_size,
        const V* filter_height, isize filter_height_size,
        const V* filter_width, isize filter_width_size,
        V* tmp, Strides<isize, N> tmp_strides, noa::cuda::Stream& stream
    ) {
        if (filter_depth_size <= 0 or N <= 2)
            filter_depth = nullptr;
        if (filter_height_size <= 0 or N <= 1)
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
        noa::cuda::AllocatorDevice::allocate_type<V> buffer{};
        if (not tmp and count > 1) {
            buffer = noa::cuda::AllocatorDevice::allocate_async<V>(shape.n_elements(), stream);
            tmp = buffer.get();
            tmp_strides = shape.strides();
        }

        constexpr auto D = N - 3;
        constexpr auto H = N - 2;
        constexpr auto W = N - 1;
        if constexpr (N >= 3) {
            if (filter_depth and filter_height and filter_width) {
                details::launch_convolve_separable<D, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, stream);
                details::launch_convolve_separable<H, BORDER>(
                    output, output_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, stream);
                details::launch_convolve_separable<W, BORDER>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, stream);
                return;
            }
        }

        if constexpr (N >= 2) {
            if constexpr (N >= 3) {
                if (filter_depth and filter_height) {
                    details::launch_convolve_separable<D, BORDER>(
                        input, input_strides, tmp, tmp_strides, shape,
                        filter_depth, filter_depth_size, stream);
                    details::launch_convolve_separable<H, BORDER>(
                        tmp, tmp_strides, output, output_strides, shape,
                        filter_height, filter_height_size, stream);
                    return;
                }
                if (filter_depth and filter_width) {
                    details::launch_convolve_separable<D, BORDER>(
                        input, input_strides, tmp, tmp_strides, shape,
                        filter_depth, filter_depth_size, stream);
                    details::launch_convolve_separable<W, BORDER>(
                        tmp, tmp_strides, output, output_strides, shape,
                        filter_width, filter_width_size, stream);
                    return;
                }
            }
            if (filter_height and filter_width) {
                details::launch_convolve_separable<H, BORDER>(
                    input, input_strides, tmp, tmp_strides, shape,
                    filter_height, filter_height_size, stream);
                details::launch_convolve_separable<W, BORDER>(
                    tmp, tmp_strides, output, output_strides, shape,
                    filter_width, filter_width_size, stream);
                return;
            }
        }

        if constexpr (N >= 3) {
            if (filter_depth) {
                details::launch_convolve_separable<D, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_depth, filter_depth_size, stream);
                return;
            }
        }
        if constexpr (N >= 2) {
            if (filter_height) {
                details::launch_convolve_separable<H, BORDER>(
                    input, input_strides, output, output_strides, shape,
                    filter_height, filter_height_size, stream);
                return;
            }
        }
        if (filter_width) {
            details::launch_convolve_separable<W, BORDER>(
                input, input_strides, output, output_strides, shape,
                filter_width, filter_width_size, stream);
        }
    }
}
