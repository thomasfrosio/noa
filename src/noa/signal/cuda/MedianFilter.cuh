#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"

#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/Error.hpp"
#include "noa/runtime/cuda/Stream.hpp"

// The current implementations only supports small squared windows. This allows to:
//  1)  Load the windows for all threads in a block in shared memory. This is useful because windows overlap.
//  2)  The exchange search can be done on the per thread registers. Only about half of the window needs to
//      be on the registers at a single time. The rest stays in shared memory. This also requires the indexing
//      to be constant, i.e. the window size should be a template argument.
// TODO Maybe look at other implementations for larger windows (e.g. with textures)?

namespace noa::signal::cuda::details {
    using MedianFilterBlock1D = noa::cuda::StaticBlock<noa::cuda::Constant::WARP_SIZE * 4, 1, 1>;
    using MedianFilterBlock2D = noa::cuda::StaticBlock<noa::cuda::Constant::WARP_SIZE / 2, noa::cuda::Constant::WARP_SIZE / 2, 1>;
    template<usize N> using MedianFilterBlockND = std::conditional_t<N == 1, MedianFilterBlock1D, MedianFilterBlock2D>;

    // Ensures a < b. If not, swap.
    template<typename T>
    __forceinline__ __device__ void sort_swap_(T& a, T& b) {
        if (b < a) {
            // TODO Use XOR swap?
            T tmp = a;
            a = b;
            b = tmp;
        }
    }

    // Sets the array v, so that the min is at v[0] and max is at v[length-1].
    template<typename T>
    __forceinline__ __device__ void order_(T* v, i32 length) {
        for (i32 i = 0; i < length / 2; i++) // ensure min in first half, max in second half
            sort_swap_(v[i], v[length - 1 - i]);
        for (i32 i = 1; i < (length + 1) / 2; i++) // move min in first half to first pos
            sort_swap_(v[0], v[i]);
        for (i32 i = length - 2; i >= length / 2; i--) // move max in second half to last pos
            sort_swap_(v[i], v[length - 1]);
    }

    // Loads the shared memory array according to the BORDER_MODE.
    // input_row:   device memory. Should point to the beginning of the current row.
    // stride:      innermost stride of the input.
    // s_mem:       shared memory. Should point to the current element.
    // shape_x:     number of logical elements in x, x being the dimension of the 1D window.
    // gx:          index of the current element in x. If out of bound, add padding according to the BORDER_MODE.
    template<bool BORDER_REFLECT, i32 HALO, typename Accessor1d, typename T>
    __device__ void median_filter_1d_load_to_shared(
        const Accessor1d& input_row,
        T* s_mem, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_REFLECT) {
            if (gx < 0)
                *s_mem = input_row[-gx]; // pad left; requires shape_x >= HALO + 1, since gx >= -HALO
            else if (gx < shape_x)
                *s_mem = input_row[gx]; // in array, no padding
            else if (gx < shape_x + HALO)
                *s_mem = input_row[(2 * (shape_x - 1) - gx)]; // pad right; requires shape_x >= HALO  + 1
            // Otherwise, don't do anything since the *s_mem element will not be used anyway.

        } else {
            if (gx < 0 or gx >= shape_x)
                *s_mem = T{};
            else
                *s_mem = input_row[gx];
        }
    }

    template<typename Config, usize N, usize R, typename Input, typename Output, bool BORDER_REFLECT, i32 WINDOW_SIZE>
    __global__ __launch_bounds__(Config::block_size)
    void median_filter_1d(
        Input input,
        Output output,
        Shape<i32, R> shape, // ((y,)x)
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        static_assert(is_odd(WINDOW_SIZE));
        using input_value_t = Input::mutable_value_type;
        using output_value_t = Output::value_type;

        // The shared memory is the shape of the block + the padding in the first dimension.
        constexpr i32 PADDING = WINDOW_SIZE - 1;
        constexpr i32 HALO = PADDING / 2;
        constexpr auto SHARED_SHAPE = Shape<i32, 2>{Config::block_size_y, Config::block_size_x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.n_elements();
        __shared__ input_value_t shared_mem[SHARED_SIZE];

        const auto gid = noa::cuda::details::global_indices<i32, N, Config>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_1d = input[gid.pop_back()];
        auto* shared_mem_row = shared_mem + tid[0] * SHARED_SHAPE[1];

        bool is_valid_row;
        if constexpr (N == 1)
            is_valid_row = true;
        else
            is_valid_row = gid[N - 2] < shape[0];

        // There's no padding in y, so if out of bounds, stop.
        if (is_valid_row) {
            // Load shared memory. Loop to take into account padding.
            for (i32 lx = tid[1], gx = gid[N - 1]; lx < SHARED_SHAPE[1]; lx += Config::block_size_x, gx += Config::block_size_x) {
                median_filter_1d_load_to_shared<BORDER_REFLECT, HALO>(
                    input_1d, shared_mem_row + lx, shape[R - 1], gx - HALO);
            }
            noa::cuda::details::block_synchronize();

            // Only continue if not out of bound.
            if (gid[N - 1] < shape[R - 1]) {
                // The goal is to reduce register pressure as much as possible, but still use registers
                // to do the exchange sort. The window is divided into two half: the first "activated" half,
                // which is where the sorting happens, and the second half, which is the pool of contestants
                // that will be progressively added.
                constexpr i32 ACTIVE = WINDOW_SIZE / 2 + 2; // the final size should be at least 3.

                // Load active half to, hopefully, the local registers (otherwise spill to device memory).
                input_value_t v[ACTIVE]; // all indexing are known at compile, so registers should be used
                for (i32 x = 0; x < ACTIVE; ++x)
                    v[x] = shared_mem_row[tid[1] + x];

                order_(v, ACTIVE); // ensure min at 0, max at ACTIVE - 1

                // We can replace the min by a new contestant and then reorder.
                // We also don't have to compare the max at ACTIVE - 1, we just need to stack the new max at the end.
                // As such, as we add new contestants, we right-truncate the active half to ignore the previous max.
                i32 length = ACTIVE;
                for (i32 k = ACTIVE; k < WINDOW_SIZE; ++k) {
                    v[0] = shared_mem_row[tid[1] + k]; // replace min by new contestant
                    --length; // ignore the previous max at the end
                    order_(v, length); // min at 0, max at length - 1
                }

                // Since we know the min and max are at v[0] and v[length-1] respectively, to find the median,
                // we simply need to (exchange) sort the elements from v[1] to v[length-2] included.
                // The median will then be at length/2.
                for (i32 k = 1; k < length - 1; k++)
                    order_(v + k, length - k);
                output(gid) = static_cast<output_value_t>(v[length / 2]);
            }
        }
    }

    // Loads the (gx,gy) element according to the BORDER_MODE.
    // input_slice: device memory. Should point to the beginning of the current slice (z).
    // stride_y/x:  stride in y and x.
    // s_mem:       shared memory. Should point to the current element.
    // shape_y/x:   number of logical elements in y/x.
    // gy/x:        index of the current element in y/x. If out of bound, add padding according to the BORDER_MODE.
    template<bool BORDER_REFLECT, i32 HALO, typename Accessor2d, typename T>
    __device__ void median_filter_2d_load_to_shared(
        const Accessor2d& input_slice, T* s_mem,
        i32 shape_y, i32 gy, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_REFLECT) {
            if (gx < 0)
                gx *= -1;
            else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return; // don't do anything; gx is out of the shape + window so *s_mem won't be used anyway
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0)
                gy *= -1;
            else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            *s_mem = input_slice(gy, gx);

        } else {
            if (gx < 0 or gx >= shape_x or gy < 0 or gy >= shape_y)
                *s_mem = T{};
            else
                *s_mem = input_slice(gy, gx);
        }
    }

    template<typename Config, usize N, typename Input, typename Output, bool BORDER_REFLECT, i32 WINDOW_SIZE>
    __global__ __launch_bounds__(Config::block_size)
    void median_filter_2d(
        Input input,
        Output output,
        Shape<i32, 2> shape,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        static_assert(is_odd(WINDOW_SIZE));
        using input_value_t = Input::mutable_value_type;
        using output_value_t = Output::value_type;

        // The shared memory is the shape of the block + the padding the first and second dimension.
        constexpr i32 TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE;
        constexpr i32 PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr i32 HALO = PADDING / 2;
        constexpr auto SHARED_SHAPE = Shape<u32, 2>{Config::block_size_y + PADDING, Config::block_size_x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.n_elements();
        __shared__ input_value_t shared_mem[SHARED_SIZE];

        const auto gid = noa::cuda::details::global_indices<i32, N, Config>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_2d = input[gid.template pop_back<2>()];

        // Load shared memory. Loop to account for the halo.
        for (i32 ly = tid[0], gy = gid[N - 2]; ly < SHARED_SHAPE[0]; ly += Config::block_size_y, gy += Config::block_size_y)
            for (i32 lx = tid[1], gx = gid[N - 1]; lx < SHARED_SHAPE[1]; lx += Config::block_size_x, gx += Config::block_size_x)
                median_filter_2d_load_to_shared<BORDER_REFLECT, HALO>(
                    input_2d, shared_mem + ly * SHARED_SHAPE[1] + lx,
                    shape[0], gy - HALO, shape[1], gx - HALO);
        noa::cuda::details::block_synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[N - 2] < shape[0] and gid[N - 1] < shape[1]) {
            constexpr i32 ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            input_value_t v[ACTIVE];
            for (i32 count = 0, y = 0; y < WINDOW_SIZE; ++y)
                for (i32 x = 0; count < ACTIVE and x < WINDOW_SIZE; ++x, ++count)
                    v[count] = shared_mem[(tid[0] + y) * SHARED_SHAPE[1] + tid[1] + x];

            order_(v, ACTIVE);
            i32 length = ACTIVE;
            for (i32 k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y) coordinates in the 2D window.
                // Then offset these coordinates to the current thread.
                const i32 coord_y = k / WINDOW_SIZE;
                const i32 coord_x = k - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_y + tid[0]) * SHARED_SHAPE[1] + coord_x + tid[1]];

                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (i32 k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            output(gid) = static_cast<output_value_t>(v[length / 2]);
        }
    }

    // Loads the (gx,gy,gz) element according to the BORDER_MODE.
    // input:        device memory. Should point to the beginning of the batch.
    // stride_z/y/x: strides
    // s_mem:        shared memory. Should point to the current element.
    // shape_z/y/x:  number of logical elements in z/y/x.
    // gz/y/x:       index of the current element in z/y/x. If out of bound, add padding according to the BORDER_MODE.
    template<bool BORDER_REFLECT, i32 HALO, typename Accessor3d, typename T>
    __device__ void median_filter_3d_load_to_shared(
        const Accessor3d& input, T* s_mem,
        i32 shape_z, i32 gz, i32 shape_y, i32 gy, i32 shape_x, i32 gx
    ) {
        if constexpr (BORDER_REFLECT) {
            if (gx < 0) {
                gx *= -1;
            } else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return;
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0) {
                gy *= -1;
            } else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            if (gz < 0) {
                gz *= -1;
            } else if (gz >= shape_z) {
                if (gz >= shape_z + HALO)
                    return;
                gz = 2 * (shape_z - 1) - gz;
            }
            *s_mem = input(gz, gy, gx);

        } else {
            if (gx < 0 or gx >= shape_x or
                gy < 0 or gy >= shape_y or
                gz < 0 or gz >= shape_z)
                *s_mem = T{0};
            else
                *s_mem = input(gz, gy, gx);
        }
    }

    // The launch config and block size is like median_filter_1d_.
    template<typename Config, usize N, typename Input, typename Output, bool BORDER_REFLECT, i32 WINDOW_SIZE>
    __global__ __launch_bounds__(Config::block_size_x * Config::block_size_y)
    void median_filter_3d(
        Input input,
        Output output,
        Shape<i32, 3> shape,
        Vec<u32, (N <= 2 ? N - 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        static_assert(is_odd(WINDOW_SIZE));
        using input_value_t = Input::mutable_value_type;
        using output_value_t = Output::value_type;

        // The shared memory is the shape of the block + the padding all 3 dimensions.
        // For the largest supported window (5), this goes up to 20*20*5=2205 elements.
        constexpr i32 TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE * WINDOW_SIZE;
        constexpr i32 PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr i32 HALO = PADDING / 2;
        constexpr auto SHARED_SHAPE = Shape<i32, 3>{WINDOW_SIZE, Config::block_size_y + PADDING, Config::block_size_x + PADDING};
        constexpr auto SHARED_SIZE = SHARED_SHAPE.n_elements();
        __shared__ input_value_t shared_mem[SHARED_SIZE];

        const auto gid = noa::cuda::details::global_indices<i32, N, Config>(grid_fused_shape_in_x_unbatched, grid_outer_offset);
        const auto tid = noa::cuda::details::thread_indices<i32, 2>();
        const auto input_3d = input[gid.template pop_back<3>()];

        // Load shared memory.
        // Each thread processes at least WINDOW_SIZE elements (the z dimension).
        for (i32 lz = 0, gz = gid[N - 3]; lz < SHARED_SHAPE[0]; ++lz, ++gz)
            for (i32 ly = tid[0], gy = gid[N - 2]; ly < SHARED_SHAPE[1]; ly += Config::block_size_y, gy += Config::block_size_y)
                for (i32 lx = tid[1], gx = gid[N - 1]; lx < SHARED_SHAPE[2]; lx += Config::block_size_x, gx += Config::block_size_x)
                    median_filter_3d_load_to_shared<BORDER_REFLECT, HALO>(
                        input_3d,
                        shared_mem + (lz * SHARED_SHAPE[1] + ly) * SHARED_SHAPE[2] + lx,
                        shape[0], gz - HALO, shape[1], gy - HALO, shape[2], gx - HALO);
        noa::cuda::details::block_synchronize();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid[N - 2] < shape[1] and gid[N - 1] < shape[2]) {
            constexpr i32 ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            input_value_t v[ACTIVE];
            for (i32 count = 0, z = 0; z < WINDOW_SIZE; ++z)
                for (i32 y = 0; count < ACTIVE and y < WINDOW_SIZE; ++y)
                    for (i32 x = 0; count < ACTIVE and x < WINDOW_SIZE; ++x, ++count)
                        v[count] = shared_mem[(z * SHARED_SHAPE[1] + tid[0] + y) * SHARED_SHAPE[2] + tid[1] + x];

            order_(v, ACTIVE);
            i32 length = ACTIVE;
            for (i32 k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y,z) coordinates in the 3D window.
                // Then offset these coordinates to the current thread.
                const i32 coord_z = k / (WINDOW_SIZE * WINDOW_SIZE);
                const i32 tmp = k - coord_z * WINDOW_SIZE * WINDOW_SIZE;
                const i32 coord_y = tmp / WINDOW_SIZE;
                const i32 coord_x = tmp - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_z * SHARED_SHAPE[1] + coord_y + tid[0]) * SHARED_SHAPE[2] + coord_x + tid[1]];
                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (i32 k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            output(gid) = static_cast<output_value_t>(v[length / 2]);
        }
    }
}

namespace noa::signal::cuda {
    template<typename T, typename U, usize N>
    void median_filter_1d(
        const T* input, const Strides<i32, N>& input_strides,
        U* output, const Strides<i32, N>& output_strides,
        const Shape<i32, N>& shape, Border border_mode, isize window_size, noa::cuda::Stream& stream
    ) {
        using Block = details::MedianFilterBlockND<N>;
        auto block_shape = Shape<i32, N>::from_value(1);
        if constexpr (N >= 2)
            block_shape[N - 2] = static_cast<i32>(Block::block_size_y);
        block_shape[N - 1] = static_cast<i32>(Block::block_size_x);
        const auto grid = noa::cuda::GridND(shape, block_shape);
        check(grid.n_launches() == 1);

        using input_t = AccessorRestrict<const T, N, i32>;
        using output_t = AccessorRestrict<U, N, i32>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        constexpr usize R = N == 1 ? 1 : 2;
        Shape<i32, R> shape_2d;
        if constexpr (N >= 2)
            shape_2d = shape.filter(N - 2, N - 1);
        else
            shape_2d = shape.filter(N - 1);

        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().pop_back();
        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = noa::cuda::LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                };
                const auto grid_offset = grid.block_offset_for_launch(z, y).template pop_front<(N <= 2 ? 3 - N : 0)>();
                switch (window_size) {
                    case 3:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 3>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 3>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 5:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 5>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 5>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 7:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 7>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 7>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 9:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 9>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 9>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 11:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 11>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 11>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 13:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 13>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 13>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 15:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 15>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 15>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 17:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 17>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 17>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 19:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 19>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 19>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 21:
                        return stream.enqueue(
                            border_mode == Border::REFLECT
                                ? details::median_filter_1d<Block, N, R, input_t, output_t, true, 21>
                                : details::median_filter_1d<Block, N, R, input_t, output_t, false, 21>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    default:
                        panic("Unsupported window size. It should be an odd number from 1 to 21, got {}", window_size);
                }
            }
        }
    }

    template<typename T, typename U, usize N>
    void median_filter_2d(
        const T* input, const Strides<i32, N>& input_strides,
        U* output, const Strides<i32, N>& output_strides,
        const Shape<i32, N>& shape, Border border_mode, isize window_size, noa::cuda::Stream& stream
    ) {
        using Block = details::MedianFilterBlockND<N>;
        auto block_shape = Shape<i32, N>::from_value(1);
        block_shape[N - 2] = static_cast<i32>(Block::block_size_y);
        block_shape[N - 1] = static_cast<i32>(Block::block_size_x);
        const auto grid = noa::cuda::GridND(shape, block_shape);
        check(grid.n_launches() == 1);

        using input_t = AccessorRestrict<const T, N, i32>;
        using output_t = AccessorRestrict<U, N, i32>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        const auto shape_2d = shape.filter(N - 2, N - 1);
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().pop_back();
        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = noa::cuda::LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                };
                const auto grid_offset = grid.block_offset_for_launch(z, y).template pop_front<(N <= 2 ? 3 - N : 0)>();
                switch (window_size) {
                    case 3:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_2d<Block, N, input_t, output_t, true, 3> :
                            details::median_filter_2d<Block, N, input_t, output_t, false, 3>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 5:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_2d<Block, N, input_t, output_t, true, 5> :
                            details::median_filter_2d<Block, N, input_t, output_t, false, 5>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 7:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_2d<Block, N, input_t, output_t, true, 7> :
                            details::median_filter_2d<Block, N, input_t, output_t, false, 7>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 9:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_2d<Block, N, input_t, output_t, true, 9> :
                            details::median_filter_2d<Block, N, input_t, output_t, false, 9>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 11:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_2d<Block, N, input_t, output_t, true, 11> :
                            details::median_filter_2d<Block, N, input_t, output_t, false, 11>,
                            config, input_accessor, output_accessor, shape_2d, grid_offset, grid_fused_shape_in_x_unbatched);
                    default:
                        panic("Unsupported window size. It should be an odd number from 1 to 11, got {}", window_size);
                }
            }
        }
    }

    template<typename T, typename U, usize N>
    void median_filter_3d(
        const T* input, const Strides<i32, N>& input_strides,
        U* output, const Strides<i32, N>& output_strides,
        const Shape<i32, N>& shape, Border border_mode, isize window_size, noa::cuda::Stream& stream
    ) {
        using Block = details::MedianFilterBlockND<N>;
        auto block_shape = Shape<i32, N>::from_value(1);
        block_shape[N - 2] = static_cast<i32>(Block::block_size_y);
        block_shape[N - 1] = static_cast<i32>(Block::block_size_x);
        const auto grid = noa::cuda::GridND(shape, block_shape);
        check(grid.n_launches() == 1);

        using input_t = AccessorRestrict<const T, N, i32>;
        using output_t = AccessorRestrict<U, N, i32>;
        const auto input_accessor = input_t(input, input_strides);
        const auto output_accessor = output_t(output, output_strides);

        const auto shape_3d = shape.filter(N - 3, N - 2, N - 1);
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().pop_back();
        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = noa::cuda::LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                    .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                };
                const auto grid_offset = grid.block_offset_for_launch(z, y).template pop_front<(N <= 2 ? 3 - N : 0)>();
                switch (window_size) {
                    case 3:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_3d<Block, N, input_t, output_t, true, 3> :
                            details::median_filter_3d<Block, N, input_t, output_t, false, 3>,
                            config, input_accessor, output_accessor, shape_3d, grid_offset, grid_fused_shape_in_x_unbatched);
                    case 5:
                        return stream.enqueue(
                            border_mode == Border::REFLECT ?
                            details::median_filter_3d<Block, N, input_t, output_t, true, 5> :
                            details::median_filter_3d<Block, N, input_t, output_t, false, 5>,
                            config, input_accessor, output_accessor, shape_3d, grid_offset, grid_fused_shape_in_x_unbatched);
                    default:
                        panic("Unsupported window size. It should be an odd number from 1 to 5, got {}", window_size);
                }
            }
        }
    }
}
