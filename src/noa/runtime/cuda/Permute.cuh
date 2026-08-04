#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Math.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/Stream.hpp"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace noa::cuda::details {
    struct PermuteConfig {
        static constexpr u32 tile_size = Constant::WARP_SIZE;
        static constexpr u32 block_size = 256;
        static constexpr u32 block_size_x = tile_size;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_work_size_x = tile_size;
        static constexpr u32 block_work_size_y = tile_size;
        static constexpr u32 block_work_size_z = 1;
        static constexpr u32 block_ndim = 2;
    };

    // Out-of-place permutation of the two rightmost axes (height and width).
    // This is used to permute any axis to the rightmost and the rightmost/innermost axis to the second rightmost.
    template<bool IsMultipleOfTile, typename T, usize N>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_hw_(
        AccessorRestrict<const T, N, u32> input,
        AccessorRestrict<T, N, u32> output,
        Shape<u32, 2> shape_hw,
        Vec<u32, (N == 2 ? 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ Uninitialized<T> tile_[TILE_SIZE][TILE_SIZE + 1]; // +1 so that elements in a column map to different banks
        T(& tile)[TILE_SIZE][TILE_SIZE + 1] = *reinterpret_cast<T(*)[TILE_SIZE][TILE_SIZE + 1]>(&tile_);

        const auto [gid_batches, gid_hw] = global_indices<u32, N, PermuteConfig, false>(
            grid_fused_shape_in_x_unbatched, grid_outer_offset).template split<N - 2>();

        const auto tid = thread_indices<u32, 2>();

        // Read tile to shared memory.
        const auto input_2d = input[gid_batches];
        const auto input_gid = gid_hw + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = input_gid[0] + repeat;
            if (IsMultipleOfTile or (input_gid[1] < shape_hw[1] and gy < shape_hw[0])) // x could be checked earlier
                tile[tid[0] + repeat][tid[1]] = input_2d(gy, input_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto output_hw = output[gid_batches];
        const auto output_gid = gid_hw.flip() + tid; // Y->X', X->Y'
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = output_gid[0] + repeat;
            if (IsMultipleOfTile or (output_gid[1] < shape_hw[0] and gy < shape_hw[1]))
                output_hw(gy, output_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    template<bool IsMultipleOfTile, typename T, usize N>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_hw_inplace_(
        AccessorRestrict<T, N, u32> output, u32 size,
        Vec<u32, (N == 2 ? 1 : 2)> grid_outer_offset, // ((z,)y)
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ Uninitialized<T> tile_src_[TILE_SIZE][TILE_SIZE + 1];
        __shared__ Uninitialized<T> tile_dst_[TILE_SIZE][TILE_SIZE + 1];
        T(& tile_src)[TILE_SIZE][TILE_SIZE + 1] = *reinterpret_cast<T(*)[TILE_SIZE][TILE_SIZE + 1]>(&tile_src_);
        T(& tile_dst)[TILE_SIZE][TILE_SIZE + 1] = *reinterpret_cast<T(*)[TILE_SIZE][TILE_SIZE + 1]>(&tile_dst_);

        const auto [gid_batches, gid_hw] = global_indices<u32, N, PermuteConfig, false>(
            grid_fused_shape_in_x_unbatched, grid_outer_offset).template split<N - 2>();

        const auto tid = thread_indices<u32, 2>();
        const auto output_hw = output[gid_batches];

        if (gid_hw[0] > gid_hw[1]) { // lower triangle
            const auto input_gid = gid_hw + tid;
            const auto output_gid = gid_hw.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 iy = input_gid[0] + repeat;
                if (IsMultipleOfTile or (input_gid[1] < size and iy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_hw(iy, input_gid[1]);

                const u32 oy = output_gid[0] + repeat;
                if (IsMultipleOfTile or (output_gid[1] < size and oy < size))
                    tile_dst[tid[0] + repeat][tid[1]] = output_hw(oy, output_gid[1]);
            }

            block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 oy = output_gid[0] + repeat;
                if (IsMultipleOfTile or (output_gid[1] < size and oy < size))
                    output_hw(oy, output_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const u32 iy = input_gid[0] + repeat;
                if (IsMultipleOfTile or (input_gid[1] < size and iy < size))
                    output_hw(iy, input_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (gid_hw[0] == gid_hw[1]) { // diagonal
            const auto gid = gid_hw + tid;

            // Read tile to shared memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 gy = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < size and gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_hw(gy, gid[1]);
            }

            block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 gy = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < size and gy < size))
                    output_hw(gy, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda {
    template<typename T, usize N> requires (N >= 2)
    void permute_copy_hw(
        const T* input, const Strides<isize, N>& input_strides, const Shape<isize, N>& input_shape,
        T* output, const Strides<isize, N>& output_strides, Stream& stream
    ) {
        const auto shape_2d = input_shape.filter(N - 2, N - 1).template as_safe<u32>();
        const bool is_inplace = input == output;
        const bool are_multiple_tile =
            noa::is_multiple_of(shape_2d[0], details::PermuteConfig::tile_size) and
            noa::is_multiple_of(shape_2d[1], details::PermuteConfig::tile_size);

        const auto input_accessor = AccessorRestrict<const T, N, u32>(input, input_strides.template as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, N, u32>(output, output_strides.template as_safe<u32>());

        auto block_shape = Shape<isize, N>::from_value(1);
        block_shape[N - 2] = details::PermuteConfig::tile_size;
        block_shape[N - 1] = details::PermuteConfig::tile_size;
        const auto grid = GridND(input_shape, block_shape);
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        check(grid.n_launches() == 1);

        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                const auto config = LaunchConfig{
                    .n_blocks = grid.dim3_shape_for_launch(z, y, 0), // for inplace about less than half will be no-op blocks...
                    .n_threads = dim3(details::PermuteConfig::block_size_x, details::PermuteConfig::block_size_y, 1),
                };

                const auto grid_outer_offset = grid.block_offset_for_launch(z, y).template pop_front<(N == 2 ? 1 : 0)>();
                if (are_multiple_tile) {
                    if (is_inplace) {
                        stream.enqueue(
                            details::permute_hw_inplace_<true, T, N>,
                            config, output_accessor, shape_2d[1], grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                    } else {
                        stream.enqueue(
                            details::permute_hw_<true, T, N>, config,
                            input_accessor, output_accessor, shape_2d, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                    }
                } else {
                    if (is_inplace) {
                        stream.enqueue(
                            details::permute_hw_inplace_<false, T, N>,
                            config, output_accessor, shape_2d[1], grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                    } else {
                        stream.enqueue(
                            details::permute_hw_<false, T, N>, config,
                            input_accessor, output_accessor, shape_2d, grid_outer_offset, grid_fused_shape_in_x_unbatched
                        );
                    }
                }
            }
        }
    }
}
