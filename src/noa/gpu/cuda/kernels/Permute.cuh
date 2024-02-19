#pragma once

#include "noa/core/math/Generic.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace noa::cuda::guts {
    // Out-of-place.
    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    // XY tile along Z becomes X'Y' (X'=Y, Y'=X) along Z' (Z'=Z)
    template<typename Config, bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0132_(
            AccessorRestrict<const T, 4, u32> input,
            AccessorRestrict<T, 4, u32> output,
            Shape2<u32> shape_yx, u32 blocks_x
    ) {
        using Config::tile_size;
        __shared__ T tile[tile_size][tile_size + 1]; // +1 so that elements in a column map to different banks.

        const auto input_2d = input[blockIdx.z][blockIdx.y];
        const auto output_2d = output[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_yx[1] and gy < shape_yx[0])) // x could be checked earlier
                tile[tid[0] + repeat][tid[1]] = input_2d(gy, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // Y->X', X->Y'
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_yx[0] and gy < shape_yx[1]))
                output_2d(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place permute the XY slices one at a time.
    template<typename Config, bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0132_inplace_(Accessor<T, 4, u32> output, u32 size, u32 blocks_x) {
        using Config::tile_size;
        __shared__ T tile_src[tile_size][tile_size + 1];
        __shared__ T tile_dst[tile_size][tile_size + 1];

        const auto output_2d = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index;

        if (offset[0] > offset[1]) { // lower triangle
            const auto src_gid = offset + tid;
            const auto dst_gid = offset.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 gy = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < size and gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, src_gid[1]);

                const u32 dy = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < size and dy < size))
                    tile_dst[tid[0] + repeat][tid[1]] = output_2d(dy, dst_gid[1]);
            }

            block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 dy = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < size and dy < size))
                    output_2d(dy, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const u32 gy = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < size and gy < size))
                    output_2d(gy, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const auto gid = offset + tid;

            // Read tile to shared memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 gy = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < size and gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, gid[1]);
            }

            block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 gy = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < size and gy < size))
                    output_2d(gy, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }

    // Out-of-place.
    // Transpose 0213 is a specific case: the innermost dimension is unchanged,
    // which makes everything much simpler. Only the last two dimensions are swapped:
    //  - input_strides[1]->output_strides[2]
    //  - input_strides[2]->output_strides[1]
    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0213_(
            AccessorRestrict<const T, 4, u32> input,
            AccessorRestrict<T, 4, u32> output_swapped,
            Shape2<u32> shape_yx, u32 blocks_x
    ) {
        using Config::tile_size;
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> gid = tile_size * index + tid;
        if (not IsMultipleOfTile and gid[1] >= shape_yx[1])
            return;

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_ = output_swapped[blockIdx.z][blockIdx.y];

        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = gid[0] + repeat;
            if (IsMultipleOfTile or gy < shape_yx[0])
                output_(gy, gid[1]) = input_(gy, gid[1]);
        }
    }

    // In-place.
    // This is simply swapping the Y with the X, such as swap(o[z][y][x], o[y][z][x]).
    // Only process one triangle, plus the diagonal. The other blocks are idle...
    // The shared memory simply acts as a per thread buffer.
    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0213_inplace_(Accessor<T, 4, u32> output, Shape2<u32> shape, u32 blocks_x) {
        using Config::tile_size;
        __shared__ T tile[Config::block_size_y][Config::block_size_x];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            tile_size * index[0] + tid[0],
                            tile_size * index[1] + tid[1]};
        if (gid[3] >= shape[1])
            return;

        const auto output_ = output[gid[0]];
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = gid[2] + repeat;
            if (gid[1] > gy) // process one triangle + diagonal
                continue;

            if (IsMultipleOfTile or gy < shape[0]) {
                T& src = output_(gid[1], gy, gid[3]);
                T& dst = output_(gy, gid[1], gid[3]); // permutation 1 <-> 2
                tile[tid[0]][tid[1]] = dst;
                dst = src;
                src = tile[tid[0]][tid[1]];
            }
        }
    }

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    // The XZ tile along Y becomes X'Y' (X'=Z, Y'=X) along Z' (Z'=Y)
    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0231_(
            AccessorRestrict<const T, 4, u32> input_swapped,
            AccessorRestrict<T, 4, u32> output,
            Shape2<u32> shape_zx, u32 blocks_x
    ) {
        using Config::tile_size;
        __shared__ T tile[tile_size][tile_size + 1];

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_ = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index; // ZX

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gz = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_zx[1] and gz < shape_zx[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Y'X'
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_zx[0] and gy < shape_zx[1]))
                output_(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    // The XY tile along Z becomes X'Z' (X'=Y, Z'=X) along Y' (Y'=Z)
    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0312_(
            AccessorRestrict<const T, 4, u32> input,
            AccessorRestrict<T, 4, u32> output_swapped,
            Shape2<u32> shape_yx /* YX */ , u32 blocks_x
    ) {
        using Config::tile_size;
        __shared__ T tile[tile_size][tile_size + 1];

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gy = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_yx[1] and gy < shape_yx[0]))
                tile[tid[0] + repeat][tid[1]] = input_(gy, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid;
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            const u32 gz = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_yx[0] and gz < shape_yx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0321_(
            AccessorRestrict<const T, 4, u32> input_swapped,
            AccessorRestrict<T, 4, u32> output_swapped,
            Shape2<u32> shape_zx, u32 blocks_x
    ) {
        using Config::tile_size;
        __shared__ T tile[tile_size][tile_size + 1];

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index; // ZX

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            u32 gz = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_zx[1] and gz < shape_zx[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
            u32 gz = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_zx[0] and gz < shape_zx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    template<typename Config, typename T, bool IsMultipleOfTile>
    __global__ __launch_bounds__(Config::block_size)
    void permute_0321_inplace_(Accessor<T, 4, u32> output_swapped, u32 shape, u32 blocks_x) {
        using Config::tile_size;
        __shared__ T tile_src[tile_size][tile_size + 1];
        __shared__ T tile_dst[tile_size][tile_size + 1];

        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = tile_size * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const auto src_gid = offset + tid; // ZX
            const auto dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 sz = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < shape and sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(sz, src_gid[1]);

                const u32 dz = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < shape and dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output_swapped_(dz, dst_gid[1]);
            }

            block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 dz = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < shape and dz < shape))
                    output_swapped_(dz, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const u32 sz = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < shape and sz < shape))
                    output_swapped_(sz, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const auto gid = offset + tid; // ZX

            // Read tile to shared memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 gz = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < shape and gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(gz, gid[1]);
            }

            block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < tile_size; repeat += Config::block_size_y) {
                const u32 gz = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < shape and gz < shape))
                    output_swapped_(gz, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }

    // Since all axes are permuted, in-place permute cannot easily be expressed as a 2D transposition
    // along a COMMON plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}
