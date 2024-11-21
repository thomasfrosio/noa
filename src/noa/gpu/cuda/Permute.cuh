#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Config.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Copy.cuh"
#include "noa/gpu/cuda/Error.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace noa::cuda::guts {
    struct PermuteConfig {
        static constexpr u32 tile_size = Constant::WARP_SIZE;
        static constexpr u32 block_size = 256;
        static constexpr u32 block_size_x = tile_size;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    // Out-of-place.
    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    // XY tile along Z becomes X'Y' (X'=Y, Y'=X) along Z' (Z'=Z)
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0132_(
        AccessorRestrict<const T, 4, u32> input,
        AccessorRestrict<T, 4, u32> output,
        Shape2<u32> shape_yx, u32 blocks_x
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile[TILE_SIZE][TILE_SIZE + 1]; // +1 so that elements in a column map to different banks.

        const auto input_2d = input[blockIdx.z][blockIdx.y];
        const auto output_2d = output[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_yx[1] and gy < shape_yx[0])) // x could be checked earlier
                tile[tid[0] + repeat][tid[1]] = input_2d(gy, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // Y->X', X->Y'
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_yx[0] and gy < shape_yx[1]))
                output_2d(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place permute the XY slices one at a time.
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0132_inplace_(Accessor<T, 4, u32> output, u32 size, u32 blocks_x) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile_src[TILE_SIZE][TILE_SIZE + 1];
        __shared__ T tile_dst[TILE_SIZE][TILE_SIZE + 1];

        const auto output_2d = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index;

        if (offset[0] > offset[1]) { // lower triangle
            const auto src_gid = offset + tid;
            const auto dst_gid = offset.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 gy = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < size and gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, src_gid[1]);

                const u32 dy = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < size and dy < size))
                    tile_dst[tid[0] + repeat][tid[1]] = output_2d(dy, dst_gid[1]);
            }

            block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
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
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 gy = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < size and gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, gid[1]);
            }

            block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
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
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0213_(
        AccessorRestrict<const T, 4, u32> input,
        AccessorRestrict<T, 4, u32> output_swapped,
        Shape2<u32> shape_yx, u32 blocks_x
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> gid = TILE_SIZE * index + tid;
        if (not IsMultipleOfTile and gid[1] >= shape_yx[1])
            return;

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_ = output_swapped[blockIdx.z][blockIdx.y];

        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = gid[0] + repeat;
            if (IsMultipleOfTile or gy < shape_yx[0])
                output_(gy, gid[1]) = input_(gy, gid[1]);
        }
    }

    // In-place.
    // This is simply swapping the Y with the X, such as swap(o[z][y][x], o[y][z][x]).
    // Only process one triangle, plus the diagonal. The other blocks are idle...
    // The shared memory simply acts as a per thread buffer.
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0213_inplace_(Accessor<T, 4, u32> output, Shape2<u32> shape, u32 blocks_x) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile[PermuteConfig::block_size_y][PermuteConfig::block_size_x];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            TILE_SIZE * index[0] + tid[0],
                            TILE_SIZE * index[1] + tid[1]};
        if (gid[3] >= shape[1])
            return;

        const auto output_ = output[gid[0]];
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
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
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0231_(
        AccessorRestrict<const T, 4, u32> input_swapped,
        AccessorRestrict<T, 4, u32> output,
        Shape2<u32> shape_zx, u32 blocks_x
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_ = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index; // ZX

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gz = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_zx[1] and gz < shape_zx[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Y'X'
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_zx[0] and gy < shape_zx[1]))
                output_(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    // The XY tile along Z becomes X'Z' (X'=Y, Z'=X) along Y' (Y'=Z)
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0312_(
        AccessorRestrict<const T, 4, u32> input,
        AccessorRestrict<T, 4, u32> output_swapped,
        Shape2<u32> shape_yx /* YX */ , u32 blocks_x
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gy = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_yx[1] and gy < shape_yx[0]))
                tile[tid[0] + repeat][tid[1]] = input_(gy, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            const u32 gz = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_yx[0] and gz < shape_yx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0321_(
        AccessorRestrict<const T, 4, u32> input_swapped,
        AccessorRestrict<T, 4, u32> output_swapped,
        Shape2<u32> shape_zx, u32 blocks_x
    ) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index; // ZX

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            u32 gz = old_gid[0] + repeat;
            if (IsMultipleOfTile or (old_gid[1] < shape_zx[1] and gz < shape_zx[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
            u32 gz = new_gid[0] + repeat;
            if (IsMultipleOfTile or (new_gid[1] < shape_zx[0] and gz < shape_zx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    template<bool IsMultipleOfTile, typename T>
    __global__ __launch_bounds__(PermuteConfig::block_size)
    void permute_0321_inplace_(Accessor<T, 4, u32> output_swapped, u32 shape, u32 blocks_x) {
        constexpr u32 TILE_SIZE = PermuteConfig::tile_size;
        __shared__ T tile_src[TILE_SIZE][TILE_SIZE + 1];
        __shared__ T tile_dst[TILE_SIZE][TILE_SIZE + 1];

        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = ni::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_SIZE * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const auto src_gid = offset + tid; // ZX
            const auto dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 sz = src_gid[0] + repeat;
                if (IsMultipleOfTile or (src_gid[1] < shape and sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(sz, src_gid[1]);

                const u32 dz = dst_gid[0] + repeat;
                if (IsMultipleOfTile or (dst_gid[1] < shape and dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output_swapped_(dz, dst_gid[1]);
            }

            block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
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
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
                const u32 gz = gid[0] + repeat;
                if (IsMultipleOfTile or (gid[1] < shape and gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(gz, gid[1]);
            }

            block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < TILE_SIZE; repeat += PermuteConfig::block_size_y) {
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

namespace noa::cuda::guts {
    template<typename T>
    void permute_0132(
        const T* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
            is_multiple_of(shape_2d[0], PermuteConfig::tile_size) and
            is_multiple_of(shape_2d[1], PermuteConfig::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };
        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue(permute_0132_<true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0132_<false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }

    template<typename T>
    void permute_0213(
        const T* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
            is_multiple_of(shape_2d[0], PermuteConfig::tile_size) and
            is_multiple_of(shape_2d[1], PermuteConfig::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0213_<true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0213_<false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }


    template<typename T>
    void permute_0312(
        const T* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile =
            is_multiple_of(shape_2d[0], PermuteConfig::tile_size) and
            is_multiple_of(shape_2d[1], PermuteConfig::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, u_shape[1], u_shape[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0312_<true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0312_<false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }

    template<typename T>
    void permute_0231(
        const T* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(1, 3);
        const bool are_multiple_tile =
            is_multiple_of(shape_2d[0], PermuteConfig::tile_size) and
            is_multiple_of(shape_2d[1], PermuteConfig::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, u_shape[2], u_shape[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>().filter(0, 2, 1, 3)); // Y -> Z'
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue(permute_0231_<true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0231_<false, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        }
    }
    template<typename T>
    void permute_0321(
        const T* input, const Strides4<i64>& input_strides,
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(1, 3);
        const bool are_multiple_tile =
            is_multiple_of(shape_2d[0], PermuteConfig::tile_size) and
            is_multiple_of(shape_2d[1], PermuteConfig::tile_size);

        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, u_shape[2], u_shape[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>().filter(0, 2, 1, 3));
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        if (are_multiple_tile) {
            stream.enqueue(permute_0321_<true, T>, launch_config,
                           input_accessor, output_accessor, shape_2d, n_blocks_x);
        } else {
            stream.enqueue(permute_0321_<false, T>, launch_config,
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

        const auto shape_u32 = shape.as_safe<u32>();
        const bool is_multiple_tile = is_multiple_of(shape_u32[3], PermuteConfig::tile_size);
        const auto accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        const u32 n_blocks_x = divide_up(shape_u32[3], PermuteConfig::tile_size); // blocks_y == blocks_x
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_x, shape_u32[1], shape_u32[0]), // about less than half will be idle blocks...
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        if (is_multiple_tile)
            stream.enqueue(permute_0132_inplace_<true, T>, launch_config, accessor, shape_u32[3], n_blocks_x);
        else
            stream.enqueue(permute_0132_inplace_<false, T>, launch_config, accessor, shape_u32[3], n_blocks_x);
    }

    template<typename T>
    void permute_0213_inplace(
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        check(shape[1] == shape[2],
              "For a \"0213\" in-place permutation, shape[1] should be equal to shape[2]. Got shape={}", shape);

        const auto shape_u32 = shape.as_safe<u32>();
        const auto shape_2d = shape_u32.filter(2, 3);
        const bool is_multiple_tile = is_multiple_of(shape_2d[0], PermuteConfig::tile_size);
        const auto accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        const u32 n_blocks_x = divide_up(shape_2d[1], PermuteConfig::tile_size);
        const u32 n_blocks_y = divide_up(shape_2d[0], PermuteConfig::tile_size);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_y, shape_u32[1], shape_u32[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        if (is_multiple_tile)
            stream.enqueue(permute_0213_inplace_<true, T>, launch_config, accessor, shape_2d, n_blocks_x);
        else
            stream.enqueue(permute_0213_inplace_<false, T>, launch_config, accessor, shape_2d, n_blocks_x);
    }

    template<typename T>
    void permute_0321_inplace(
        T* output, const Strides4<i64>& output_strides,
        const Shape4<i64>& shape, Stream& stream
    ) {
        check(shape[1] == shape[3],
              "For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got shape={}", shape);

        const auto shape_u32 = shape.as_safe<u32>();
        const bool is_multiple_tile = is_multiple_of(shape_u32[1], PermuteConfig::tile_size);
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>().filter(0, 2, 1, 3));

        const u32 n_blocks_x = divide_up(shape_u32[1], PermuteConfig::tile_size); // blocks_z == blocks_x
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x * n_blocks_x, shape_u32[2], shape_u32[0]),
            .n_threads = dim3(PermuteConfig::block_size_x, PermuteConfig::block_size_y, 1),
        };

        if (is_multiple_tile) {
            stream.enqueue(permute_0321_inplace_<true, T>, launch_config,
                           output_accessor, shape_u32[1], n_blocks_x);
        } else {
            stream.enqueue(permute_0321_inplace_<false, T>, launch_config,
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
        const auto idx =
                permutation[0] * 1000 +
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
                    // Expected to be much slower...
                    const auto output_shape = ni::reorder(input_shape, permutation);
                    const auto input_strides_permuted = ni::reorder(input_strides, permutation);
                    copy(input, input_strides_permuted, output, output_strides, output_shape, stream);
            }
        }
    }
}
