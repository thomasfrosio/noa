#include "noa/core/math/Generic.hpp"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Permute.h"
#include "noa/gpu/cuda/utils/Block.cuh"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace {
    using namespace ::noa;

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr u32 TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // XY tile along Z becomes X'Y' (X'=Y, Y'=X) along Z' (Z'=Z)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0132_(AccessorRestrict<const T, 4, u32> input,
                       AccessorRestrict<T, 4, u32> output,
                       Shape2<u32> shape_yx, u32 blocks_x) {
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1]; // +1 so that elements in a column map to different banks.
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        const auto input_2d = input[blockIdx.z][blockIdx.y];
        const auto output_2d = output[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_DIM * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gy = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape_yx[1] && gy < shape_yx[0])) // x could be checked earlier
                tile[tid[0] + repeat][tid[1]] = input_2d(gy, old_gid[1]);
        }

        noa::cuda::utils::block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // Y->X', X->Y'
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gy = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape_yx[0] && gy < shape_yx[1]))
                output_2d(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place permute the XY slices one at a time.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0132_inplace_(Accessor<T, 4, u32> output, u32 size, u32 blocks_x) {
        using uninit_t = cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        const auto output_2d = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_DIM * index;

        if (offset[0] > offset[1]) { // lower triangle
            const auto src_gid = offset + tid;
            const auto dst_gid = offset.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < size && gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, src_gid[1]);

                const u32 dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < size && dy < size))
                    tile_dst[tid[0] + repeat][tid[1]] = output_2d(dy, dst_gid[1]);
            }

            noa::cuda::utils::block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < size && dy < size))
                    output_2d(dy, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const u32 gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < size && gy < size))
                    output_2d(gy, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const auto gid = offset + tid;

            // Read tile to shared memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < size && gy < size))
                    tile_src[tid[0] + repeat][tid[1]] = output_2d(gy, gid[1]);
            }

            noa::cuda::utils::block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < size && gy < size))
                    output_2d(gy, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute_0132(const Shared<T[]>& input, const Strides4<i64>& input_strides,
                      const Shared<T[]>& output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile = noa::all((shape_2d % TILE_DIM) == 0);

        const u32 blocks_y = noa::math::divide_up(shape_2d[0], TILE_DIM);
        const u32 blocks_x = noa::math::divide_up(shape_2d[1], TILE_DIM);
        const dim3 blocks(blocks_y * blocks_x, u_shape[1], u_shape[0]);
        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input.get(), input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output.get(), output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue("memory::permute0132", permute_0132_<T, true>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, shape_2d, blocks_x);
        } else {
            stream.enqueue("memory::permute0132", permute_0132_<T, false>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, shape_2d, blocks_x);
        }
        stream.attach(input, output);
    }

    template<typename T>
    void permute_0132_inplace(
            const Shared<T[]>& output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place permutation, shape[2] should be equal to shape[3]. Got shape:{}", shape);

        const auto u_shape = shape.as_safe<u32>();
        const bool is_multiple_tile = (u_shape[3] % TILE_DIM) == 0;

        const u32 blocks_x = noa::math::divide_up(u_shape[3], TILE_DIM); // blocks_y == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[1], shape[0]); // about less than half will be idle blocks.
        const auto accessor = Accessor<T, 4, u32>(output.get(), output_strides.as_safe<u32>());

        if (is_multiple_tile) {
            stream.enqueue(
                    "memory::permute0132_inplace", permute_0132_inplace_<T, true>, {blocks, BLOCK_SIZE},
                    accessor, u_shape[3], blocks_x);
        } else {
            stream.enqueue(
                    "memory::permute0132_inplace", permute_0132_inplace_<T, false>, {blocks, BLOCK_SIZE},
                    accessor, u_shape[3], blocks_x);
        }
        stream.attach(output);
    }

    #define NOA_INSTANTIATE_TRANSPOSE_(T)           \
    template void permute_0132<T>(                  \
        const Shared<T[]>&, const Strides4<i64>&,   \
        const Shared<T[]>&, const Strides4<i64>&,   \
        const Shape4<i64>&, Stream&);               \
    template void permute_0132_inplace<T>(          \
        const Shared<T[]>&, const Strides4<i64>&,   \
        const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_TRANSPOSE_(bool);
    NOA_INSTANTIATE_TRANSPOSE_(i8);
    NOA_INSTANTIATE_TRANSPOSE_(i16);
    NOA_INSTANTIATE_TRANSPOSE_(i32);
    NOA_INSTANTIATE_TRANSPOSE_(i64);
    NOA_INSTANTIATE_TRANSPOSE_(u8);
    NOA_INSTANTIATE_TRANSPOSE_(u16);
    NOA_INSTANTIATE_TRANSPOSE_(u32);
    NOA_INSTANTIATE_TRANSPOSE_(u64);
    NOA_INSTANTIATE_TRANSPOSE_(f16);
    NOA_INSTANTIATE_TRANSPOSE_(f32);
    NOA_INSTANTIATE_TRANSPOSE_(f64);
    NOA_INSTANTIATE_TRANSPOSE_(c16);
    NOA_INSTANTIATE_TRANSPOSE_(c32);
    NOA_INSTANTIATE_TRANSPOSE_(c64);
}
