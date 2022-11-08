#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/utils/Traits.h"
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
    using namespace ::noa::cuda;

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr uint32_t TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // XY tile along Z becomes X'Y' (X'=Y, Y'=X) along Z' (Z'=Z)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0132_(AccessorRestrict<const T, 4, uint32_t> input,
                      AccessorRestrict<T, 4, uint32_t> output,
                      uint2_t shape /* YX */, uint32_t blocks_x) {
        using uninit_t = cuda::utils::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1]; // +1 so that elements in a column map to different banks.
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_ = output[blockIdx.z][blockIdx.y];

        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index;

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint32_t gy = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gy < shape[0])) // x could be checked before
                tile[tid[0] + repeat][tid[1]] = input_(gy, old_gid[1]);
        }

        utils::block::synchronize();

        // Write permuted tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // Y->X', X->Y'
        for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint32_t gy = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gy < shape[1]))
                output_(gy, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place permute the XY slices one at a time.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0132_inplace_(Accessor<T, 4, uint32_t> output, uint32_t shape, uint32_t blocks_x) {
        using uninit_t = cuda::utils::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        const auto output_ = output[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index;

        if (offset[0] > offset[1]) { // lower triangle
            const uint2_t src_gid = offset + tid;
            const uint2_t dst_gid = offset.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && gy < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_(gy, src_gid[1]);

                const uint32_t dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dy < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output_(dy, dst_gid[1]);
            }

            utils::block::synchronize();

            // Write permuted tiles to global memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dy < shape))
                    output_(dy, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const uint32_t gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && gy < shape))
                    output_(gy, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const uint2_t gid = offset + tid;

            // Read tile to shared memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gy < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_(gy, gid[1]);
            }

            utils::block::synchronize();

            // Write permuted tile to global memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gy < shape))
                    output_(gy, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute0132(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides,
                     dim4_t shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const auto uint_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint32_t blocks_y = math::divideUp(uint_shape[0], TILE_DIM);
        const uint32_t blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const dim3 blocks(blocks_y * blocks_x, shape[1], shape[0]);
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        if (are_multiple_tile) {
            stream.enqueue("memory::permute0132", permute0132_<T, true>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::permute0132", permute0132_<T, false>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, uint_shape, blocks_x);
        }
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void permute0132(const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place permutation, shape[2] should be equal to shape[3]. Got shape:{}", shape);

        const auto uint_shape = safe_cast<uint32_t>(shape[3]);
        const bool is_multiple_tile = (uint_shape % TILE_DIM) == 0;

        const uint32_t blocks_x = math::divideUp(uint_shape, TILE_DIM); // blocks_y == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[1], shape[0]); // about less than half will be idle blocks.
        const Accessor<T, 4, uint32_t> accessor(output.get(), safe_cast<uint4_t>(output_strides));

        if (is_multiple_tile) {
            stream.enqueue(
                    "memory::permute0132_inplace", permute0132_inplace_<T, true>, {blocks, BLOCK_SIZE},
                    accessor, uint_shape, blocks_x);
        } else {
            stream.enqueue(
                    "memory::permute0132_inplace", permute0132_inplace_<T, false>, {blocks, BLOCK_SIZE},
                    accessor, uint_shape, blocks_x);
        }
        stream.attach(output);
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                                                   \
template void noa::cuda::memory::details::permute0132<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
template void noa::cuda::memory::details::inplace::permute0132<T>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

NOA_INSTANTIATE_TRANSPOSE_(bool);
NOA_INSTANTIATE_TRANSPOSE_(int8_t);
NOA_INSTANTIATE_TRANSPOSE_(int16_t);
NOA_INSTANTIATE_TRANSPOSE_(int32_t);
NOA_INSTANTIATE_TRANSPOSE_(int64_t);
NOA_INSTANTIATE_TRANSPOSE_(uint8_t);
NOA_INSTANTIATE_TRANSPOSE_(uint16_t);
NOA_INSTANTIATE_TRANSPOSE_(uint32_t);
NOA_INSTANTIATE_TRANSPOSE_(uint64_t);
NOA_INSTANTIATE_TRANSPOSE_(half_t);
NOA_INSTANTIATE_TRANSPOSE_(float);
NOA_INSTANTIATE_TRANSPOSE_(double);
NOA_INSTANTIATE_TRANSPOSE_(chalf_t);
NOA_INSTANTIATE_TRANSPOSE_(cfloat_t);
NOA_INSTANTIATE_TRANSPOSE_(cdouble_t);
