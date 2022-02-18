#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Transpose.h"

#include "noa/gpu/cuda/util/Block.cuh"

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
    constexpr uint TILE_DIM = 32;
    constexpr dim3 THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // XY tile along Z becomes X'Y' (X'=Y, Y'=X) along Z' (Z'=Z)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose0132_(const T* __restrict__ input, uint4_t input_stride,
                        T* __restrict__ output, uint4_t output_stride,
                        uint2_t shape /* YX */, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1]; // +1 so that elements in a column map to different banks.
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        input += at(blockIdx.z, blockIdx.y, input_stride);
        output += at(blockIdx.z, blockIdx.y, output_stride);

        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index;

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            const uint gy = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gy < shape[0])) // x could be checked before
                tile[tid[0] + repeat][tid[1]] = input[gy * input_stride[2] + old_gid[1] * input_stride[3]];
        }

        util::block::synchronize();

        // Write transposed tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // Y->X', X->Y'
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            const uint gy = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gy < shape[1]))
                output[gy * output_stride[2] + new_gid[1] * output_stride[3]] = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place transpose the XY slices one at a time.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose0132_inplace_(T* output, uint4_t output_stride, uint shape, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        output += at(blockIdx.z, blockIdx.y, output_stride);

        // Get the current indexes.
        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index;

        if (offset[0] > offset[1]) { // lower triangle
            const uint2_t src_gid = offset + tid;
            const uint2_t dst_gid = offset.flip() + tid; // Y->X', X->Y'

            // Read tiles to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && gy < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[gy * output_stride[2] + src_gid[1] * output_stride[3]];

                const uint dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dy < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output[dy * output_stride[2] + dst_gid[1] * output_stride[3]];
            }

            util::block::synchronize();

            // Write transposed tiles to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint dy = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dy < shape))
                    output[dy * output_stride[2] + dst_gid[1] * output_stride[3]] = tile_src[tid[1]][tid[0] + repeat];

                const uint gy = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && gy < shape))
                    output[gy * output_stride[2] + src_gid[1] * output_stride[3]] = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const uint2_t gid = offset + tid;

            // Read tile to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gy < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[gy * output_stride[2] + gid[1] * output_stride[3]];
            }

            util::block::synchronize();

            // Write transposed tile to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint gy = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gy < shape))
                    output[gy * output_stride[2] + gid[1] * output_stride[3]] = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0132(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                       size4_t shape, Stream& stream) {
        const uint2_t uint_shape(shape.get() + 2);
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_y = math::divideUp(uint_shape[0], TILE_DIM);
        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const dim3 blocks(blocks_y * blocks_x, shape[1], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0132", transpose0132_<T, true>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0132", transpose0132_<T, false>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        }
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose0132(T* output, size4_t output_stride, size4_t shape, Stream& stream) {
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place permutation, shape[2] should be equal to shape[3]. Got shape:{}", shape);

        const uint uint_shape = static_cast<uint>(shape[3]);
        const bool is_multiple_tile = (uint_shape % TILE_DIM) == 0;

        const uint blocks_x = math::divideUp(uint_shape, TILE_DIM); // blocks_y == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[1], shape[0]); // about less than half will be idle blocks.
        if (is_multiple_tile) {
            stream.enqueue("memory::transpose0132_inplace", transpose0132_inplace_<T, true>, {blocks, THREADS},
                           output, uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0132_inplace", transpose0132_inplace_<T, false>, {blocks, THREADS},
                           output, uint4_t{output_stride}, uint_shape, blocks_x);
        }
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                           \
template void noa::cuda::memory::details::transpose0132<T>(const T*, size4_t, T*, size4_t, size4_t, Stream&);   \
template void noa::cuda::memory::details::inplace::transpose0132<T>(T*, size4_t, size4_t, Stream&)

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
