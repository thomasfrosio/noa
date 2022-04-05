#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Transpose.h"

namespace {
    using namespace ::noa;

    constexpr uint TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // Transpose 0213 is a specific case: the innermost dimension is unchanged,
    // which makes everything much simpler. Only the last two dimensions are swapped:
    //  - input_stride[1]->output_stride[2]
    //  - input_stride[2]->output_stride[1]
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void transpose0213_(const T* __restrict__ input, uint4_t input_stride,
                        T* __restrict__ output, uint4_t output_stride,
                        uint2_t shape /* YX */ , uint blocks_x) {
        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t gid = TILE_DIM * index + tid;
        if (!IS_MULTIPLE_OF_TILE && gid[1] >= shape[1])
            return;

        input += blockIdx.z * input_stride[0];
        output += blockIdx.z * output_stride[0];
        input += blockIdx.y * input_stride[1]; // Z->Y'
        output += blockIdx.y * output_stride[2];

        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint gy = gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || gy < shape[0])
                output[gy * output_stride[1] + gid[1] * output_stride[3]] =
                        input[gy * input_stride[2] + gid[1] * input_stride[3]];
        }
    }

    // In-place.
    // This is simply swapping the Y with the X, such as swap(o[z][y][x], o[y][z][x]).
    // Only process one triangle, plus the diagonal. The other blocks are idle...
    // The shared memory simply acts as a per thread buffer.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void transpose0213_inplace_(T* output, uint4_t output_stride, uint2_t shape, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[BLOCK_SIZE.y][BLOCK_SIZE.x];
        T(& tile)[BLOCK_SIZE.y][BLOCK_SIZE.x] = *reinterpret_cast<T(*)[BLOCK_SIZE.y][BLOCK_SIZE.x]>(&buffer);

        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          TILE_DIM * index[0] + tid[0],
                          TILE_DIM * index[1] + tid[1]);
        if (gid[3] >= shape[1])
            return;

        output += gid[0] * output_stride[0];
        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint gy = gid[2] + repeat;
            if (gid[1] > gy) // process one triangle + diagonal
                continue;

            if (IS_MULTIPLE_OF_TILE || gy < shape[0]) {
                const uint src_offset = gid[1] * output_stride[1] + gy * output_stride[2] + gid[3] * output_stride[3];
                const uint dst_offset = gid[1] * output_stride[2] + gy * output_stride[1] + gid[3] * output_stride[3];
                tile[tid[0]][tid[1]] = output[dst_offset];
                output[dst_offset] = output[src_offset];
                output[src_offset] = tile[tid[0]][tid[1]];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0213(const shared_t<const T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream) {
        const uint2_t uint_shape{shape.get() + 2};
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_y = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0213", transpose0213_<T, true>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0213", transpose0213_<T, false>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        }
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose0213(const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream) {
        if (shape[1] != shape[2])
            NOA_THROW("For a \"0213\" in-place permutation, shape[1] should be equal to shape[2]. Got {}", shape);

        const uint2_t uint_shape{shape.get() + 2};
        const bool are_multiple_tile = (uint_shape[0] % TILE_DIM) == 0;

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_y = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0213_inplace", transpose0213_inplace_<T, true>, {blocks, BLOCK_SIZE},
                           output.get(), uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0213_inplace", transpose0213_inplace_<T, false>, {blocks, BLOCK_SIZE},
                           output.get(), uint4_t{output_stride}, uint_shape, blocks_x);
        }
        stream.attach(output);
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                                                             \
template void noa::cuda::memory::details::transpose0213<T>(const shared_t<const T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&); \
template void noa::cuda::memory::details::inplace::transpose0213<T>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

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
