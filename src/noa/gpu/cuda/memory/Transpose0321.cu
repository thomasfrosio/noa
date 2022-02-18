#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Transpose.h"

#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr uint TILE_DIM = 32;
    constexpr dim3 THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose0321_(const T* __restrict__ input, uint4_t input_stride,
                       T* __restrict__ output, uint4_t output_stride,
                       uint2_t shape /* ZX */, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        input += blockIdx.z * input_stride[0];
        output += blockIdx.z * output_stride[0];
        input += input_stride[2] * blockIdx.y; // Y->Y'
        output += output_stride[2] * blockIdx.y;

        // Get the current indexes.
        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gz < shape[0]))
                tile[tid[0] + repeat][tid[1]] = input[gz * input_stride[1] + old_gid[1] * input_stride[3]];
        }

        util::block::synchronize();

        // Write transposed tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gz < shape[1]))
                output[gz * output_stride[1] + new_gid[1] * output_stride[3]] = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose0321_inplace_(T* output, uint4_t output_stride, uint shape, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        output += blockIdx.z * output_stride[0];
        output += blockIdx.y * output_stride[2];

        // Get the current indexes.
        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const uint2_t src_gid = offset + tid; // ZX
            const uint2_t dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[sz * output_stride[1] + src_gid[1] * output_stride[3]];

                const uint dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output[dz * output_stride[1] + dst_gid[1] * output_stride[3]];
            }

            util::block::synchronize();

            // Write transposed tiles to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    output[dz * output_stride[1] + dst_gid[1] * output_stride[3]] = tile_src[tid[1]][tid[0] + repeat];

                const uint sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    output[sz * output_stride[1] + src_gid[1] * output_stride[3]] = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const uint2_t gid = offset + tid; // ZX

            // Read tile to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[gz * output_stride[1] + gid[1] * output_stride[3]];
            }

            util::block::synchronize();

            // Write transposed tile to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                const uint gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    output[gz * output_stride[1] + gid[1] * output_stride[3]] = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0321(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                      size4_t shape, Stream& stream) {
        const uint2_t uint_shape{shape[1], shape[3]};
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_z = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0321", transpose0321_<T, true>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0321", transpose0321_<T, false>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        }
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose0321(T* output, size4_t output_stride, size4_t shape, Stream& stream) {
        if (shape[1] != shape[3])
            NOA_THROW("For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got {}", shape);

        const auto uint_shape = static_cast<uint>(shape[1]);
        const bool is_multiple_tile = (uint_shape % TILE_DIM) == 0;

        const uint blocks_x = math::divideUp(uint_shape, TILE_DIM); // blocks_z == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[2], shape[0]);
        if (is_multiple_tile) {
            stream.enqueue("memory::transpose0321_inplace", transpose0321_inplace_<T, true>, {blocks, THREADS},
                           output, uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0321_inplace", transpose0321_inplace_<T, false>, {blocks, THREADS},
                           output, uint4_t{output_stride}, uint_shape, blocks_x);
        }
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                           \
template void noa::cuda::memory::details::transpose0321<T>(const T*, size4_t, T*, size4_t, size4_t, Stream&);   \
template void noa::cuda::memory::details::inplace::transpose0321<T>(T*, size4_t, size4_t, Stream&)

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
