#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Permute.h"

#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr uint TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0321_(const T* __restrict__ input, uint4_t input_strides,
                      T* __restrict__ output, uint4_t output_strides,
                      uint2_t shape /* ZX */, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        input += blockIdx.z * input_strides[0];
        output += blockIdx.z * output_strides[0];
        input += input_strides[2] * blockIdx.y; // Y->Y'
        output += output_strides[2] * blockIdx.y;

        // Get the current indexes.
        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            uint gz = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gz < shape[0]))
                tile[tid[0] + repeat][tid[1]] = input[gz * input_strides[1] + old_gid[1] * input_strides[3]];
        }

        util::block::synchronize();

        // Write permuted tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            uint gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gz < shape[1]))
                output[gz * output_strides[1] + new_gid[1] * output_strides[3]] = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0321_inplace_(T* output, uint4_t output_strides, uint shape, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        output += blockIdx.z * output_strides[0];
        output += blockIdx.y * output_strides[2];

        // Get the current indexes.
        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const uint2_t src_gid = offset + tid; // ZX
            const uint2_t dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[sz * output_strides[1] + src_gid[1] * output_strides[3]];

                const uint dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output[dz * output_strides[1] + dst_gid[1] * output_strides[3]];
            }

            util::block::synchronize();

            // Write permuted tiles to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    output[dz * output_strides[1] + dst_gid[1] * output_strides[3]] = tile_src[tid[1]][tid[0] + repeat];

                const uint sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    output[sz * output_strides[1] + src_gid[1] * output_strides[3]] = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const uint2_t gid = offset + tid; // ZX

            // Read tile to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output[gz * output_strides[1] + gid[1] * output_strides[3]];
            }

            util::block::synchronize();

            // Write permuted tile to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    output[gz * output_strides[1] + gid[1] * output_strides[3]] = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute0321(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream) {
        const uint2_t uint_shape{shape[1], shape[3]};
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_z = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::permute0321", permute0321_<T, true>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t(input_strides), output.get(), uint4_t(output_strides),
                           uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::permute0321", permute0321_<T, false>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t(input_strides), output.get(), uint4_t(output_strides),
                           uint_shape, blocks_x);
        }
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void permute0321(const shared_t<T[]>& output, size4_t output_strides, size4_t shape, Stream& stream) {
        if (shape[1] != shape[3])
            NOA_THROW("For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got {}", shape);

        const auto uint_shape = static_cast<uint>(shape[1]);
        const bool is_multiple_tile = (uint_shape % TILE_DIM) == 0;

        const uint blocks_x = math::divideUp(uint_shape, TILE_DIM); // blocks_z == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[2], shape[0]);
        if (is_multiple_tile) {
            stream.enqueue("memory::permute0321_inplace", permute0321_inplace_<T, true>, {blocks, BLOCK_SIZE},
                           output.get(), uint4_t(output_strides), uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::permute0321_inplace", permute0321_inplace_<T, false>, {blocks, BLOCK_SIZE},
                           output.get(), uint4_t(output_strides), uint_shape, blocks_x);
        }
        stream.attach(output);
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                                                       \
template void noa::cuda::memory::details::permute0321<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
template void noa::cuda::memory::details::inplace::permute0321<T>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

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
