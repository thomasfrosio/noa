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
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XZ tile along Y becomes X'Y' (X'=Z, Y'=X) along Z' (Z'=Y)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void transpose0231_(const T* __restrict__ input, uint4_t input_stride,
                       T* __restrict__ output, uint4_t output_stride,
                       uint2_t shape /* ZX */, uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        input += blockIdx.z * input_stride[0];
        output += blockIdx.z * output_stride[0];
        input += blockIdx.y * input_stride[2]; // Y->Z'
        output += blockIdx.y * output_stride[1];

        // Get the current indexes.
        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint gz = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gz < shape[0]))
                tile[tid[0] + repeat][tid[1]] = input[gz * input_stride[1] + old_gid[1] * input_stride[3]];
        }

        util::block::synchronize();

        // Write transposed tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Y'X'
        for (uint repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const uint gy = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gy < shape[1]))
                output[gy * output_stride[2] + new_gid[1] * output_stride[3]] = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Since all axes are permuted, in-place transpose cannot easily be expressed as a 2D transposition
    // along a COMMON plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0231(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream) {
        const uint2_t uint_shape{shape[1], shape[3]};
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_z = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0231", transpose0231_<T, true>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0231", transpose0231_<T, false>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        }
        stream.attach(input, output);
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T) \
template void noa::cuda::memory::details::transpose0231<T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

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
