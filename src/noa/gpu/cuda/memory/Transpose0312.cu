#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Transpose.h"

#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr uint TILE_DIM = 32;
    constexpr dim3 THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XY tile along Z becomes X'Z' (X'=Y, Z'=X) along Y' (Y'=Z)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose0312_(const T* __restrict__ input, uint4_t input_stride,
                        T* __restrict__ output, uint4_t output_stride,
                        uint2_t shape /* YX */ , uint blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        input += blockIdx.z * input_stride[0];
        output += blockIdx.z * output_stride[0];
        input += blockIdx.y * input_stride[1];
        output += blockIdx.y * output_stride[2];

        const uint2_t tid(threadIdx.y, threadIdx.x);
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index;

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            const uint gy = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gy < shape[0]))
                tile[tid[0] + repeat][tid[1]] = input[gy * input_stride[2] + old_gid[1] * input_stride[3]];
        }

        util::block::synchronize();

        // Write transposed tile to global memory.
        const uint2_t new_gid = offset.flip() + tid;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            const uint gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gz < shape[1]))
                output[gz * output_stride[1] + new_gid[1] * output_stride[3]] = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Since all axes are permuted, in-place transpose cannot easily be expressed as a 2D transposition
    // along a common plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0312(const T* input, size4_t input_stride, T* output, size4_t output_stride,
                       size4_t shape, Stream& stream) {
        const uint2_t uint_shape(shape.get() + 2);
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint blocks_y = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        if (are_multiple_tile) {
            stream.enqueue("memory::transpose0312", transpose0312_<T, true>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::transpose0312", transpose0312_<T, false>, {blocks, THREADS},
                           input, uint4_t{input_stride}, output, uint4_t{output_stride}, uint_shape, blocks_x);
        }
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T) \
template void noa::cuda::memory::details::transpose0312<T>(const T*, size4_t, T*, size4_t, size4_t, Stream&)

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
