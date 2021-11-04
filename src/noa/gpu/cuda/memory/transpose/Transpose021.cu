#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Transpose.h"

// More info on Transpose102.cu

namespace {
    using namespace ::noa;

    constexpr uint TILE_DIM = 32;
    constexpr uint2_t THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // Transpose 021 is a specific case. There's no transposition of the XY tile.
    // Only the last two dimensions are swapped, so use different offset.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose021_(const T* __restrict__ inputs, uint inputs_pitch, T* __restrict__ outputs, uint outputs_pitch,
                       uint3_t shape, uint blocks_x) {
        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * rows(shape) * inputs_pitch;
        outputs += batch * rows(shape) * outputs_pitch;

        // Offset to current z.
        inputs += inputs_pitch * shape.y * blockIdx.y; // blockIdx.y == z
        outputs += outputs_pitch * blockIdx.y; // z becomes the new y

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const uint gid_x = TILE_DIM * block_x + tid.x;
        const uint gid_y = TILE_DIM * block_y + tid.y;

        if (!IS_MULTIPLE_OF_TILE && gid_x >= shape.x)
            return;

        uint offset = outputs_pitch * shape.z; // y is the third dimension in the output.
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint idx = gid_y + repeat;
            if (IS_MULTIPLE_OF_TILE || idx < shape.y)
                outputs[idx * offset + gid_x] = inputs[idx * inputs_pitch + gid_x];
        }
    }

    // In-place.
    // This is simply swapping the Y with the X, such as swap(o[z][y][x], o[y][z][x]).
    // The shared memory simply acts as a per thread tmp buffer.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transpose021_(T* outputs, uint outputs_pitch, uint shapeX, uint shapeYZ, uint blocks_x) {
        __shared__ T tile[THREADS.y][THREADS.x];

        // Offset to current batch.
        const uint rows = shapeYZ * shapeYZ;
        const uint batch = blockIdx.z;
        outputs += batch * rows * outputs_pitch;

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const uint3_t gid(TILE_DIM * block_x + tid.x,
                          TILE_DIM * block_y + tid.y,
                          blockIdx.y);

        if (gid.x >= shapeX)
            return;

        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gy = gid.y + repeat;
            if (gid.z > gy)
                continue;

            if (IS_MULTIPLE_OF_TILE || gy < shapeYZ) {
                uint src_idx = (gid.z * shapeYZ + gy) * outputs_pitch + gid.x;
                uint dst_idx = (gy * shapeYZ + gid.z) * outputs_pitch + gid.x;
                tile[tid.y][tid.x] = outputs[dst_idx];
                outputs[dst_idx] = outputs[src_idx];
                outputs[src_idx] = tile[tid.y][tid.x];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose021(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, size_t batches, Stream& stream) {
        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0 && (tmp_shape.y % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        uint blocks_y = math::divideUp(tmp_shape.y, TILE_DIM);
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        if (are_multiple_tile)
            transpose021_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        else
            transpose021_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose021(T* outputs, size_t outputs_pitch, size3_t shape, size_t batches, Stream& stream) {
        if (shape.y != shape.z)
            NOA_THROW("For a \"021\" in-place permutation, shape[1] should be equal to shape[2]. Got {}", shape);

        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.y % TILE_DIM) == 0;

        const uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        const uint blocks_y = math::divideUp(tmp_shape.y, TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        if (are_multiple_tile)
            transpose021_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.y, blocks_x);
        else
            transpose021_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.y, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                               \
template void noa::cuda::memory::details::transpose021<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
template void noa::cuda::memory::details::inplace::transpose021<T>(T*, size_t, size3_t, size_t, Stream&)

NOA_INSTANTIATE_TRANSPOSE_(unsigned char);
NOA_INSTANTIATE_TRANSPOSE_(unsigned short);
NOA_INSTANTIATE_TRANSPOSE_(unsigned int);
NOA_INSTANTIATE_TRANSPOSE_(unsigned long);
NOA_INSTANTIATE_TRANSPOSE_(unsigned long long);
NOA_INSTANTIATE_TRANSPOSE_(char);
NOA_INSTANTIATE_TRANSPOSE_(short);
NOA_INSTANTIATE_TRANSPOSE_(int);
NOA_INSTANTIATE_TRANSPOSE_(long);
NOA_INSTANTIATE_TRANSPOSE_(long long);
NOA_INSTANTIATE_TRANSPOSE_(float);
NOA_INSTANTIATE_TRANSPOSE_(double);
