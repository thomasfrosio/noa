#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Transpose.h"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace {
    using namespace ::noa;

    // 102 transpose: transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr uint TILE_DIM = 32;
    constexpr uint2_t THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose102_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, uint blocks_x) {
        __shared__ T tile[TILE_DIM][TILE_DIM + 1]; // +1 so that elements in a column map to different banks.

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * rows(shape) * inputs_pitch;
        outputs += batch * shape.z * shape.x * outputs_pitch;

        // Offset to current z.
        inputs += inputs_pitch * shape.y * blockIdx.y; // blockIdx.y == z
        outputs += outputs_pitch * shape.x * blockIdx.y;

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const uint block_x_offset = TILE_DIM * block_x;
        const uint block_y_offset = TILE_DIM * block_y;

        // Read tile to shared memory.
        const uint old_gid_x = block_x_offset + tid.x;
        const uint old_gid_y = block_y_offset + tid.y;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gy = old_gid_y + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid_x < shape.x && gy < shape.y)) // x could be checked before
                tile[tid.y + repeat][tid.x] = inputs[gy * inputs_pitch + old_gid_x];
        }

        __syncthreads();

        // Write transposed tile to global memory.
        const uint new_gid_x = block_y_offset + tid.x; // y becomes the new x
        const uint new_gid_y = block_x_offset + tid.y; // x becomes the new y
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gy = new_gid_y + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid_x < shape.y && gy < shape.x))
                outputs[gy * outputs_pitch + new_gid_x] = tile[tid.x][tid.y + repeat]; // copy a column to a row.
        }
    }

    // In-place.
    // Since the last dimension is unchanged, we can simply in-place transpose the XY slices one at a time.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose102_(T* outputs, uint outputs_pitch, uint shapeXY, uint shapeZ, uint blocks_x) {
        __shared__ T tile_src[TILE_DIM][TILE_DIM + 1];
        __shared__ T tile_dst[TILE_DIM][TILE_DIM + 1];

        // Offset to current z.
        const uint rows = shapeXY * shapeZ;
        const uint batch = blockIdx.z;
        outputs += batch * rows * outputs_pitch;
        outputs += outputs_pitch * shapeXY * blockIdx.y; // blockIdx.y == z

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const uint block_x_offset = TILE_DIM * block_x;
        const uint block_y_offset = TILE_DIM * block_y;

        if (block_y > block_x) { // lower triangle
            uint src_gid_x = block_x_offset + tid.x;
            uint src_gid_y = block_y_offset + tid.y;
            uint dst_gid_x = block_y_offset + tid.x; // y becomes the new x
            uint dst_gid_y = block_x_offset + tid.y; // x becomes the new y

            // Read tiles to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gy = src_gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid_x < shapeXY && gy < shapeXY))
                    tile_src[tid.y + repeat][tid.x] = outputs[gy * outputs_pitch + src_gid_x];

                uint dy = dst_gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid_x < shapeXY && dy < shapeXY))
                    tile_dst[tid.y + repeat][tid.x] = outputs[dy * outputs_pitch + dst_gid_x];
            }

            __syncthreads();

            // Write transposed tiles to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint dy = dst_gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid_x < shapeXY && dy < shapeXY))
                    outputs[dy * outputs_pitch + dst_gid_x] = tile_src[tid.x][tid.y + repeat];

                uint gy = src_gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid_x < shapeXY && gy < shapeXY))
                    outputs[gy * outputs_pitch + src_gid_x] = tile_dst[tid.x][tid.y + repeat];
            }

        } else if (block_y == block_x) { // diagonal
            uint gid_x = block_x_offset + tid.x;
            uint gid_y = block_y_offset + tid.y;

            // Read tile to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gy = gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid_x < shapeXY && gy < shapeXY))
                    tile_src[tid.y + repeat][tid.x] = outputs[gy * outputs_pitch + gid_x];
            }

            __syncthreads();

            // Write transposed tile to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gy = gid_y + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid_x < shapeXY && gy < shapeXY))
                    outputs[gy * outputs_pitch + gid_x] = tile_src[tid.x][tid.y + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose102(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream) {
        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0 && (tmp_shape.y % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        uint blocks_y = math::divideUp(tmp_shape.y, TILE_DIM);
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        if (are_multiple_tile)
            transpose102_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        else
            transpose102_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose102(T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream) {
        if (shape.x != shape.y)
            NOA_THROW("For a \"102\" in-place permutation, shape[0] should be equal to shape[1]. Got {}", shape);

        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM); // blocks_y == blocks_x
        dim3 blocks(blocks_x * blocks_x, shape.z, batches); // about less than half will be idle blocks.
        if (are_multiple_tile)
            transpose102_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.z, blocks_x);
        else
            transpose102_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.z, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                               \
template void noa::cuda::memory::details::transpose102<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
template void noa::cuda::memory::details::inplace::transpose102<T>(T*, size_t, size3_t, uint, Stream&)

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
