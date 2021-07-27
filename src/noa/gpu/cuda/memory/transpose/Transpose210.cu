#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Transpose.h"

namespace {
    using namespace ::noa;

    // 210 transpose: transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr uint TILE_DIM = 32;
    constexpr uint2_t THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose210_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, uint blocks_x) {
        __shared__ T tile[TILE_DIM][TILE_DIM + 1];

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * getRows(shape) * inputs_pitch;
        outputs += batch * shape.x * shape.y * outputs_pitch;

        // Offset to current y.
        inputs += inputs_pitch * blockIdx.y; // blockIdx.y == y
        outputs += outputs_pitch * blockIdx.y; // y is still the second dimension in the output

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_z = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_z * blocks_x;
        const uint block_x_offset = TILE_DIM * block_x;
        const uint block_z_offset = TILE_DIM * block_z;

        // Read tile to shared memory.
        const uint old_gid_x = block_x_offset + tid.x;
        const uint old_gid_z = block_z_offset + tid.y;
        uint offset = inputs_pitch * shape.y;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = old_gid_z + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid_x < shape.x && gz < shape.z))
                tile[tid.y + repeat][tid.x] = inputs[gz * offset + old_gid_x];
        }

        __syncthreads();

        // Write transposed tile to global memory.
        const uint new_gid_x = block_z_offset + tid.x; // z becomes the new x dimension
        const uint new_gid_z = block_x_offset + tid.y; // x becomes the new z dimension
        offset = outputs_pitch * shape.y;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = new_gid_z + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid_x < shape.z && gz < shape.x))
                outputs[gz * offset + new_gid_x] = tile[tid.x][tid.y + repeat];
        }
    }

    // In-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose210_(T* outputs, uint outputs_pitch, uint shapeXZ, uint shapeY, uint blocks_x) {
        __shared__ T tile_src[TILE_DIM][TILE_DIM + 1];
        __shared__ T tile_dst[TILE_DIM][TILE_DIM + 1];

        // Offset to current batch and y.
        const uint rows = shapeXZ * shapeY;
        const uint batch = blockIdx.z;
        outputs += batch * rows * outputs_pitch;
        outputs += outputs_pitch * blockIdx.y; // blockIdx.y == y

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_z = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_z * blocks_x;
        const uint block_x_offset = TILE_DIM * block_x;
        const uint block_z_offset = TILE_DIM * block_z;

        if (block_z > block_x) { // lower triangle
            uint src_gid_x = block_x_offset + tid.x;
            uint src_gid_z = block_z_offset + tid.y;
            uint dst_gid_x = block_z_offset + tid.x; // z becomes the new x
            uint dst_gid_z = block_x_offset + tid.y; // x becomes the new z
            uint offset = outputs_pitch * shapeY;

            // Read tiles to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gz = src_gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid_x < shapeXZ && gz < shapeXZ))
                    tile_src[tid.y + repeat][tid.x] = outputs[gz * offset + src_gid_x];

                uint dz = dst_gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid_x < shapeXZ && dz < shapeXZ))
                    tile_dst[tid.y + repeat][tid.x] = outputs[dz * offset + dst_gid_x];
            }

            __syncthreads();

            // Write transposed tiles to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint dz = dst_gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid_x < shapeXZ && dz < shapeXZ))
                    outputs[dz * offset + dst_gid_x] = tile_src[tid.x][tid.y + repeat];

                uint gz = src_gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid_x < shapeXZ && gz < shapeXZ))
                    outputs[gz * offset + src_gid_x] = tile_dst[tid.x][tid.y + repeat];
            }

        } else if (block_x == block_z) { // diagonal
            uint gid_x = block_x_offset + tid.x;
            uint gid_z = block_z_offset + tid.y;
            uint offset = outputs_pitch * shapeY;

            // Read tile to shared memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gz = gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid_x < shapeXZ && gz < shapeXZ))
                    tile_src[tid.y + repeat][tid.x] = outputs[gz * offset + gid_x];
            }

            __syncthreads();

            // Write transposed tile to global memory.
            for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
                uint gz = gid_z + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid_x < shapeXZ && gz < shapeXZ))
                    outputs[gz * offset + gid_x] = tile_src[tid.x][tid.y + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose210(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream) {
        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0 && (tmp_shape.z % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        uint blocks_z = math::divideUp(tmp_shape.z, TILE_DIM);
        dim3 blocks(blocks_x * blocks_z, shape.y, batches);
        if (are_multiple_tile)
            transpose210_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        else
            transpose210_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose210(T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream) {
        if (shape.x != shape.z)
            NOA_THROW("For a \"210\" in-place permutation, shape[0] should be equal to shape[2]. Got {}", shape);

        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool is_multiple_tile = (tmp_shape.x % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM); // blocks_z == blocks_x
        dim3 blocks(blocks_x * blocks_x, shape.y, batches);
        if (is_multiple_tile)
            transpose210_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.y, blocks_x);
        else
            transpose210_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    outputs, outputs_pitch, tmp_shape.x, tmp_shape.y, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                               \
template void noa::cuda::memory::details::transpose210<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
template void noa::cuda::memory::details::inplace::transpose210<T>(T*, size_t, size3_t, uint, Stream&)

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
