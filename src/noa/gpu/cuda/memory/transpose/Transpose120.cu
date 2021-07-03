#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Transpose.h"

namespace {
    using namespace ::noa;

    // 120 transpose: transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr uint TILE_DIM = 32;
    constexpr uint2_t THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose120_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, uint blocks_x) {
        __shared__ T tile[TILE_DIM][TILE_DIM + 1];

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * getRows(shape) * inputs_pitch;
        outputs += batch * shape.x * shape.z * outputs_pitch;

        // Offset to current z.
        inputs += inputs_pitch * shape.y * blockIdx.y; // blockIdx.y == z
        outputs += outputs_pitch * blockIdx.y; // z becomes the new y

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
            if (IS_MULTIPLE_OF_TILE || (old_gid_x < shape.x && gy < shape.y))
                tile[tid.y + repeat][tid.x] = inputs[gy * inputs_pitch + old_gid_x];
        }

        __syncthreads();

        // Write transposed tile to global memory.
        const uint new_gid_x = block_y_offset + tid.x; // y becomes the new x
        const uint new_gid_z = block_x_offset + tid.y; // x becomes the new z
        const uint offset = outputs_pitch * shape.z;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = new_gid_z + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid_x < shape.y && gz < shape.x))
                outputs[gz * offset + new_gid_x] = tile[tid.x][tid.y + repeat];
        }
    }

    // Since all axes are permuted, in-place transpose cannot easily be expressed as a 2D transposition
    // along a COMMON plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose120(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream) {
        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0 && (tmp_shape.y % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        uint blocks_y = math::divideUp(tmp_shape.y, TILE_DIM);
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        if (are_multiple_tile)
            transpose120_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        else
            transpose120_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

#define INSTANTIATE_TRANSPOSE(T) \
template void noa::cuda::memory::details::transpose120<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&)

INSTANTIATE_TRANSPOSE(unsigned char);
INSTANTIATE_TRANSPOSE(unsigned short);
INSTANTIATE_TRANSPOSE(unsigned int);
INSTANTIATE_TRANSPOSE(unsigned long);
INSTANTIATE_TRANSPOSE(unsigned long long);
INSTANTIATE_TRANSPOSE(char);
INSTANTIATE_TRANSPOSE(short);
INSTANTIATE_TRANSPOSE(int);
INSTANTIATE_TRANSPOSE(long);
INSTANTIATE_TRANSPOSE(long long);
INSTANTIATE_TRANSPOSE(float);
INSTANTIATE_TRANSPOSE(double);
