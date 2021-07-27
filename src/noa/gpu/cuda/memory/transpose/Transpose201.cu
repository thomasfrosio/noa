#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Transpose.h"

namespace {
    using namespace ::noa;

    // 201 transpose: transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr uint TILE_DIM = 32;
    constexpr uint2_t THREADS(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ void transpose201_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                  uint3_t shape, uint blocks_x) {
        __shared__ T tile[TILE_DIM][TILE_DIM + 1];

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * getRows(shape) * inputs_pitch;
        outputs += batch * shape.y * shape.x * outputs_pitch;

        // Offset to current y.
        inputs += inputs_pitch * blockIdx.y; // blockIdx.y == y
        outputs += outputs_pitch * shape.x * blockIdx.y; // y becomes the new z

        // Get the current indexes.
        const uint2_t tid(threadIdx.x, threadIdx.y);
        const uint block_z = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_z * blocks_x;
        const uint block_x_offset = TILE_DIM * block_x;
        const uint block_z_offset = TILE_DIM * block_z;

        // Read tile to shared memory.
        const uint old_gid_x = block_x_offset + tid.x;
        const uint old_gid_z = block_z_offset + tid.y;
        const uint offset = inputs_pitch * shape.y;
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gz = old_gid_z + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid_x < shape.x && gz < shape.z))
                tile[tid.y + repeat][tid.x] = inputs[gz * offset + old_gid_x];
        }

        __syncthreads();

        // Write transposed tile to global memory.
        const uint new_gid_x = block_z_offset + tid.x; // z becomes the new x
        const uint new_gid_y = block_x_offset + tid.y; // x becomes the new y
        for (uint repeat = 0; repeat < TILE_DIM; repeat += THREADS.y) {
            uint gy = new_gid_y + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid_x < shape.z && gy < shape.x))
                outputs[gy * outputs_pitch + new_gid_x] = tile[tid.x][tid.y + repeat];
        }
    }

    // Since all axes are permuted, in-place transpose cannot easily be expressed as a 2D transposition
    // along a COMMON plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose201(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream) {
        const uint3_t tmp_shape(shape);
        const dim3 threads(THREADS.x, THREADS.y);
        const bool are_multiple_tile = (tmp_shape.x % TILE_DIM) == 0 && (tmp_shape.z % TILE_DIM) == 0;

        uint blocks_x = math::divideUp(tmp_shape.x, TILE_DIM);
        uint blocks_z = math::divideUp(tmp_shape.z, TILE_DIM);
        dim3 blocks(blocks_x * blocks_z, shape.y, batches);
        if (are_multiple_tile)
            transpose201_<T, true><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        else
            transpose201_<T, false><<<blocks, threads, 0, stream.id()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T) \
template void noa::cuda::memory::details::transpose201<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&)

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
