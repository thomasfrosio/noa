#include "noa/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Shared.h"
#include "noa/gpu/cuda/filter/Convolve.h"

namespace {
    using namespace ::noa;

    constexpr uint2_t THREADS(16, 16);
    constexpr int MAX_FILTER_SIZE = 129;
    __constant__ char cfilter[MAX_FILTER_SIZE * sizeof(double)];

    // The block is a 2D block (16x16).
    // The launch config:
    //      gridDim.x = number of blocks to compute one entire 2D slice.
    //      gridDim.y = number of slices (the z).
    //      gridDim.z = batches
    //
    // filter_size: Size of the filter. We assume it is a odd value.
    // blocks_x:    This is the number of blocks per row and is used to get the
    //              {x,y} index of the current block (see idx_x and idx_y).
    template<typename T>
    __global__ void convolve1_(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                               uint3_t shape, int filter_size, uint blocks_x) {
        T* shared = cuda::memory::Shared<T>::getBlockResource();

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // index withing the 2D slice

        // Offset to current batch.
        const uint rows = getRows(shape);
        const uint batch = blockIdx.z;
        inputs += batch * rows * inputs_pitch;
        outputs += batch * rows * outputs_pitch;

        if (gid.y < shape.y) {
            const int PADDING = filter_size - 1;
            const int HALO = PADDING / 2;
            const int SHARED_LEN = static_cast<int>(THREADS.x) + PADDING;
            const uint tmp = (gid.z * shape.y + gid.y);
            shared += tid.y * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (int lx = tid.x, gx = gid.x; lx < SHARED_LEN; lx += THREADS.x, gx += THREADS.x) {
                int idx = gx - HALO;
                shared[lx] = (idx >= 0 && idx < shape.x) ? inputs[tmp * inputs_pitch + idx] : static_cast<T>(0);
            }

            __syncthreads();

            if (gid.x < shape.x) {
                // Weighted sum.
                const T* window = reinterpret_cast<T*>(cfilter);
                T result = static_cast<T>(0);
                for (int idx = 0; idx < filter_size; ++idx)
                    result += shared[tid.x + idx] * window[idx];
                outputs[tmp * outputs_pitch + gid.x] = result;
            }
        }
    }
}

namespace noa::cuda::filter {
    template<typename T>
    void convolve1(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                   size3_t shape, uint batches, const T* filter, uint filter_size, Stream& stream) {
        if (filter_size == 1) {
            memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                         size3_t(shape.x, getRows(shape), batches), stream);
            return;
        }

        // Copy to constant memory.
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        uint3_t tmp_shape(shape);
        uint blocks_x = math::divideUp(tmp_shape.x, THREADS.x);
        uint blocks_y = math::divideUp(tmp_shape.y, THREADS.y);
        uint shared_bytes = (THREADS.x + filter_size - 1) * THREADS.y * sizeof(T);

        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        dim3 threads(THREADS.x, THREADS.y);
        convolve1_<<<blocks, threads, shared_bytes, stream.get()>>>(
                inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, filter_size, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_CONV1(T) \
    template void convolve1<T>(const T*, size_t, T*, size_t, size3_t, uint, const T*, uint, Stream&)

    INSTANTIATE_CONV1(float);
    INSTANTIATE_CONV1(double);
}
