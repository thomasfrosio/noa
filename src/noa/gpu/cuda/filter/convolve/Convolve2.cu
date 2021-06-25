#include "noa/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Shared.h"
#include "noa/gpu/cuda/filter/Convolve.h"

namespace {
    using namespace ::noa;

    constexpr uint2_t THREADS(16, 16);
    constexpr int MAX_FILTER_SIZE = 17;
    __constant__ char cfilter[MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(double)];

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
    __global__ void convolve2_(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                               uint3_t shape, int2_t filter_size, uint blocks_x) {
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

        const int OFFSET = static_cast<int>(THREADS.x); // block is 16x16 square
        const int2_t PADDING(filter_size - 1);
        const int2_t HALO(PADDING / 2);
        const int2_t SHARED_LEN(OFFSET + PADDING);
        const uint tmp = gid.z * shape.y;

        // Load shared memory. Loop to take into account padding.
        for (int ly = tid.y, gy = gid.y; ly < SHARED_LEN.y; ly += OFFSET, gy += OFFSET) {
            int i_y = gy - HALO.y;
            bool is_in_y = (i_y >= 0 && i_y < shape.y);
            for (int lx = tid.x, gx = gid.x; lx < SHARED_LEN.x; lx += OFFSET, gx += OFFSET) {
                int i_x = gx - HALO.x;
                bool is_in_x = (i_x >= 0 && i_x < shape.x);
                shared[ly * SHARED_LEN.x + lx] = (is_in_y && is_in_x) ?
                                                 inputs[(tmp + i_y) * inputs_pitch + i_x] :
                                                 static_cast<T>(0);
            }
        }

        __syncthreads();

        if (gid.x < shape.x && gid.y < shape.y) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int y = 0; y < filter_size.y; ++y)
                for (int x = 0; x < filter_size.x; ++x)
                    result += shared[(tid.y + y) * SHARED_LEN.x + tid.x + x] * window[y * filter_size.x + x];
            outputs[(tmp + gid.y) * outputs_pitch + gid.x] = result;
        }
    }
}

namespace noa::cuda::filter {
    template<typename T>
    void convolve2(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                   size3_t shape, uint batches, const T* filter, uint2_t filter_size, Stream& stream) {
        if (all(filter_size == 1U)) {
            memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                         size3_t(shape.x, getRows(shape), batches), stream);
            return;
        }

        // Copy to constant memory.
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, math::prod(filter_size) * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        uint3_t tmp_shape(shape);
        uint blocks_x = math::divideUp(tmp_shape.x, THREADS.x);
        uint blocks_y = math::divideUp(tmp_shape.y, THREADS.y);
        uint shared_bytes = (THREADS.x + filter_size.x - 1) * (THREADS.y + filter_size.y - 1) * sizeof(T);

        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        dim3 threads(THREADS.x, THREADS.y);
        convolve2_<<<blocks, threads, shared_bytes, stream.get()>>>(
                inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, int2_t(filter_size), blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_CONV2(T) \
    template void convolve2<T>(const T*, size_t, T*, size_t, size3_t, uint, const T*, uint2_t, Stream&)

    INSTANTIATE_CONV2(float);
    INSTANTIATE_CONV2(double);
}
