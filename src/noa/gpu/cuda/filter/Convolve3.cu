#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/ExternShared.h"
#include "noa/gpu/cuda/filter/Convolve.h"

namespace {
    using namespace ::noa;

    constexpr uint2_t THREADS(16, 16);
    constexpr int MAX_FILTER_SIZE = 5;
    __constant__ char cfilter[MAX_FILTER_SIZE * MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(double)];

    // The block is a 2D block (16x16).
    // The launch config:
    //      gridDim.x = number of blocks to compute one entire 2D slice.
    //      gridDim.y = number of slices (the z).
    //      gridDim.z = batches
    //
    // filter_size: Size of the filter. We assume it is a odd value.
    // blocks_x:    This is the number of blocks per row and is used to get the
    //              {x,y} index of the current block (see idx_x and idx_y).
    template<typename T, int FILTER_LEN>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void convolve3_(const T* __restrict__ inputs, uint inputs_pitch,
                    T* __restrict__ outputs, uint outputs_pitch,
                    uint3_t shape, uint blocks_x) {
        static_assert(FILTER_LEN % 2); // only support odd windows.
        constexpr int PADDING = FILTER_LEN - 1; // assume odd
        constexpr int HALO = FILTER_LEN / 2;
        constexpr int3_t SHARED_LEN(THREADS.x + PADDING, THREADS.y + PADDING, FILTER_LEN);
        __shared__ T shared[SHARED_LEN.x * SHARED_LEN.y * SHARED_LEN.z];

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // index withing the 2D slice

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * rows(shape) * inputs_pitch;
        outputs += batch * rows(shape) * outputs_pitch;

        // Load shared memory. Loop to take into account padding.
        constexpr int2_t OFFSET(THREADS);
        for (int lz = 0, gz = gid.z; lz < SHARED_LEN.z; ++lz, ++gz) {
            int i_z = gz - HALO;
            int tmp_z = lz * SHARED_LEN.y * SHARED_LEN.x;
            bool is_in_z = (i_z >= 0 && i_z < shape.z);
            for (int ly = tid.y, gy = gid.y; ly < SHARED_LEN.y; ly += OFFSET.y, gy += OFFSET.y) {
                int i_y = gy - HALO;
                int tmp = tmp_z + ly * SHARED_LEN.x;
                bool is_in_y = (i_y >= 0 && i_y < shape.y);
                for (int lx = tid.x, gx = gid.x; lx < SHARED_LEN.x; lx += OFFSET.x, gx += OFFSET.x) {
                    int i_x = gx - HALO;
                    bool is_in_x = (i_x >= 0 && i_x < shape.x);
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ?
                                       inputs[(i_z * shape.y + i_y) * inputs_pitch + i_x] :
                                       static_cast<T>(0);
                }
            }
        }
        __syncthreads();

        if (gid.x < shape.x && gid.y < shape.y) { // gid.z cannot be out of bounds
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int z = 0; z < FILTER_LEN; ++z)
                for (int y = 0; y < FILTER_LEN; ++y)
                    for (int x = 0; x < FILTER_LEN; ++x)
                        result += shared[(z * SHARED_LEN.y + tid.y + y) * SHARED_LEN.x + tid.x + x] *
                                  window[(z * FILTER_LEN + y) * FILTER_LEN + x];
            outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] = result;
        }
    }

    // Version with filter_length not fixed at compile time.
    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void convolve3_(const T* __restrict__ inputs, uint inputs_pitch,
                    T* __restrict__ outputs, uint outputs_pitch,
                    uint3_t shape, int3_t filter_length, uint blocks_x) {
        const int3_t PADDING(filter_length - 1); // assume odd
        const int3_t HALO = PADDING / 2;
        const int3_t SHARED_LEN(THREADS.x + PADDING.x, THREADS.y + PADDING.y, filter_length.z);
        T* shared = cuda::ExternShared<T>::getBlockResource();

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // index withing the 2D slice

        // Offset to current batch.
        const uint batch = blockIdx.z;
        inputs += batch * rows(shape) * inputs_pitch;
        outputs += batch * rows(shape) * outputs_pitch;

        // Load shared memory. Loop to take into account padding.
        constexpr int2_t OFFSET(THREADS);
        for (int lz = 0, gz = gid.z; lz < SHARED_LEN.z; ++lz, ++gz) {
            int i_z = gz - HALO.z;
            int tmp_z = lz * SHARED_LEN.y * SHARED_LEN.x;
            bool is_in_z = (i_z >= 0 && i_z < shape.z);
            for (int ly = tid.y, gy = gid.y; ly < SHARED_LEN.y; ly += OFFSET.y, gy += OFFSET.y) {
                int i_y = gy - HALO.y;
                int tmp = tmp_z + ly * SHARED_LEN.x;
                bool is_in_y = (i_y >= 0 && i_y < shape.y);
                for (int lx = tid.x, gx = gid.x; lx < SHARED_LEN.x; lx += OFFSET.x, gx += OFFSET.x) {
                    int i_x = gx - HALO.x;
                    bool is_in_x = (i_x >= 0 && i_x < shape.x);
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ?
                                       inputs[(i_z * shape.y + i_y) * inputs_pitch + i_x] :
                                       static_cast<T>(0);
                }
            }
        }
        __syncthreads();

        if (gid.x < shape.x && gid.y < shape.y) { // gid.z cannot be out of bounds
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int z = 0; z < filter_length.z; ++z)
                for (int y = 0; y < filter_length.y; ++y)
                    for (int x = 0; x < filter_length.x; ++x)
                        result += shared[(z * SHARED_LEN.y + tid.y + y) * SHARED_LEN.x + tid.x + x] *
                                  window[(z * filter_length.y + y) * filter_length.x + x];
            outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] = result;
        }
    }
}

namespace noa::cuda::filter {
    template<typename T>
    void convolve3(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                   size3_t shape, size_t batches, const T* filter, size3_t filter_size, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (all(filter_size == size_t{1}))
            return memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                                size3_t(shape.x, rows(shape), batches), stream);

        // Copy to constant memory.
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, math::prod(filter_size) * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        uint3_t int_shape(shape);
        uint blocks_x = math::divideUp(int_shape.x, THREADS.x);
        uint blocks_y = math::divideUp(int_shape.y, THREADS.y);

        dim3 blocks(blocks_x * blocks_y, shape.z, batches);
        dim3 threads(THREADS.x, THREADS.y);
        if (all(filter_size == size_t{5})) {
            convolve3_<T, 5><<<blocks, threads, 0, stream.get()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, int_shape, blocks_x);
        } else if (all(filter_size == size_t{3})) {
            convolve3_<T, 3><<<blocks, threads, 0, stream.get()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, int_shape, blocks_x);
        } else {
            uint shared_bytes = (THREADS.x + filter_size.x - 1) *
                                (THREADS.y + filter_size.y - 1) *
                                filter_size.z * sizeof(T);
            convolve3_<<<blocks, threads, shared_bytes, stream.get()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, int_shape, int3_t(filter_size), blocks_x);
        }

        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_CONV3_(T) \
    template void convolve3<T>(const T*, size_t, T*, size_t, size3_t, size_t, const T*, size3_t, Stream&)

    NOA_INSTANTIATE_CONV3_(float);
    NOA_INSTANTIATE_CONV3_(double);
}
