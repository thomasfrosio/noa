#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/util/ExternShared.h"
#include "noa/gpu/cuda/filter/Convolve.h"

namespace {
    using namespace ::noa;

    constexpr uint2_t THREADS(16, 16);
    constexpr int MAX_FILTER_SIZE = 129;
    __constant__ char cfilter[MAX_FILTER_SIZE * sizeof(double)];

    // This is identical to the convolve1_ kernel.
    template<typename T>
    __global__ void convolveSeparableX_(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                                        uint3_t shape, int filter_size, uint blocks_x) {
        T* shared = cuda::ExternShared<T>::getBlockResource();

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

        const int PADDING = filter_size - 1;
        const int HALO = PADDING / 2;
        const int SHARED_LEN = static_cast<int>(THREADS.x) + PADDING;

        // Filter along x.
        if (gid.y < shape.y) {
            const uint tmp = (gid.z * shape.y + gid.y);
            shared += tid.y * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (int lx = tid.x, gx = gid.x; lx < SHARED_LEN; lx += THREADS.x, gx += THREADS.x) {
                int i_x = gx - HALO;
                shared[lx] = (i_x >= 0 && i_x < shape.x) ? inputs[tmp * inputs_pitch + i_x] : static_cast<T>(0);
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

    template<typename T>
    __global__ void convolveSeparableY_(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                                        uint3_t shape, int filter_size, uint blocks_x) {
        T* shared = cuda::ExternShared<T>::getBlockResource();

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

        const int PADDING = filter_size - 1;
        const int HALO = PADDING / 2;
        const int SHARED_LEN_Y = static_cast<int>(THREADS.y) + PADDING;

        // Filter along y.
        if (gid.x < shape.x) {
            const uint tmp = gid.z * shape.y;

            for (int ly = tid.y, gy = gid.y; ly < SHARED_LEN_Y; ly += THREADS.y, gy += THREADS.y) {
                int i_y = gy - HALO;
                shared[ly * THREADS.x + tid.x] = (i_y >= 0 && i_y < shape.y) ?
                                                 inputs[(tmp + i_y) * inputs_pitch + gid.x] :
                                                 static_cast<T>(0);
            }
            __syncthreads();

            if (gid.y < shape.y) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result = static_cast<T>(0);
                for (int idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid.y + idx) * THREADS.x + tid.x] * window[idx];
                outputs[(tmp + gid.y) * outputs_pitch + gid.x] = result;
            }
        }
    }

    // The block is a 2D block (16x16), with the first dimension along X and the second along Z.
    // The launch config:
    //      gridDim.x = number of blocks to compute one entire XZ slice.
    //      gridDim.y = number of Y slices (the shape.y).
    //      gridDim.z = batches
    //
    // filter_size: Size of the filter. We assume it is a odd value.
    // blocks_x:    This is the number of blocks per row and is used to get the
    //              {x,z} index of the current block (see idx_x and idx_z).
    template<typename T>
    __global__ void convolveSeparableZ_(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                                        uint3_t shape, int filter_size, uint blocks_x) {
        T* shared = cuda::ExternShared<T>::getBlockResource();

        // Get the current indexes.
        const uint idx_z = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_z * blocks_x;
        const int tid_x = static_cast<int>(threadIdx.x);
        const int tid_z = static_cast<int>(threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + tid_x,
                         blockIdx.y,
                         THREADS.y * idx_z + tid_z); // index withing the 2D slice

        // Offset to current batch.
        const uint rows = getRows(shape);
        const uint batch = blockIdx.z;
        inputs += batch * rows * inputs_pitch;
        outputs += batch * rows * outputs_pitch;

        const int PADDING = filter_size - 1;
        const int HALO = PADDING / 2;
        const int SHARED_LEN_Z = static_cast<int>(THREADS.y) + PADDING;

        if (gid.x < shape.x) { // gid.y cannot be out of bounds and there's no padding in x
            for (int lz = tid_z, gz = gid.z; lz < SHARED_LEN_Z; lz += THREADS.y, gz += THREADS.y) {
                int i_z = gz - HALO;
                shared[lz * THREADS.x + tid_x] = (i_z >= 0 && i_z < shape.z) ?
                                                 inputs[(i_z * shape.y + gid.y) * inputs_pitch + gid.x] :
                                                 static_cast<T>(0);
            }
            __syncthreads();

            // Weighted sum.
            if (gid.z < shape.z) {
                const T* window = reinterpret_cast<T*>(cfilter);
                T result = static_cast<T>(0);
                for (int idx = 0; idx < filter_size; ++idx)
                    result += shared[(tid_z + idx) * THREADS.x + tid_x] * window[idx];
                outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] = result;
            }
        }
    }

    template<int DIM, typename T>
    void launchXY(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, uint batches,
                  const T* filter, uint filter_size, cuda::Stream& stream) {
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        uint3_t tmp_shape(shape);
        uint blocks_x = math::divideUp(tmp_shape.x, THREADS.x);
        uint blocks_y = math::divideUp(tmp_shape.y, THREADS.y);
        dim3 blocks(blocks_x * blocks_y, tmp_shape.z, batches);
        dim3 threads(THREADS.x, THREADS.y);
        if constexpr (DIM == 0) {
            uint shared_bytes = (THREADS.x + filter_size - 1) * THREADS.y * sizeof(T);
            convolveSeparableX_<<<blocks, threads, shared_bytes, stream.get()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, filter_size, blocks_x);
        } else {
            uint shared_bytes = THREADS.x * (THREADS.y + filter_size - 1) * sizeof(T);
            convolveSeparableY_<<<blocks, threads, shared_bytes, stream.get()>>>(
                    inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, filter_size, blocks_x);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void launchZ(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, uint batches,
                 const T* filter, uint filter_size, cuda::Stream& stream) {
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter, filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        uint3_t tmp_shape(shape);
        uint blocks_x = math::divideUp(tmp_shape.x, THREADS.x);
        uint blocks_z = math::divideUp(tmp_shape.z, THREADS.y);
        uint shared_bytes = THREADS.x * (THREADS.y + filter_size - 1) * sizeof(T);

        dim3 blocks(blocks_x * blocks_z, tmp_shape.y, batches);
        dim3 threads(THREADS.x, THREADS.y);
        convolveSeparableZ_<<<blocks, threads, shared_bytes, stream.get()>>>(
                inputs, inputs_pitch, outputs, outputs_pitch, tmp_shape, filter_size, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace noa::cuda::filter {
    template<typename T>
    void convolve(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                  size3_t shape, uint batches,
                  const T* filter0, uint filter0_size,
                  const T* filter1, uint filter1_size,
                  const T* filter2, uint filter2_size,
                  Stream& stream,
                  T* tmp, size_t tmp_pitch) {

        if (filter0 && filter1 && filter2) {
            launchXY<0>(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter0, filter0_size, stream);
            launchXY<1>(outputs, outputs_pitch, tmp, tmp_pitch, shape, batches, filter1, filter1_size, stream);
            launchZ(tmp, tmp_pitch, outputs, outputs_pitch, shape, batches, filter2, filter2_size, stream);

        } else if (filter0 && filter1) {
            launchXY<0>(inputs, inputs_pitch, tmp, tmp_pitch, shape, batches, filter0, filter0_size, stream);
            launchXY<1>(tmp, tmp_pitch, outputs, outputs_pitch, shape, batches, filter1, filter1_size, stream);

        } else if (filter0 && filter2) {
            launchXY<0>(inputs, inputs_pitch, tmp, tmp_pitch, shape, batches, filter0, filter0_size, stream);
            launchZ(tmp, tmp_pitch, outputs, outputs_pitch, shape, batches, filter2, filter2_size, stream);

        } else if (filter1 && filter2) {
            launchXY<1>(inputs, inputs_pitch, tmp, tmp_pitch, shape, batches, filter1, filter1_size, stream);
            launchZ(tmp, tmp_pitch, outputs, outputs_pitch, shape, batches, filter2, filter2_size, stream);

        } else if (filter0) {
            launchXY<0>(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter0, filter0_size, stream);
        } else if (filter1) {
            launchXY<1>(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter1, filter1_size, stream);
        } else if (filter2) {
            launchZ(inputs, inputs_pitch, outputs, outputs_pitch, shape, batches, filter2, filter2_size, stream);
        }
    }

    #define NOA_INSTANTIATE_CONV_(T) \
    template void convolve<T>(const T*, size_t, T*, size_t, size3_t, uint, const T*, uint, const T*, uint, const T*, uint, Stream&, T*, size_t)

    NOA_INSTANTIATE_CONV_(float);
    NOA_INSTANTIATE_CONV_(double);
}
