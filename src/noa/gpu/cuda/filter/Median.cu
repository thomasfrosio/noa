#include "noa/common/Math.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/filter/Median.h"
#include "noa/gpu/cuda/memory/Copy.h"

// The current implementations only supports small squared windows. This allows to:
//  1)  Load the windows for all threads in a block in shared memory. This is useful because windows overlap.
//  2)  The exchange search can be done on the per thread registers. Only about half of the window needs to
//      be on the registers at a single time. The rest stays in shared memory. This also requires the indexing
//      to be constant, i.e. the window size should be a template argument.
// TODO Maybe look at other implementations for larger windows?

namespace {
    using namespace noa;

    constexpr int2_t THREADS(16, 16);

    // Ensures a < b. If not, swap.
    // TODO test with #define and min/max version.
    template<typename T>
    __forceinline__ __device__ void swap_(T& a, T& b) {
        if (b < a) {
            T tmp = a;
            a = b;
            b = tmp;
        }
    }

    // Sets the array v, so that the min is at v[0] and max is at v[length-1].
    // TODO make sure the force inline still allows constant indexing, otherwise replace with a macro.
    template<typename T>
    __forceinline__ __device__ void order_(T* v, int length) {
        for (int i = 0; i < length / 2; i++) // ensure min in first half, max in second half
            swap_(v[i], v[length - 1 - i]);
        for (int i = 1; i < (length + 1) / 2; i++) // move min in first half to first pos
            swap_(v[0], v[i]);
        for (int i = length - 2; i >= length / 2; i--) // move max in second half to last pos
            swap_(v[i], v[length - 1]);
    }

    // Loads the shared memory array according to the BORDER_MODE.
    // in:      device memory. Should point to the beginning of the current row.
    // s_out:   shared memory. Should point to the current element.
    // shape_x: Number of logical elements in x, x being the dimension of the 1D window.
    // gx:      index of the current element in x. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int BORDER_MODE, int HALO>
    __device__ void loadToShared1D_(const T* in, T* s_out, int shape_x, int gx) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0)
                *s_out = in[-gx]; // pad left; requires shape_x >= HALO + 1, since gx >= -HALO
            else if (gx < shape_x)
                *s_out = in[gx]; // in array, no padding
            else if (gx < shape_x + HALO)
                *s_out = in[2 * (shape_x - 1) - gx]; // pad right; requires shape_x >= HALO  + 1
            // Otherwise, don't do anything since the *s_out element will not be used anyway.

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x)
                *s_out = static_cast<T>(0);
            else
                *s_out = in[gx]; // this should coalesce nicely
        }
    }

    // The block is a 2D block (16x16).
    // The launch config:
    //      gridDim.x = number of blocks to compute one entire 2D slice.
    //      gridDim.y = number of slices (the z).
    //      gridDim.z = batches
    //
    // blocks_x: This is the number of blocks per row and is used to get the
    //           {x,y} index of the current block (see idx_x and idx_y).
    template<typename T, int BORDER_MODE, int WINDOW_SIZE>
    __global__ void medfilt1_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape, uint blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding in the first dimension.
        constexpr int2_t SHARED_SIZE(THREADS.x + PADDING, THREADS.y);
        __shared__ T shared_mem[SHARED_SIZE.y * SHARED_SIZE.x];

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // index withing the 2D slice

        // Offset to current batch.
        const uint batch = blockIdx.z;
        in += batch * rows(shape) * in_pitch;
        out += batch * rows(shape) * out_pitch;

        // There's no padding in y, so if out of bounds, stop.
        if (gid.y < shape.y) {
            // Load shared memory. Loop to take into account padding.
            uint tmp = (gid.z * shape.y + gid.y);
            for (int lx = tid.x, gx = gid.x; lx < SHARED_SIZE.x; lx += THREADS.x, gx += THREADS.x) {
                loadToShared1D_<T, BORDER_MODE, HALO>(in + tmp * in_pitch,
                                                      shared_mem + tid.y * SHARED_SIZE.x + lx,
                                                      shape.x, gx - HALO);
            }

            __syncthreads();

            // Only continue if not out of bound.
            if (gid.x < shape.x) {
                // The goal is to reduce register pressure as much as possible, but still use registers
                // to do the exchange sort. The window is divided into two half: the first "activate" half,
                // which is where the sorting happens, and the second half, which is the pool of contestants
                // that will be progressively added.
                constexpr int ACTIVE = WINDOW_SIZE / 2 + 2; // the final size should be at least 3.

                // Load active half to, hopefully, the local registers (otherwise spill to device memory).
                T v[ACTIVE]; // all indexing are known at compile, so registers should be used
                for (int x = 0; x < ACTIVE; ++x)
                    v[x] = shared_mem[tid.y * SHARED_SIZE.x + tid.x + x];

                order_(v, ACTIVE); // ensure min at 0, max at ACTIVE - 1

                // We can replace the min by a new contestant and then reorder.
                // We also don't have to compare the max at ACTIVE - 1, we just need to stack the new max at the end.
                // As such, as we add new contestants, we right-truncate the active half to ignore the previous max.
                int length = ACTIVE;
                for (int k = ACTIVE; k < WINDOW_SIZE; ++k) {
                    v[0] = shared_mem[tid.y * SHARED_SIZE.x + tid.x + k]; // replace min by new contestant
                    --length; // ignore the previous max at the end
                    order_(v, length); // min at 0, max at length - 1
                }

                // Since we know the min and max are at v[0] and v[length-1] respectively, to find the median,
                // we simply need to (exchange) sort the elements from v[1] to v[length-2] included.
                // The median will then be at length/2.
                for (int k = 1; k < length - 1; k++)
                    order_(v + k, length - k);
                out[tmp * out_pitch + gid.x] = v[length / 2];
            }
        }
    }

    // Loads the (gx,gy) element according to the BORDER_MODE.
    // in:          device memory. Should point to the beginning of the current page (z).
    // pitch:       number of physical elements in x.
    // s_out:       shared memory. Should point to the current element.
    // shape_x/y:   number of logical elements in x/y.
    // gx/y:        index of the current element in x/y. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int BORDER_MODE, int HALO>
    __device__ void loadToShared2D_(const T* in, uint pitch, T* s_out,
                                    int shape_x, int gx, int shape_y, int gy) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0)
                gx *= -1;
            else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return; // don't do anything; gx is out of the shape + window so *s_out won't be used anyway
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0)
                gy *= -1;
            else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            *s_out = in[gy * pitch + gx];

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x || gy < 0 || gy >= shape_y)
                *s_out = static_cast<T>(0);
            else
                *s_out = in[gy * pitch + gx];
        }
    }

    // The launch config and block size is like medfilt1_.
    template<typename T, int BORDER_MODE, int WINDOW_SIZE>
    __global__ void medfilt2_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape, uint blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE;
        constexpr int PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding the first and second dimension.
        constexpr int2_t SHARED_SIZE(THREADS.x + PADDING, THREADS.y + PADDING);
        __shared__ T shared_mem[SHARED_SIZE.y * SHARED_SIZE.x];

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // indexes withing the 2D slice

        // Offset to current batch.
        const uint batch = blockIdx.z;
        in += batch * rows(shape) * in_pitch;
        out += batch * rows(shape) * out_pitch;

        // Load shared memory. Loop to account for the halo.
        const T* in_z = in + gid.z * shape.y * in_pitch; // offset to current slice
        for (int ly = tid.y, gy = gid.y; ly < SHARED_SIZE.y; ly += THREADS.y, gy += THREADS.y)
            for (int lx = tid.x, gx = gid.x; lx < SHARED_SIZE.x; lx += THREADS.x, gx += THREADS.x)
                loadToShared2D_<T, BORDER_MODE, HALO>(in_z, in_pitch,
                                                      shared_mem + ly * SHARED_SIZE.x + lx,
                                                      shape.x, gx - HALO,
                                                      shape.y, gy - HALO);

        __syncthreads();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid.x < shape.x && gid.y < shape.y) {
            constexpr int ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (int count = 0, y = 0; y < WINDOW_SIZE; ++y)
                for (int x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                    v[count] = shared_mem[(tid.y + y) * SHARED_SIZE.x + tid.x + x];

            order_(v, ACTIVE);
            int length = ACTIVE;
            for (int k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y) coordinates in the 2D window.
                // Then offset these coordinates to the current thread.
                int coord_y = k / WINDOW_SIZE;
                int coord_x = k - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_y + tid.y) * SHARED_SIZE.x + coord_x + tid.x];

                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (int k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            out[(gid.z * shape.y + gid.y) * out_pitch + gid.x] = v[length / 2];
        }
    }

    // Loads the (gx,gy,gz) element according to the BORDER_MODE.
    // in:          device memory. Should point to the beginning of the batch.
    // pitch:       number of physical elements in x.
    // s_out:       shared memory. Should point to the current element.
    // shape_x/y/z: number of logical elements in x/y/z.
    // gx/y/z:      index of the current element in x/y/z. If out of bound, add padding according to the BORDER_MODE.
    template<typename T, int BORDER_MODE, int HALO>
    __device__ void loadToShared3D_(const T* in, uint pitch, T* s_out,
                                    int shape_x, int gx,
                                    int shape_y, int gy,
                                    int shape_z, int gz) {
        static_assert(BORDER_MODE == BORDER_REFLECT || BORDER_MODE == BORDER_ZERO);
        if constexpr (BORDER_MODE == BORDER_REFLECT) {
            if (gx < 0) {
                gx *= -1;
            } else if (gx >= shape_x) {
                if (gx >= shape_x + HALO)
                    return;
                gx = 2 * (shape_x - 1) - gx;
            }
            if (gy < 0) {
                gy *= -1;
            } else if (gy >= shape_y) {
                if (gy >= shape_y + HALO)
                    return;
                gy = 2 * (shape_y - 1) - gy;
            }
            if (gz < 0) {
                gz *= -1;
            } else if (gz >= shape_z) {
                if (gz >= shape_z + HALO)
                    return;
                gz = 2 * (shape_z - 1) - gz;
            }
            *s_out = in[(gz * shape_y + gy) * pitch + gx];

        } else if constexpr (BORDER_MODE == BORDER_ZERO) {
            if (gx < 0 || gx >= shape_x ||
                gy < 0 || gy >= shape_y ||
                gz < 0 || gz >= shape_z)
                *s_out = static_cast<T>(0);
            else
                *s_out = in[(gz * shape_y + gy) * pitch + gx];
        }
    }

    // The launch config and block size is like medfilt1_.
    template<typename T, int BORDER_MODE, uint WINDOW_SIZE>
    __global__ void medfilt3_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape, uint blocks_x) {
        static_assert(WINDOW_SIZE % 2); // only support odd windows.
        constexpr int TILE_SIZE = WINDOW_SIZE * WINDOW_SIZE * WINDOW_SIZE;
        constexpr int PADDING = WINDOW_SIZE - 1; // assume odd
        constexpr int HALO = PADDING / 2;

        // The shared memory is the shape of the block + the padding all 3 dimensions.
        // For the largest supported window (5), this goes up to 20*20*5=2205 elements.
        constexpr int3_t SHARED_SIZE(THREADS.x + PADDING, THREADS.y + PADDING, WINDOW_SIZE);
        __shared__ T shared_mem[SHARED_SIZE.z * SHARED_SIZE.y * SHARED_SIZE.x];

        // Offset to current batch.
        const uint batch = blockIdx.z;
        in += batch * rows(shape) * in_pitch;
        out += batch * rows(shape) * out_pitch;

        // Get the current indexes.
        const uint idx_y = blockIdx.x / blocks_x;
        const uint idx_x = blockIdx.x - idx_y * blocks_x;
        const int2_t tid(threadIdx.x, threadIdx.y); // indexes withing the block
        const int3_t gid(THREADS.x * idx_x + threadIdx.x,
                         THREADS.y * idx_y + threadIdx.y,
                         blockIdx.y); // index withing the 2D slice

        // Load shared memory.
        // Each thread processes at least WINDOW_SIZE elements (the z dimension).
        for (int lz = 0, gz = gid.z; lz < SHARED_SIZE.z; ++lz, ++gz)
            for (int ly = tid.y, gy = gid.y; ly < SHARED_SIZE.y; ly += THREADS.y, gy += THREADS.y)
                for (int lx = tid.x, gx = gid.x; lx < SHARED_SIZE.x; lx += THREADS.x, gx += THREADS.x)
                    loadToShared3D_<T, BORDER_MODE, HALO>(in, in_pitch,
                                                          shared_mem + (lz * SHARED_SIZE.y + ly) * SHARED_SIZE.x + lx,
                                                          shape.x, gx - HALO,
                                                          shape.y, gy - HALO,
                                                          shape.z, gz - HALO);

        __syncthreads();

        // Only continue if not out of bound. gid.z cannot be out of bound.
        if (gid.x < shape.x && gid.y < shape.y) {
            constexpr int ACTIVE = TILE_SIZE / 2 + 2;

            // Load active window from shared memory into this 1D array.
            T v[ACTIVE];
            for (int count = 0, z = 0; z < WINDOW_SIZE; ++z)
                for (int y = 0; count < ACTIVE && y < WINDOW_SIZE; ++y)
                    for (int x = 0; count < ACTIVE && x < WINDOW_SIZE; ++x, ++count)
                        v[count] = shared_mem[(z * SHARED_SIZE.y + tid.y + y) * SHARED_SIZE.x + tid.x + x];

            order_(v, ACTIVE);
            int length = ACTIVE;
            for (int k = ACTIVE; k < TILE_SIZE; ++k) {
                // k is the index, but we want the corresponding (x,y,z) coordinates in the 3D window.
                // Then offset these coordinates to the current thread.
                int coord_z = k / (WINDOW_SIZE * WINDOW_SIZE);
                int tmp = k - coord_z * WINDOW_SIZE * WINDOW_SIZE;
                int coord_y = tmp / WINDOW_SIZE;
                int coord_x = tmp - coord_y * WINDOW_SIZE;
                v[0] = shared_mem[(coord_z * SHARED_SIZE.y + coord_y + tid.y) * SHARED_SIZE.x + coord_x + tid.x];
                --length;
                order_(v, length);
            }

            // Sort the final elements.
            for (int k = 1; k < length - 1; k++)
                order_(v + k, length - k);
            out[(gid.z * shape.y + gid.y) * out_pitch + gid.x] = v[length / 2];
        }
    }
}

namespace noa::cuda::filter {
    #define NOA_SWITCH_CASE_(KERNEL, MODE, N)                                               \
        case N: {                                                                           \
            KERNEL<T, MODE, N><<<blocks, threads, 0, stream.get()>>>(                       \
                inputs, inputs_pitch, outputs, outputs_pitch, uint3_t(shape), blocks_x);    \
            break;                                                                          \
        }

    template<typename T>
    void median1(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, uint batches,
                 BorderMode border_mode, uint window, Stream& stream) {
        if (window == 1) {
            memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                         size3_t(shape.x, rows(shape), batches), stream);
            return;
        }

        dim3 threads(THREADS.x, THREADS.y);
        uint blocks_x = (shape.x + threads.x - 1) / threads.x;
        uint blocks_y = (shape.y + threads.y - 1) / threads.y;
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);

        if (border_mode == BORDER_REFLECT) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 3)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 5)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 7)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 9)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 11)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 13)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 15)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 17)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 19)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_REFLECT, 21)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 21, got {}", window);
            }
        } else if (border_mode == BORDER_ZERO) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 3)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 5)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 7)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 9)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 11)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 13)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 15)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 17)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 19)
                NOA_SWITCH_CASE_(medfilt1_, BORDER_ZERO, 21)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 21, got {}", window);
            }
        } else {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void median2(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, uint batches,
                 BorderMode border_mode, uint window, Stream& stream) {
        if (window == 1) {
            memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                         size3_t(shape.x, rows(shape), batches), stream);
            return;
        }

        dim3 threads(THREADS.x, THREADS.y);
        uint blocks_x = (shape.x + threads.x - 1) / threads.x;
        uint blocks_y = (shape.y + threads.y - 1) / threads.y;
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);

        if (border_mode == BORDER_REFLECT) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt2_, BORDER_REFLECT, 3)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_REFLECT, 5)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_REFLECT, 7)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_REFLECT, 9)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_REFLECT, 11)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window);
            }
        } else if (border_mode == BORDER_ZERO) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt2_, BORDER_ZERO, 3)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_ZERO, 5)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_ZERO, 7)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_ZERO, 9)
                NOA_SWITCH_CASE_(medfilt2_, BORDER_ZERO, 11)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 11, got {}", window);
            }
        } else {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void median3(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape, uint batches,
                 BorderMode border_mode, uint window, Stream& stream) {
        if (window == 1) {
            memory::copy(inputs, inputs_pitch, outputs, outputs_pitch,
                         size3_t(shape.x, rows(shape), batches), stream);
            return;
        }

        dim3 threads(THREADS.x, THREADS.y);
        uint blocks_x = (shape.x + threads.x - 1) / threads.x;
        uint blocks_y = (shape.y + threads.y - 1) / threads.y;
        dim3 blocks(blocks_x * blocks_y, shape.z, batches);

        if (border_mode == BORDER_REFLECT) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt3_, BORDER_REFLECT, 3)
                NOA_SWITCH_CASE_(medfilt3_, BORDER_REFLECT, 5)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 5, got {}", window);
            }
        } else if (border_mode == BORDER_ZERO) {
            switch (window) {
                NOA_SWITCH_CASE_(medfilt3_, BORDER_ZERO, 3)
                NOA_SWITCH_CASE_(medfilt3_, BORDER_ZERO, 5)
                default:
                    NOA_THROW("Unsupported window size. It should be an odd number from 1 to 5, got {}", window);
            }
        } else {
            NOA_THROW("BorderMode not supported. Should be {} or {}, got {}",
                      BORDER_REFLECT, BORDER_ZERO, border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_MEDIAN_(T)                                                                      \
    template void median1<T>(const T*, size_t, T*, size_t, size3_t, uint, BorderMode, uint, Stream&);  \
    template void median2<T>(const T*, size_t, T*, size_t, size3_t, uint, BorderMode, uint, Stream&);  \
    template void median3<T>(const T*, size_t, T*, size_t, size3_t, uint, BorderMode, uint, Stream&)

    NOA_INSTANTIATE_MEDIAN_(float);
    NOA_INSTANTIATE_MEDIAN_(double);
    NOA_INSTANTIATE_MEDIAN_(int);
    NOA_INSTANTIATE_MEDIAN_(uint);
}
