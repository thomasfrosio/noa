// Implementation for Math::minMax() for contiguous and padded layouts.

#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"

// -------------------------- //
// -- FORWARD DECLARATIONS -- //
// -------------------------- //

// Intermediary kernel to reduce large contiguous arrays to max 512 elements.
namespace Noa::CUDA::Math::Details::Contiguous {
    static uint getBlocks(size_t elements);

    template<typename T>
    static void launch(T* input, T* tmp_mins, T* tmp_maxs, uint elements, uint blocks, cudaStream_t stream);

    template<bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* input, T* tmp_mins, T* tmp_maxs, uint elements);
}

// Intermediary kernel to reduce large padded arrays to max 512 elements.
namespace Noa::CUDA::Math::Details::Padded {
    static uint getBlocks(uint rows);

    template<typename T>
    static void launch(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape, uint blocks, cudaStream_t stream);

    template<bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape);
}

namespace Noa::CUDA::Math::Details::Final {
    static uint getThreads(size_t elements);

    // Kernel to reduce small arrays (one array per block). Computes 2 values per batch.
    template<typename T>
    static void launch(T* inputs, T* output_mins, T* output_maxs,
                       size_t elements, uint batches, uint threads, cudaStream_t stream);

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* inputs, uint elements,
                                  T* output_mins, T* output_maxs, T* output_sums, T* output_means, T scale);

    // Kernel to reduce the intermediary results (2 input arrays, per block). Computes 2 values per batch.
    template<typename T>
    static void launch(T* tmp_mins, T* tmp_maxs, T* output_mins, T* output_maxs,
                       size_t tmps, uint batches, uint threads, cudaStream_t stream);

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* tmp_mins, T* tmp_maxs, uint tmps, T* output_mins, T* output_maxs);
}

// ----------------- //
// -- DEFINITIONS -- //
// ----------------- //

namespace Noa::CUDA::Math {
    template<typename T>
    void minMax(T* inputs, T* output_mins, T* output_maxs, size_t elements, uint batches, Stream& stream) {
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = Details::Final::getThreads(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    T* input = inputs + batch * elements;
                    T* mins = output_mins + batch;
                    T* maxs = output_maxs + batch;
                    uint blocks = Noa::Math::min(batches - batch, 32768U);
                    Details::Final::launch(input, mins, maxs, elements, blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream);

        } else {
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> tmp(blocks * 2 * batches); // all mins, then all maxs.
            T* mins, * maxs;
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                mins = tmp.get() + batch * blocks;
                maxs = mins + batches * blocks;
                Details::Contiguous::launch(input, mins, maxs, elements, blocks, stream.get());
            }
            uint threads = Details::Final::getThreads(blocks);
            mins = tmp.get();
            maxs = mins + batches * blocks;
            Details::Final::launch(mins, maxs, output_mins, output_maxs, blocks, batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void minMax(T* inputs, size_t pitch_inputs, T* output_mins, T* output_maxs,
                size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements) {
            Stream::synchronize(stream);
            return;
        }

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> tmp(blocks * 2 * batches); // all mins, then all maxs.
        T* mins, * maxs;
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_inputs * shape_2d.y * batch;
            mins = tmp.get() + batch * blocks;
            maxs = mins + batches * blocks;
            Details::Padded::launch(input, pitch_inputs, mins, maxs, shape_2d, blocks, stream.get());
        }
        uint threads = Details::Final::getThreads(blocks);
        mins = tmp.get();
        maxs = mins + batches * blocks;
        Details::Final::launch(mins, maxs, output_mins, output_maxs, blocks, batches, threads, stream.id());
        Stream::synchronize(stream);
    }
}

// -------------------- //
// -- IMPLEMENTATION -- //
// -------------------- //

// COMMON:
namespace Noa::CUDA::Math::Details {
    template<typename T>
    static NOA_DEVICE void warpMinReduce(volatile T* s_data_tid) {
        if (s_data_tid[32] < *s_data_tid) *s_data_tid = s_data_tid[32];
        if (s_data_tid[16] < *s_data_tid) *s_data_tid = s_data_tid[16];
        if (s_data_tid[8] < *s_data_tid) *s_data_tid = s_data_tid[8];
        if (s_data_tid[4] < *s_data_tid) *s_data_tid = s_data_tid[4];
        if (s_data_tid[2] < *s_data_tid) *s_data_tid = s_data_tid[2];
        if (s_data_tid[1] < *s_data_tid) *s_data_tid = s_data_tid[1];
    }

    template<typename T>
    static NOA_DEVICE void warpMaxReduce(volatile T* s_data_tid) {
        if (*s_data_tid < s_data_tid[32]) *s_data_tid = s_data_tid[32];
        if (*s_data_tid < s_data_tid[16]) *s_data_tid = s_data_tid[16];
        if (*s_data_tid < s_data_tid[8]) *s_data_tid = s_data_tid[8];
        if (*s_data_tid < s_data_tid[4]) *s_data_tid = s_data_tid[4];
        if (*s_data_tid < s_data_tid[2]) *s_data_tid = s_data_tid[2];
        if (*s_data_tid < s_data_tid[1]) *s_data_tid = s_data_tid[1];
    }

    template<typename T>
    static NOA_FD void inPlaceMinMax(T* current_min, T* current_max, T candidate) {
        if (candidate < *current_min) *current_min = candidate;
        if (*current_max < candidate) *current_max = candidate;
    }

    template<int BLOCK_SIZE, typename T>
    static NOA_ID void reduceSharedData(int tid, T* s_mins, T* s_maxs, T* output_min, T* output_max) {
        if constexpr (BLOCK_SIZE == 32) {
            if (tid == 0) {
                for (int i = 1; i < 32; ++i) {
                    if (s_mins[i] < *s_mins) *s_mins = s_mins[i];
                    if (*s_maxs < s_maxs[i]) *s_maxs = s_maxs[i];
                }
                *output_min = *s_mins;
                *output_max = *s_maxs;
            }

        } else {
            T* s_mins_tid = s_mins + tid;
            T* s_maxs_tid = s_maxs + tid;

            if constexpr (BLOCK_SIZE >= 512) {
                if (tid < 256) {
                    if (s_mins_tid[256] < *s_mins_tid) *s_mins_tid = s_mins_tid[256];
                    if (*s_maxs_tid < s_maxs_tid[256]) *s_maxs_tid = s_maxs_tid[256];
                }
                __syncthreads();
            }
            if constexpr (BLOCK_SIZE >= 256) {
                if (tid < 128) {
                    if (s_mins_tid[128] < *s_mins_tid) *s_mins_tid = s_mins_tid[128];
                    if (*s_maxs_tid < s_maxs_tid[128]) *s_maxs_tid = s_maxs_tid[128];
                }
                __syncthreads();
            }
            if constexpr (BLOCK_SIZE >= 128) {
                if (tid < 64) {
                    if (s_mins_tid[64] < *s_mins_tid) *s_mins_tid = s_mins_tid[64];
                    if (*s_maxs_tid < s_maxs_tid[64]) *s_maxs_tid = s_maxs_tid[64];
                }
                __syncthreads();
            }
            if constexpr (BLOCK_SIZE >= 64) {
                if (tid < 32) {
                    warpMinReduce(s_mins_tid);
                    warpMaxReduce(s_maxs_tid);
                }
            }
            if (tid == 0) {
                *output_min = *s_mins;
                *output_max = *s_maxs;
            }
        }
    }
}

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 512U;

    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, T* tmp_mins, T* tmp_maxs, uint elements) {
        __shared__ T s_mins[BLOCK_SIZE];
        __shared__ T s_maxs[BLOCK_SIZE];

        T min = *input, max = *input;
        for (uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x; idx < elements; idx += BLOCK_SIZE * 2 * gridDim.x) {
            inPlaceMinMax(&min, &max, input[idx]);
            if constexpr (TWO_BY_TWO) {
                inPlaceMinMax(&min, &max, input[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlaceMinMax(&min, &max, input[idx + BLOCK_SIZE]);
            }
        }
        s_mins[threadIdx.x] = min;
        s_maxs[threadIdx.x] = max;
        __syncthreads();

        reduceSharedData<BLOCK_SIZE>(threadIdx.x, s_mins, s_maxs, tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x);
    }

    uint getBlocks(size_t elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    template<typename T>
    void launch(T* input, T* tmp_mins, T* tmp_maxs, uint elements, uint blocks, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<true><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, elements);
        } else {
            kernel<false><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16);
    constexpr uint THREADS = BLOCK_SIZE.x * BLOCK_SIZE.y;

    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape) {
        uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x;
        __shared__ T s_mins[THREADS * 2];
        T* s_maxs = s_mins + THREADS;

        T min = *input, max = *input;
        uint offset;
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            offset = row * pitch;
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                    inPlaceMinMax(&min, &max, input[offset + idx]);
                    inPlaceMinMax(&min, &max, input[offset + idx + BLOCK_SIZE.x]);
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    inPlaceMinMax(&min, &max, input[offset + idx]);
            }
        }
        s_mins[tid] = min;
        s_maxs[tid] = max;
        __syncthreads();

        reduceSharedData<THREADS>(tid, s_mins, s_maxs, tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x);
    }

    uint getBlocks(uint rows) {
        constexpr uint MAX_BLOCKS = 512;
        constexpr uint WARPS = BLOCK_SIZE.y;
        uint blocks = (rows + (WARPS - 1)) / WARPS;
        return Noa::Math::min(blocks, MAX_BLOCKS);
    }

    template<typename T>
    void launch(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape, uint blocks, cudaStream_t stream) {
        dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
        bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
        if (two_by_two) {
            kernel<true><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, shape);
        } else {
            kernel<false><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// FINAL REDUCTION:
namespace Noa::CUDA::Math::Details::Final {
    uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2); // compute at least 2 elements.
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* inputs, uint elements, T* output_mins, T* output_maxs) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        __shared__ T s_mins[BLOCK_SIZE * 2];
        T* s_maxs = s_mins + BLOCK_SIZE;

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        T min = *inputs, max = min;
        for (uint idx = tid; idx < elements; idx += BLOCK_SIZE * 2) {
            inPlaceMinMax(&min, &max, inputs[idx]);

            if constexpr (TWO_BY_TWO) {
                inPlaceMinMax(&min, &max, inputs[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlaceMinMax(&min, &max, inputs[idx + BLOCK_SIZE]);
            }
        }
        s_mins[tid] = min;
        s_maxs[tid] = max;
        __syncthreads();

        reduceSharedData<BLOCK_SIZE>(tid, s_mins, s_maxs, output_mins + batch, output_maxs + batch);
    }

    template<typename T>
    void launch(T* inputs, T* output_mins, T* output_maxs,
                size_t elements, uint batches, uint threads, cudaStream_t stream) {
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, 256, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 128:
                    kernel<128, true><<<batches, 128, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 64:
                    kernel<64, true><<<batches, 64, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 32:
                    kernel<32, true><<<batches, 32, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, 256, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 128:
                    kernel<128, false><<<batches, 128, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 64:
                    kernel<64, false><<<batches, 64, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                case 32:
                    kernel<32, false><<<batches, 32, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* tmp_mins, T* tmp_maxs, uint tmps, T* output_mins, T* output_maxs) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        __shared__ T s_mins[BLOCK_SIZE];
        __shared__ T s_maxs[BLOCK_SIZE];

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        uint offset = tmps * batch;
        tmp_mins += offset, tmp_maxs += offset;

        T min = *tmp_mins, max = *tmp_maxs;
        for (uint idx = tid; idx < tmps; idx += BLOCK_SIZE * 2) {
            if (tmp_mins[idx] < min) min = tmp_mins[idx];
            if (max < tmp_maxs[idx]) max = tmp_maxs[idx];

            if constexpr (TWO_BY_TWO) {
                if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < tmps) {
                    if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                    if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
                }
            }
        }
        s_mins[tid] = min;
        s_maxs[tid] = max;
        __syncthreads();

        reduceSharedData<BLOCK_SIZE>(tid, s_mins, s_maxs, output_mins + batch, output_maxs + batch);
    }

    template<typename T>
    void launch(T* tmp_mins, T* tmp_maxs, T* output_mins, T* output_maxs,
                size_t tmps, uint batches, uint threads, cudaStream_t stream) {
        bool two_by_two = !(tmps % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 128:
                    kernel<128, true><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 64:
                    kernel<64, true><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 32:
                    kernel<32, true><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with tmps:{}", threads, tmps);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 128:
                    kernel<128, false><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 64:
                    kernel<64, false><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                case 32:
                    kernel<32, false><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmps, output_mins, output_maxs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with tmps:{}", threads, tmps);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_MIN_MAX(T)                                      \
    template void minMax<T>(T*, T*, T*, size_t, uint, Stream&);         \
    template void minMax<T>(T*, size_t, T*, T*, size3_t, uint, Stream&)

    INSTANTIATE_MIN_MAX(float);
    INSTANTIATE_MIN_MAX(double);
    INSTANTIATE_MIN_MAX(int);
    INSTANTIATE_MIN_MAX(uint);
    INSTANTIATE_MIN_MAX(char);
    INSTANTIATE_MIN_MAX(unsigned char);
}
