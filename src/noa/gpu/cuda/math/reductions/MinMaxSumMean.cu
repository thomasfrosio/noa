// Implementation for Math::minMaxSumMean() for contiguous and padded layouts.

#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace {
    using namespace Noa;

    template<typename T>
    __device__ void warpSumReduce_(volatile T* s_data_tid) {
        T t = *s_data_tid;
        t = t + s_data_tid[32];
        *s_data_tid = t;
        t = t + s_data_tid[16];
        *s_data_tid = t;
        t = t + s_data_tid[8];
        *s_data_tid = t;
        t = t + s_data_tid[4];
        *s_data_tid = t;
        t = t + s_data_tid[2];
        *s_data_tid = t;
        t = t + s_data_tid[1];
        *s_data_tid = t;
    }

    template<typename T>
    __device__ void warpMinReduce_(volatile T* s_data_tid) {
        if (s_data_tid[32] < *s_data_tid) *s_data_tid = s_data_tid[32];
        if (s_data_tid[16] < *s_data_tid) *s_data_tid = s_data_tid[16];
        if (s_data_tid[8] < *s_data_tid) *s_data_tid = s_data_tid[8];
        if (s_data_tid[4] < *s_data_tid) *s_data_tid = s_data_tid[4];
        if (s_data_tid[2] < *s_data_tid) *s_data_tid = s_data_tid[2];
        if (s_data_tid[1] < *s_data_tid) *s_data_tid = s_data_tid[1];
    }

    template<typename T>
    __device__ void warpMaxReduce_(volatile T* s_data_tid) {
        if (*s_data_tid < s_data_tid[32]) *s_data_tid = s_data_tid[32];
        if (*s_data_tid < s_data_tid[16]) *s_data_tid = s_data_tid[16];
        if (*s_data_tid < s_data_tid[8]) *s_data_tid = s_data_tid[8];
        if (*s_data_tid < s_data_tid[4]) *s_data_tid = s_data_tid[4];
        if (*s_data_tid < s_data_tid[2]) *s_data_tid = s_data_tid[2];
        if (*s_data_tid < s_data_tid[1]) *s_data_tid = s_data_tid[1];
    }

    template<typename T>
    inline __device__ void inPlaceMinMaxSum_(T* current_min, T* current_max, T* current_sum, T candidate) {
        *current_sum += candidate;
        if (candidate < *current_min) *current_min = candidate;
        if (*current_max < candidate) *current_max = candidate;
    }

    template<typename T>
    inline __device__ void reduceSharedData_(int tid, T* s_mins, T* s_maxs, T* s_sums,
                                             T* output_min, T* output_max, T* output_sum) {
        if (tid < 256) {
            s_sums[tid] += s_sums[tid + 256];
            if (s_mins[tid + 256] < s_mins[tid]) s_mins[tid] = s_mins[tid + 256];
            if (s_maxs[tid] < s_maxs[tid + 256]) s_maxs[tid] = s_maxs[tid + 256];
        }
        __syncthreads();
        if (tid < 128) {
            s_sums[tid] += s_sums[tid + 128];
            if (s_mins[tid + 128] < s_mins[tid]) s_mins[tid] = s_mins[tid + 128];
            if (s_maxs[tid] < s_maxs[tid + 128]) s_maxs[tid] = s_maxs[tid + 128];
        }
        __syncthreads();
        if (tid < 64) {
            s_sums[tid] += s_sums[tid + 64];
            if (s_mins[tid + 64] < s_mins[tid]) s_mins[tid] = s_mins[tid + 64];
            if (s_maxs[tid] < s_maxs[tid + 64]) s_maxs[tid] = s_maxs[tid + 64];
        }
        __syncthreads();
        if (tid < 32) {
            warpSumReduce_(s_sums + tid);
            warpMinReduce_(s_mins + tid);
            warpMaxReduce_(s_maxs + tid);
        }
        if (tid == 0) {
            *output_sum = *s_sums;
            *output_min = *s_mins;
            *output_max = *s_maxs;
        }
    }

    template<int BLOCK_SIZE, typename T>
    inline __device__ void reduceSharedData_(int tid, T* s_mins, T* s_maxs, T* s_sums,
                                             T* out_min, T* out_max, T* out_sum, T* out_mean, T scale) {
        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128) {
                s_sums[tid] += s_sums[tid + 128];
                if (s_mins[tid + 128] < s_mins[tid]) s_mins[tid] = s_mins[tid + 128];
                if (s_maxs[tid] < s_maxs[tid + 128]) s_maxs[tid] = s_maxs[tid + 128];
            }
            __syncthreads();

        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64) {
                s_sums[tid] += s_sums[tid + 64];
                if (s_mins[tid + 64] < s_mins[tid]) s_mins[tid] = s_mins[tid + 64];
                if (s_maxs[tid] < s_maxs[tid + 64]) s_maxs[tid] = s_maxs[tid + 64];
            }
            __syncthreads();
        }

        if constexpr (BLOCK_SIZE >= 64) {
            if (tid < 32) {
                warpSumReduce_(s_sums + tid);
                warpMinReduce_(s_mins + tid);
                warpMaxReduce_(s_maxs + tid);
            }
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < 32; ++i) {
                    *s_sums += s_sums[i];
                    if (s_mins[i] < *s_mins) *s_mins = s_mins[i];
                    if (*s_maxs < s_maxs[i]) *s_maxs = s_maxs[i];
                }
            }
            *out_min = *s_mins;
            *out_max = *s_maxs;
            T final_sum = *s_sums;
            if (out_sum)
                *out_sum = final_sum;
            if (out_mean)
                *out_mean = final_sum / scale;
        }
    }

    // Intermediary kernel to reduce large contiguous arrays to max 512 elements.
    namespace Contiguous_ {
        constexpr uint BLOCK_SIZE = 512U;

        template<bool TWO_BY_TWO, typename T>
        __global__ void kernel_(T* input, T* tmp_mins, T* tmp_maxs, T* tmp_sums, uint elements) {
            __shared__ T s_sums[BLOCK_SIZE];
            __shared__ T s_mins[BLOCK_SIZE];
            __shared__ T s_maxs[BLOCK_SIZE];

            T sum = 0, min = *input, max = *input;
            for (uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x; idx < elements;
                 idx += BLOCK_SIZE * 2 * gridDim.x) {
                inPlaceMinMaxSum_(&min, &max, &sum, input[idx]);
                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMaxSum_(&min, &max, &sum, input[idx + BLOCK_SIZE]);
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        inPlaceMinMaxSum_(&min, &max, &sum, input[idx + BLOCK_SIZE]);
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_(threadIdx.x, s_mins, s_maxs, s_sums,
                              tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x, tmp_sums + blockIdx.x);
        }

        uint getBlocks_(size_t elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
            return Noa::Math::min(MAX_BLOCKS, blocks);
        }

        template<typename T>
        void launch_(T* input, T* tmp_mins, T* tmp_maxs, T* tmp_sums,
                     uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE * 2));
            if (two_by_two) {
                kernel_<true><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, tmp_sums, elements);
            } else {
                kernel_<false><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, tmp_sums, elements);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    // Intermediary kernel to reduce large padded arrays to 1-512 elements.
    namespace Padded_ {
        constexpr uint2_t BLOCK_SIZE(32, 16);
        constexpr uint THREADS = BLOCK_SIZE.x * BLOCK_SIZE.y;

        template<bool TWO_BY_TWO, typename T>
        __global__ void kernel_(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, T* tmp_sums, uint2_t shape) {
            uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x;
            __shared__ T s_sums[THREADS];
            __shared__ T s_mins[THREADS];
            __shared__ T s_maxs[THREADS];

            T min = *input, max = *input, sum = 0;
            uint offset;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                offset = row * pitch;
                if constexpr (TWO_BY_TWO) {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                        inPlaceMinMaxSum_(&min, &max, &sum, input[offset + idx]);
                        inPlaceMinMaxSum_(&min, &max, &sum, input[offset + idx + BLOCK_SIZE.x]);
                    }
                } else {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                        inPlaceMinMaxSum_(&min, &max, &sum, input[offset + idx]);
                }
            }
            s_sums[tid] = sum;
            s_mins[tid] = min;
            s_maxs[tid] = max;
            __syncthreads();

            reduceSharedData_(tid, s_mins, s_maxs, s_sums,
                              tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x, tmp_sums + blockIdx.x);
        }

        uint getBlocks_(uint rows) {
            constexpr uint MAX_BLOCKS = 512;
            constexpr uint WARPS = BLOCK_SIZE.y;
            uint blocks = (rows + (WARPS - 1)) / WARPS;
            return Noa::Math::min(blocks, MAX_BLOCKS);
        }

        template<typename T>
        void launch_(T* input, uint pitch, T* tmp_mins, T* tmp_maxs, T* tmp_sums,
                     uint2_t shape, uint blocks, cudaStream_t stream) {
            dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
            bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
            if (two_by_two) {
                kernel_<true><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, tmp_sums, shape);
            } else {
                kernel_<false><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, tmp_sums, shape);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    namespace Final_ {
        // Kernel to reduce small arrays (one array per block). Computes 4 (2 optional) values per batch.
        uint getThreads_(size_t elements) {
            uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2); // compute at least 2 elements.
            return Noa::Math::clamp(threads, 32U, 256U);
        }

        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void kernel_(T* inputs, uint elements,
                                T* output_mins, T* output_maxs, T* output_sums, T* output_means, T scale) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_sums[BLOCK_SIZE];
            __shared__ T s_mins[BLOCK_SIZE];
            __shared__ T s_maxs[BLOCK_SIZE];

            uint batch = blockIdx.x;
            inputs += elements * batch;

            T sum = 0, min = *inputs, max = min;
            for (uint idx = threadIdx.x; idx < elements; idx += BLOCK_SIZE * 2) {
                inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx]);

                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx + BLOCK_SIZE]);
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx + BLOCK_SIZE]);
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_<BLOCK_SIZE>(threadIdx.x, s_mins, s_maxs, s_sums,
                                          output_mins + batch, output_maxs + batch,
                                          output_sums + batch, output_means + batch, scale);
        }

        template<typename T>
        void launch_(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                     size_t elements, T scale, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        kernel_<256, true><<<batches, 256, 0, stream>>>(inputs, elements,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, true><<<batches, 128, 0, stream>>>(inputs, elements,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, true><<<batches, 64, 0, stream>>>(inputs, elements,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, true><<<batches, 32, 0, stream>>>(inputs, elements,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        kernel_<256, false><<<batches, 256, 0, stream>>>(inputs, elements,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, false><<<batches, 128, 0, stream>>>(inputs, elements,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, false><<<batches, 64, 0, stream>>>(inputs, elements,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, false><<<batches, 32, 0, stream>>>(inputs, elements,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }

        // Kernel to reduce the intermediary results (3 input arrays, per block).
        // Computes 4 (2 optional) values per batch.
        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void kernel_(T* tmp_mins, T* tmp_maxs, T* tmp_sums, uint tmps,
                                T* output_mins, T* output_maxs, T* output_sums, T* output_means, T scale) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_sums[BLOCK_SIZE];
            __shared__ T s_mins[BLOCK_SIZE];
            __shared__ T s_maxs[BLOCK_SIZE];

            uint batch = blockIdx.x;
            uint offset = tmps * batch;
            tmp_mins += offset, tmp_maxs += offset, tmp_sums += offset;

            T sum = 0, min = *tmp_mins, max = *tmp_maxs;
            for (uint idx = threadIdx.x; idx < tmps; idx += BLOCK_SIZE * 2) {
                sum += tmp_sums[idx];
                if (tmp_mins[idx] < min) min = tmp_mins[idx];
                if (max < tmp_maxs[idx]) max = tmp_maxs[idx];
                if constexpr (TWO_BY_TWO) {
                    sum += tmp_sums[idx + BLOCK_SIZE];
                    if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                    if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
                } else {
                    if (idx + BLOCK_SIZE < tmps) {
                        sum += tmp_sums[idx + BLOCK_SIZE];
                        if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                        if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
                    }
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_<BLOCK_SIZE>(threadIdx.x, s_mins, s_maxs, s_sums,
                                          output_mins + batch, output_maxs + batch,
                                          output_sums + batch, output_means + batch, scale);
        }

        template<typename T>
        void launch_(T* tmp_mins, T* tmp_maxs, T* tmp_sums,
                     T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                     size_t tmps, T scale, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(tmps % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        kernel_<256, true><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, true><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, true><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, true><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with tmps:{}", threads, tmps);
                }
            } else {
                switch (threads) {
                    case 256:
                        kernel_<256, false><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, false><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, false><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, false><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmps,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with tmps:{}", threads, tmps);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }
}

namespace Noa::CUDA::Math {
    template<typename T>
    void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, uint batches, Stream& stream) {
        if (elements <= 32768 || batches > 16) {
            if (elements) {
                uint threads = Final_::getThreads_(elements);
                auto scale = static_cast<T>(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    T* input = inputs + batch * elements;
                    T* mins = output_mins + batch;
                    T* maxs = output_maxs + batch;
                    T* sums = output_sums == nullptr ? output_sums : output_sums + batch;
                    T* means = output_means == nullptr ? output_means : output_means + batch;
                    uint blocks = Noa::Math::min(batches - batch, 32768U);
                    Final_::launch_(input, mins, maxs, sums, means,
                                    elements, scale, blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream);

        } else {
            uint blocks = Contiguous_::getBlocks_(elements);
            Memory::PtrDevice<T> tmp(blocks * 3 * batches); // all mins, then all maxs, then all sums.
            T* mins, * maxs, * sums;
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                mins = tmp.get() + batch * blocks;
                maxs = mins + batches * blocks;
                sums = maxs + batches * blocks;
                Contiguous_::launch_(input, mins, maxs, sums, elements, blocks, stream.get());
            }
            uint threads = Final_::getThreads_(blocks);
            auto scale = static_cast<T>(elements);
            mins = tmp.get();
            maxs = mins + batches * blocks;
            sums = maxs + batches * blocks;
            Final_::launch_(mins, maxs, sums, output_mins, output_maxs, output_sums, output_means,
                            blocks, scale, batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void minMaxSumMean(T* inputs, size_t pitch_inputs,
                       T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements) {
            Stream::synchronize(stream);
            return;
        }

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d.y);
        Memory::PtrDevice<T> tmp(blocks * 3 * batches); // all mins, then all maxs, then all sums.
        T* mins, * maxs, * sums;
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_inputs * shape_2d.y * batch;
            mins = tmp.get() + batch * blocks;
            maxs = mins + batches * blocks;
            sums = maxs + batches * blocks;
            Padded_::launch_(input, pitch_inputs, mins, maxs, sums, shape_2d, blocks, stream.get());
        }
        uint threads = Final_::getThreads_(blocks);
        auto scale = static_cast<T>(elements);
        mins = tmp.get();
        maxs = mins + batches * blocks;
        sums = maxs + batches * blocks;
        Final_::launch_(mins, maxs, sums, output_mins, output_maxs, output_sums, output_means,
                        blocks, scale, batches, threads, stream.id());
        Stream::synchronize(stream);
    }

    #define INSTANTIATE_MIN_OR_MAX(T)                                                   \
    template void minMaxSumMean<T>(T*, T*, T*, T*, T*, size_t, uint, Stream&);          \
    template void minMaxSumMean<T>(T*, size_t, T*, T*, T*, T*, size3_t, uint, Stream&)

    INSTANTIATE_MIN_OR_MAX(float);
    INSTANTIATE_MIN_OR_MAX(double);
    INSTANTIATE_MIN_OR_MAX(int);
    INSTANTIATE_MIN_OR_MAX(uint);
    INSTANTIATE_MIN_OR_MAX(char);
    INSTANTIATE_MIN_OR_MAX(unsigned char);
}
