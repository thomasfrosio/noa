// Implementation for math::minMaxSumMean() for contiguous and padded layouts.

#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace {
    using namespace noa;

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
    inline __device__ void inPlaceMinMaxSum_(T* __restrict__ current_min,
                                             T* __restrict__ current_max,
                                             T* __restrict__ current_sum, T candidate) {
        *current_sum += candidate;
        if (candidate < *current_min) *current_min = candidate;
        if (*current_max < candidate) *current_max = candidate;
    }

    template<typename T>
    inline __device__ void reduceSharedData_(int tid,
                                             T* __restrict__ s_mins, T* __restrict__ s_maxs, T* __restrict__ s_sums,
                                             T* __restrict__ output_min,
                                             T* __restrict__ output_max,
                                             T* __restrict__ output_sum) {
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

    template<int THREADS, typename T>
    inline __device__ void reduceSharedData_(int tid,
                                             T* __restrict__ s_mins, T* __restrict__ s_maxs, T* __restrict__ s_sums,
                                             T* __restrict__ out_min, T* __restrict__ out_max,
                                             T* __restrict__ out_sum, T* __restrict__ out_mean, T scale) {
        if constexpr (THREADS >= 256) {
            if (tid < 128) {
                s_sums[tid] += s_sums[tid + 128];
                if (s_mins[tid + 128] < s_mins[tid]) s_mins[tid] = s_mins[tid + 128];
                if (s_maxs[tid] < s_maxs[tid + 128]) s_maxs[tid] = s_maxs[tid + 128];
            }
            __syncthreads();

        }
        if constexpr (THREADS >= 128) {
            if (tid < 64) {
                s_sums[tid] += s_sums[tid + 64];
                if (s_mins[tid + 64] < s_mins[tid]) s_mins[tid] = s_mins[tid + 64];
                if (s_maxs[tid] < s_maxs[tid + 64]) s_maxs[tid] = s_maxs[tid + 64];
            }
            __syncthreads();
        }

        if constexpr (THREADS >= 64) {
            if (tid < 32) {
                warpSumReduce_(s_sums + tid);
                warpMinReduce_(s_mins + tid);
                warpMaxReduce_(s_maxs + tid);
            }
        }

        if (tid == 0) {
            if constexpr (THREADS == 32) {
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
    namespace contiguous_ {
        constexpr uint THREADS = 512U;

        template<bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS)
        void kernel_(const T* __restrict__ input,
                     T* __restrict__ tmp_mins,
                     T* __restrict__ tmp_maxs,
                     T* __restrict__ tmp_sums, uint elements) {
            __shared__ T s_sums[THREADS];
            __shared__ T s_mins[THREADS];
            __shared__ T s_maxs[THREADS];

            T sum = 0, min = *input, max = *input;
            for (uint idx = blockIdx.x * THREADS * 2 + threadIdx.x; idx < elements;
                 idx += THREADS * 2 * gridDim.x) {
                inPlaceMinMaxSum_(&min, &max, &sum, input[idx]);
                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMaxSum_(&min, &max, &sum, input[idx + THREADS]);
                } else {
                    if (idx + THREADS < elements)
                        inPlaceMinMaxSum_(&min, &max, &sum, input[idx + THREADS]);
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_(threadIdx.x, s_mins, s_maxs, s_sums,
                              tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x, tmp_sums + blockIdx.x);
        }

        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = noa::math::divideUp(elements, THREADS * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        template<typename T>
        void launch_(const T* input, T* tmp_mins, T* tmp_maxs, T* tmp_sums,
                     uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (THREADS * 2));
            if (two_by_two) {
                kernel_<true><<<blocks, THREADS, 0, stream>>>(input, tmp_mins, tmp_maxs, tmp_sums, elements);
            } else {
                kernel_<false><<<blocks, THREADS, 0, stream>>>(input, tmp_mins, tmp_maxs, tmp_sums, elements);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    // Intermediary kernel to reduce large padded arrays to 1-512 elements.
    namespace padded_ {
        constexpr uint2_t THREADS(32, 16);

        template<bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void kernel_(const T* __restrict__ input, uint pitch,
                     T* __restrict__ tmp_mins, T* __restrict__ tmp_maxs, T* __restrict__ tmp_sums, uint2_t shape) {
            uint tid = threadIdx.y * THREADS.x + threadIdx.x;
            __shared__ T s_sums[THREADS.x * THREADS.y];
            __shared__ T s_mins[THREADS.x * THREADS.y];
            __shared__ T s_maxs[THREADS.x * THREADS.y];

            T min = *input, max = *input, sum = 0;
            uint offset;
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                offset = row * pitch;
                if constexpr (TWO_BY_TWO) {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x * 2) {
                        inPlaceMinMaxSum_(&min, &max, &sum, input[offset + idx]);
                        inPlaceMinMaxSum_(&min, &max, &sum, input[offset + idx + THREADS.x]);
                    }
                } else {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
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
            uint blocks = noa::math::divideUp(rows, THREADS.y);
            return noa::math::min(blocks, MAX_BLOCKS);
        }

        template<typename T>
        void launch_(const T* input, uint pitch, T* tmp_mins, T* tmp_maxs, T* tmp_sums,
                     uint2_t shape, uint blocks, cudaStream_t stream) {
            dim3 threads(THREADS.x, THREADS.y);
            bool two_by_two = !(shape.x % (THREADS.x * 2));
            if (two_by_two) {
                kernel_<true><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, tmp_sums, shape);
            } else {
                kernel_<false><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, tmp_sums, shape);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    namespace final_ {
        // Kernel to reduce small arrays (one array per block). Computes 4 (2 optional) values per batch.
        uint getThreads_(size_t elements) {
            uint threads = noa::math::nextPowerOf2((elements + 1) / 2); // compute at least 2 elements.
            return noa::math::clamp(threads, 32U, 256U);
        }

        template<int THREADS, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(256)
        void kernel_(const T* __restrict__ inputs, uint elements,
                     T* __restrict__ output_mins, T* __restrict__ output_maxs,
                     T* __restrict__ output_sums, T* __restrict__ output_means, T scale) {
            static_assert(THREADS >= 32 && THREADS <= 256);
            __shared__ T s_sums[THREADS];
            __shared__ T s_mins[THREADS];
            __shared__ T s_maxs[THREADS];

            uint batch = blockIdx.x;
            inputs += elements * batch;

            T sum = 0, min = *inputs, max = min;
            for (uint idx = threadIdx.x; idx < elements; idx += THREADS * 2) {
                inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx]);

                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx + THREADS]);
                } else {
                    if (idx + THREADS < elements)
                        inPlaceMinMaxSum_(&min, &max, &sum, inputs[idx + THREADS]);
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_<THREADS>(threadIdx.x, s_mins, s_maxs, s_sums,
                                       output_mins + batch, output_maxs + batch,
                                       output_sums + batch, output_means + batch, scale);
        }

        template<typename T>
        void launch_(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                     uint elements, T scale, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        kernel_<256, true><<<batches, 256, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, true><<<batches, 128, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, true><<<batches, 64, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, true><<<batches, 32, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        kernel_<256, false><<<batches, 256, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, false><<<batches, 128, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, false><<<batches, 64, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, false><<<batches, 32, 0, stream>>>(
                                inputs, elements, output_mins, output_maxs, output_sums, output_means, scale);
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
        template<int THREADS, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(256)
        void kernel_(const T* __restrict__ tmp_mins, const T* __restrict__ tmp_maxs,
                     const T* __restrict__ tmp_sums, uint tmp_elements,
                     T* __restrict__ output_mins, T* __restrict__ output_maxs,
                     T* __restrict__ output_sums, T* __restrict__ output_means, T scale) {
            static_assert(THREADS >= 32 && THREADS <= 256);
            __shared__ T s_sums[THREADS];
            __shared__ T s_mins[THREADS];
            __shared__ T s_maxs[THREADS];

            uint batch = blockIdx.x;
            uint offset = tmp_elements * batch;
            tmp_mins += offset, tmp_maxs += offset, tmp_sums += offset;

            T sum = 0, min = *tmp_mins, max = *tmp_maxs;
            for (uint idx = threadIdx.x; idx < tmp_elements; idx += THREADS * 2) {
                sum += tmp_sums[idx];
                if (tmp_mins[idx] < min) min = tmp_mins[idx];
                if (max < tmp_maxs[idx]) max = tmp_maxs[idx];
                if constexpr (TWO_BY_TWO) {
                    sum += tmp_sums[idx + THREADS];
                    if (tmp_mins[idx + THREADS] < min) min = tmp_mins[idx + THREADS];
                    if (max < tmp_maxs[idx + THREADS]) max = tmp_maxs[idx + THREADS];
                } else {
                    if (idx + THREADS < tmp_elements) {
                        sum += tmp_sums[idx + THREADS];
                        if (tmp_mins[idx + THREADS] < min) min = tmp_mins[idx + THREADS];
                        if (max < tmp_maxs[idx + THREADS]) max = tmp_maxs[idx + THREADS];
                    }
                }
            }
            s_sums[threadIdx.x] = sum;
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_<THREADS>(threadIdx.x, s_mins, s_maxs, s_sums,
                                       output_mins + batch, output_maxs + batch,
                                       output_sums + batch, output_means + batch, scale);
        }

        template<typename T>
        void launch_(const T* tmp_mins, const T* tmp_maxs, const T* tmp_sums,
                     T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                     uint tmp_elements, T scale, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(tmp_elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        kernel_<256, true><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, true><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                        output_mins, output_maxs,
                                                                        output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, true><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, true><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                      output_mins, output_maxs,
                                                                      output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with tmp_elements:{}", threads, tmp_elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        kernel_<256, false><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 128:
                        kernel_<128, false><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                         output_mins, output_maxs,
                                                                         output_sums, output_means, scale);
                        break;
                    case 64:
                        kernel_<64, false><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    case 32:
                        kernel_<32, false><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_sums, tmp_elements,
                                                                       output_mins, output_maxs,
                                                                       output_sums, output_means, scale);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with tmp_elements:{}", threads, tmp_elements);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void minMaxSumMean(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (elements <= 32768 || batches > 16) {
            if (elements) {
                uint threads = final_::getThreads_(elements);
                auto scale = static_cast<T>(elements);
                for (size_t batch = 0; batch < batches; batch += 32768U) {
                    const T* input = inputs + batch * elements;
                    T* o_mins = output_mins + batch;
                    T* o_maxs = output_maxs + batch;
                    T* o_sums = output_sums == nullptr ? output_sums : output_sums + batch;
                    T* o_means = output_means == nullptr ? output_means : output_means + batch;
                    uint blocks = noa::math::min(batches - batch, size_t{32768U});
                    final_::launch_(input, o_mins, o_maxs, o_sums, o_means,
                                    elements, scale, blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream);

        } else {
            uint blocks = contiguous_::getBlocks_(elements);
            memory::PtrDevice<T> tmp(blocks * 3 * batches); // all mins, then all maxs, then all sums.
            T* tmp_mins, * tmp_maxs, * tmp_sums;
            for (size_t batch = 0; batch < batches; ++batch) {
                const T* input = inputs + batch * elements;
                tmp_mins = tmp.get() + batch * blocks;
                tmp_maxs = tmp_mins + batches * blocks;
                tmp_sums = tmp_maxs + batches * blocks;
                contiguous_::launch_(input, tmp_mins, tmp_maxs, tmp_sums, elements, blocks, stream.get());
            }
            uint threads = final_::getThreads_(blocks);
            auto scale = static_cast<T>(elements);
            tmp_mins = tmp.get();
            tmp_maxs = tmp_mins + batches * blocks;
            tmp_sums = tmp_maxs + batches * blocks;
            final_::launch_(tmp_mins, tmp_maxs, tmp_sums, output_mins, output_maxs, output_sums, output_means,
                            blocks, scale, batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void minMaxSumMean(const T* inputs, size_t inputs_pitch,
                       T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (!elements)
            return Stream::synchronize(stream);

        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d.y);
        memory::PtrDevice<T> tmp(blocks * 3 * batches); // all mins, then all maxs, then all sums.
        T* tmp_mins, * tmp_maxs, * tmp_sums;
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + inputs_pitch * shape_2d.y * batch;
            tmp_mins = tmp.get() + batch * blocks;
            tmp_maxs = tmp_mins + batches * blocks;
            tmp_sums = tmp_maxs + batches * blocks;
            padded_::launch_(input, inputs_pitch, tmp_mins, tmp_maxs, tmp_sums, shape_2d, blocks, stream.get());
        }
        uint threads = final_::getThreads_(blocks);
        auto scale = static_cast<T>(elements);
        tmp_mins = tmp.get();
        tmp_maxs = tmp_mins + batches * blocks;
        tmp_sums = tmp_maxs + batches * blocks;
        final_::launch_(tmp_mins, tmp_maxs, tmp_sums, output_mins, output_maxs, output_sums, output_means,
                        blocks, scale, batches, threads, stream.id());
        Stream::synchronize(stream);
    }

    #define NOA_INSTANTIATE_MIN_OR_MAX_(T)                                              \
    template void minMaxSumMean<T>(const T*, T*, T*, T*, T*, size_t, size_t, Stream&);  \
    template void minMaxSumMean<T>(const T*, size_t, T*, T*, T*, T*, size3_t, size_t, Stream&)

    NOA_INSTANTIATE_MIN_OR_MAX_(float);
    NOA_INSTANTIATE_MIN_OR_MAX_(double);
    NOA_INSTANTIATE_MIN_OR_MAX_(char);
    NOA_INSTANTIATE_MIN_OR_MAX_(short);
    NOA_INSTANTIATE_MIN_OR_MAX_(int);
    NOA_INSTANTIATE_MIN_OR_MAX_(long);
    NOA_INSTANTIATE_MIN_OR_MAX_(long long);
    NOA_INSTANTIATE_MIN_OR_MAX_(unsigned char);
    NOA_INSTANTIATE_MIN_OR_MAX_(unsigned short);
    NOA_INSTANTIATE_MIN_OR_MAX_(unsigned int);
    NOA_INSTANTIATE_MIN_OR_MAX_(unsigned long);
    NOA_INSTANTIATE_MIN_OR_MAX_(unsigned long long);
}
