#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/memory/Shared.h"

// Implementation for Math::minMaxSumMean() for contiguous and padded layouts.
// These kernels follow the same logic as the kernels for Math::min(), Math::max() and Math::sum().
// See implementation for Noa::CUDA::Math::sum() for more details, in Min_Max_SumMean.cu.

using namespace Noa;

// COMMON:
namespace Noa::CUDA::Math::Details {
    template<typename T>
    NOA_DEVICE void warpSumReduce(volatile T* s_data, uint tid) {
        T t = s_data[tid];
        t = t + s_data[tid + 32];
        s_data[tid] = t;
        t = t + s_data[tid + 16];
        s_data[tid] = t;
        t = t + s_data[tid + 8];
        s_data[tid] = t;
        t = t + s_data[tid + 4];
        s_data[tid] = t;
        t = t + s_data[tid + 2];
        s_data[tid] = t;
        t = t + s_data[tid + 1];
        s_data[tid] = t;
    }

    template<typename T>
    NOA_DEVICE void warpMinReduce(volatile T* s_data, uint tid) {
        if (s_data[tid + 32] < s_data[tid]) s_data[tid] = s_data[tid + 32];
        if (s_data[tid + 16] < s_data[tid]) s_data[tid] = s_data[tid + 16];
        if (s_data[tid + 8] < s_data[tid]) s_data[tid] = s_data[tid + 8];
        if (s_data[tid + 4] < s_data[tid]) s_data[tid] = s_data[tid + 4];
        if (s_data[tid + 2] < s_data[tid]) s_data[tid] = s_data[tid + 2];
        if (s_data[tid + 1] < s_data[tid]) s_data[tid] = s_data[tid + 1];
    }

    template<typename T>
    NOA_DEVICE void warpMaxReduce(volatile T* s_data, uint tid) {
        if (s_data[tid] < s_data[tid + 32]) s_data[tid] = s_data[tid + 32];
        if (s_data[tid] < s_data[tid + 16]) s_data[tid] = s_data[tid + 16];
        if (s_data[tid] < s_data[tid + 8]) s_data[tid] = s_data[tid + 8];
        if (s_data[tid] < s_data[tid + 4]) s_data[tid] = s_data[tid + 4];
        if (s_data[tid] < s_data[tid + 2]) s_data[tid] = s_data[tid + 2];
        if (s_data[tid] < s_data[tid + 1]) s_data[tid] = s_data[tid + 1];
    }

    template<typename T>
    NOA_ID void inPlaceMinMaxSum(T* current_min, T* current_max, T* current_sum, T candidate) {
        *current_sum += candidate;
        if (candidate < *current_min) *current_min = candidate;
        if (*current_max < candidate) *current_max = candidate;
    }

    template<typename T>
    NOA_ID void reduceSharedData(int tid, T* s_mins, T* s_maxs, T* s_sums,
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

        // Reduces the last 2 warps to one element.
        if constexpr (Noa::Traits::is_complex_v<T>) {
            if (tid == 0) {
                for (int i = 1; i < 64; ++i) {
                    *s_sums += s_sums[i];
                    if (s_mins[i] < *s_mins) *s_mins = s_mins[i];
                    if (*s_maxs < s_maxs[i]) *s_maxs = s_maxs[i];
                }
                *output_sum = *s_sums;
                *output_min = *s_mins;
                *output_max = *s_maxs;
            }
        } else {
            if (tid < 32) {
                warpSumReduce(s_sums, tid);
                warpMinReduce(s_mins, tid);
                warpMaxReduce(s_maxs, tid);
            }
            if (tid == 0) {
                *output_sum = *s_sums;
                *output_min = *s_mins;
                *output_max = *s_maxs;
            }
        }
    }

    template<int BLOCK_SIZE, typename T, typename U>
    NOA_ID void reduceSharedData(int tid, T* s_mins, T* s_maxs, T* s_sums,
                                 T* out_min, T* out_max, T* out_sum, T* out_mean, U scale) {
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
            if constexpr (Noa::Traits::is_complex_v<T>) {
                if (tid == 0) {
                    for (int i = 1; i < 64; ++i) {
                        *s_sums += s_sums[i];
                        if (s_mins[i] < *s_mins) *s_mins = s_mins[i];
                        if (*s_maxs < s_maxs[i]) *s_maxs = s_maxs[i];
                    }
                    *out_sum = *s_sums;
                    *out_min = *s_mins;
                    *out_max = *s_maxs;
                }
            } else {
                if (tid < 32) {
                    warpSumReduce(s_sums, tid);
                    warpMinReduce(s_mins, tid);
                    warpMaxReduce(s_maxs, tid);
                }
            }
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                // Reduce the last warp to one element.
                for (int i = 1; i < BLOCK_SIZE; ++i) {
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
}

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 512U;

    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, T* partial_mins, T* partial_maxs, T* partial_sums, uint elements) {
        uint tid = threadIdx.x;
        T* s_sums = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) * 3 bytes.
        T* s_mins = s_sums + 512;
        T* s_maxs = s_sums + 512 * 2;

        T sum = 0, min = *input, max = *input;
        uint increment = BLOCK_SIZE * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
        while (idx < elements) {
            inPlaceMinMaxSum(&min, &max, &sum, input[idx]);

            if constexpr (TWO_BY_TWO) {
                inPlaceMinMaxSum(&min, &max, &sum, input[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlaceMinMaxSum(&min, &max, &sum, input[idx + BLOCK_SIZE]);

            }
            idx += increment;
        }
        *s_sums = sum;
        *s_mins = min;
        *s_maxs = max;
        __syncthreads();

        reduceSharedData(tid, s_mins, s_maxs, s_sums,
                         partial_mins + blockIdx.x,
                         partial_maxs + blockIdx.x,
                         partial_sums + blockIdx.x);
    }

    NOA_HOST uint getBlocks(size_t elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    template<typename T>
    NOA_IH void launch(T* input, T* mins, T* maxs, T* sums, uint elements, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE * sizeof(T) * 3;
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<true><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, mins, maxs, sums, elements);
        } else {
            kernel<false><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, mins, maxs, sums, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16);

    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, uint pitch, T* partial_mins, T* partial_maxs, T* partial_sums, uint2_t shape) {
        uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x;
        T* s_sums = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) * 3 bytes.
        T* s_mins = s_sums + 512;
        T* s_maxs = s_sums + 512 * 2;

        T min = *input, max = *input, sum = 0;
        uint offset;
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            offset = row * pitch;
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                    inPlaceMinMaxSum(&min, &max, &sum, input[offset + idx]);
                    inPlaceMinMaxSum(&min, &max, &sum, input[offset + idx + BLOCK_SIZE.x]);
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    inPlaceMinMaxSum(&min, &max, &sum, input[offset + idx]);
            }
        }
        *s_sums = sum;
        *s_mins = min;
        *s_maxs = max;
        __syncthreads();

        reduceSharedData(tid, s_mins, s_maxs, s_sums,
                         partial_mins + blockIdx.x,
                         partial_maxs + blockIdx.x,
                         partial_sums + blockIdx.x);
    }

    NOA_HOST uint getBlocks(uint rows) {
        constexpr uint MAX_BLOCKS = 512;
        constexpr uint WARPS = BLOCK_SIZE.y;
        uint blocks = (rows + (WARPS - 1)) / WARPS;
        return Noa::Math::min(blocks, MAX_BLOCKS);
    }

    template<typename T>
    NOA_HOST void launch(T* input, uint pitch, T* mins, T* maxs, T* sums,
                         uint2_t shape, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE.x * BLOCK_SIZE.y * sizeof(T) * 3;
        dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
        bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
        if (two_by_two) {
            kernel<true><<<blocks, threads, bytes_sh, stream>>>(input, pitch, mins, maxs, sums, shape);
        } else {
            kernel<false><<<blocks, threads, bytes_sh, stream>>>(input, pitch, mins, maxs, sums, shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// FINAL REDUCTION:
namespace Noa::CUDA::Math::Details::Final {
    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void kernel(T* inputs, uint elements, T* out_mins, T* out_maxs, T* out_sums, T* out_means, U scale) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_sums = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE * sizeof(T) * 3 bytes.
        T* s_mins = s_sums + BLOCK_SIZE;
        T* s_maxs = s_sums + BLOCK_SIZE * 2;

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        T sum = 0, min = *inputs, max = min;
        uint idx = tid;
        while (idx < elements) {
            inPlaceMinMaxSum(&sum, &min, &max, inputs[idx]);

            if constexpr (TWO_BY_TWO) {
                inPlaceMinMaxSum(&sum, &min, &max, inputs[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlaceMinMaxSum(&sum, &min, &max, inputs[idx + BLOCK_SIZE]);
            }
            idx += BLOCK_SIZE * 2;
        }
        s_sums[tid] = sum;
        s_mins[tid] = min;
        s_maxs[tid] = max;
        __syncthreads();

        reduceSharedData<BLOCK_SIZE>(tid, s_mins, s_maxs, s_sums,
                                     out_mins + batch, out_maxs + batch,
                                     out_sums + batch, out_means + batch, scale);
    }

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void kernel(T* in_mins, T* in_maxs, T* in_sums, uint elements,
                           T* out_mins, T* out_maxs, T* out_sums, T* out_means, U scale) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_sums = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE * sizeof(T) * 3 bytes.
        T* s_mins = s_sums + BLOCK_SIZE;
        T* s_maxs = s_sums + BLOCK_SIZE * 2;

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        uint offset = elements * batch;
        in_mins += offset, in_maxs += offset, in_sums += offset;

        T sum = 0, min = *in_mins, max = *in_maxs;
        uint idx = tid;
        while (idx < elements) {
            sum += in_sums[idx];
            if (in_mins[idx] < min) min = in_mins[idx];
            if (max < in_maxs[idx]) max = in_maxs[idx];

            if constexpr (TWO_BY_TWO) {
                sum += in_sums[idx + BLOCK_SIZE];
                if (in_mins[idx + BLOCK_SIZE] < min) min = in_mins[idx + BLOCK_SIZE];
                if (max < in_maxs[idx + BLOCK_SIZE]) max = in_maxs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements) {
                    sum += in_sums[idx + BLOCK_SIZE];
                    if (in_mins[idx + BLOCK_SIZE] < min) min = in_mins[idx + BLOCK_SIZE];
                    if (max < in_maxs[idx + BLOCK_SIZE]) max = in_maxs[idx + BLOCK_SIZE];
                }
            }
            idx += BLOCK_SIZE * 2;
        }
        s_sums[tid] = sum;
        s_mins[tid] = min;
        s_maxs[tid] = max;
        __syncthreads();

        reduceSharedData<BLOCK_SIZE>(tid, s_mins, s_maxs, s_sums,
                                     out_mins + batch, out_maxs + batch,
                                     out_sums + batch, out_means + batch, scale);
    }

    NOA_HOST uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    template<typename T, typename U>
    NOA_HOST void launch(T* inputs, T* out_mins, T* out_maxs, T* out_sums, T* out_means,
                         size_t elements, U scale, uint batches, uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T) * 3;
        bool two_by_two = !(elements % (threads * 2));

        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                              out_sums, out_means, scale);
                    break;
                case 128:
                    kernel<128, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                              out_sums, out_means, scale);
                    break;
                case 64:
                    kernel<64, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                             out_sums, out_means, scale);
                    break;
                case 32:
                    kernel<32, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                             out_sums, out_means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                               out_sums, out_means, scale);
                    break;
                case 128:
                    kernel<128, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                               out_sums, out_means, scale);
                    break;
                case 64:
                    kernel<64, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                              out_sums, out_means, scale);
                    break;
                case 32:
                    kernel<32, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, out_mins, out_maxs,
                                                                              out_sums, out_means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    NOA_HOST void launch(T* in_mins, T* in_maxs, T* in_sums, T* out_mins, T* out_maxs, T* out_sums, T* out_means,
                         size_t elements, U scale, uint batches, uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T) * 3;
        bool two_by_two = !(elements % (threads * 2));

        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                              out_mins, out_maxs, out_sums, out_means,
                                                                              scale);
                    break;
                case 128:
                    kernel<128, true><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                              out_mins, out_maxs, out_sums, out_means,
                                                                              scale);
                    break;
                case 64:
                    kernel<64, true><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                             out_mins, out_maxs, out_sums, out_means,
                                                                             scale);
                    break;
                case 32:
                    kernel<32, true><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                             out_mins, out_maxs, out_sums, out_means,
                                                                             scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                               out_mins, out_maxs, out_sums, out_means,
                                                                               scale);
                    break;
                case 128:
                    kernel<128, false><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                               out_mins, out_maxs, out_sums, out_means,
                                                                               scale);
                    break;
                case 64:
                    kernel<64, false><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                              out_mins, out_maxs, out_sums, out_means,
                                                                              scale);
                    break;
                case 32:
                    kernel<32, false><<<batches, threads, bytes_sm, stream>>>(in_mins, in_maxs, in_sums, elements,
                                                                              out_mins, out_maxs, out_sums, out_means,
                                                                              scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T>
    void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, uint batches, Stream& stream) {
        if (elements <= 1024 || batches > 16) {
            if (!elements)
                return;

            uint threads = Details::Final::getThreads(elements);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            for (int batch = 0; batch < batches; batch += 32768U) {
                T* input = inputs + batch * elements;
                T* mins = output_mins + batch;
                T* maxs = output_maxs + batch;
                T* sums = output_sums == nullptr ? output_sums : output_sums + batch;
                T* means = output_means == nullptr ? output_means : output_means + batch;
                uint blocks = Noa::Math::min(batches - batch, 32768U);
                Details::Final::launch(input, mins, maxs, sums, means, elements, scale, blocks, threads, stream.id());
            }
        } else {
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> partial_values(blocks * 3 * batches); // all mins, then all maxs, then all sums.
            T* mins;
            T* maxs;
            T* sums;
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                mins = partial_values.get() + batch * blocks * 3;
                maxs = mins + batch * blocks;
                sums = mins + batch * blocks * 2;
                Details::Contiguous::launch(input, mins, maxs, sums, elements, blocks, stream.get());
            }
            uint threads = Details::Final::getThreads(blocks);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            mins = partial_values.get();
            maxs = mins + batches + blocks;
            sums = mins + batches + blocks * 2;
            Details::Final::launch(mins, maxs, sums, output_mins, output_maxs, output_sums, output_means,
                                   blocks, scale, batches, threads, stream.id());
        }
        CUDA::Stream::synchronize(stream);
    }

    template<typename T>
    void minMaxSumMean(T* inputs, size_t pitch_inputs,
                       T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements)
            return;

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> partial_values(blocks * 3 * batches); // all mins, then all maxs, then all sums.
        T* mins;
        T* maxs;
        T* sums;
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_inputs * shape_2d.y * batch;
            mins = partial_values.get() + batch * blocks * 3;
            maxs = mins + batch * blocks;
            sums = mins + batch * blocks * 2;
            Details::Padded::launch(input, pitch_inputs, mins, maxs, sums, shape_2d, blocks, stream.get());
        }
        uint threads = Details::Final::getThreads(blocks);
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
        mins = partial_values.get();
        maxs = mins + batches + blocks;
        sums = mins + batches + blocks * 2;
        Details::Final::launch(mins, maxs, sums, output_mins, output_maxs, output_sums, output_means,
                               blocks, scale, batches, threads, stream.id());
        CUDA::Stream::synchronize(stream);
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
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
