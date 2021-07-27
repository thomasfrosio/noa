// Implementation for math::minMax() for contiguous and padded layouts.

#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace {
    using namespace noa;

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
    __forceinline__ __device__ void inPlaceMinMax_(T* current_min, T* current_max, T candidate) {
        if (candidate < *current_min) *current_min = candidate;
        if (*current_max < candidate) *current_max = candidate;
    }

    template<int BLOCK_SIZE, typename T>
    inline __device__ void reduceSharedData_(int tid, T* s_mins, T* s_maxs, T* output_min, T* output_max) {
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
                    warpMinReduce_(s_mins_tid);
                    warpMaxReduce_(s_maxs_tid);
                }
            }
            if (tid == 0) {
                *output_min = *s_mins;
                *output_max = *s_maxs;
            }
        }
    }

    // Intermediary kernel to reduce large contiguous arrays to max 512 elements.
    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 512U;

        template<bool TWO_BY_TWO, typename T>
        __global__ void reduce_(const T* input, T* tmp_mins, T* tmp_maxs, uint elements) {
            __shared__ T s_mins[BLOCK_SIZE];
            __shared__ T s_maxs[BLOCK_SIZE];

            T min = *input, max = *input;
            for (uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
                 idx < elements;
                 idx += BLOCK_SIZE * 2 * gridDim.x) {
                inPlaceMinMax_(&min, &max, input[idx]);
                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMax_(&min, &max, input[idx + BLOCK_SIZE]);
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        inPlaceMinMax_(&min, &max, input[idx + BLOCK_SIZE]);
                }
            }
            s_mins[threadIdx.x] = min;
            s_maxs[threadIdx.x] = max;
            __syncthreads();

            reduceSharedData_<BLOCK_SIZE>(threadIdx.x, s_mins, s_maxs, tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x);
        }

        uint getBlocks_(size_t elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        template<typename T>
        void launch_(const T* input, T* tmp_mins, T* tmp_maxs, uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE * 2));
            if (two_by_two) {
                reduce_<true><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, elements);
            } else {
                reduce_<false><<<blocks, BLOCK_SIZE, 0, stream>>>(input, tmp_mins, tmp_maxs, elements);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    // Intermediary kernel to reduce large padded arrays to max 512 elements.
    namespace padded_ {
        constexpr uint2_t BLOCK_SIZE(32, 16);
        constexpr uint THREADS = BLOCK_SIZE.x * BLOCK_SIZE.y;

        template<bool TWO_BY_TWO, typename T>
        __global__ void reduce_(const T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape) {
            uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x;
            __shared__ T s_mins[THREADS * 2];
            T* s_maxs = s_mins + THREADS;

            T min = *input, max = *input;
            uint offset;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                offset = row * pitch;
                if constexpr (TWO_BY_TWO) {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                        inPlaceMinMax_(&min, &max, input[offset + idx]);
                        inPlaceMinMax_(&min, &max, input[offset + idx + BLOCK_SIZE.x]);
                    }
                } else {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                        inPlaceMinMax_(&min, &max, input[offset + idx]);
                }
            }
            s_mins[tid] = min;
            s_maxs[tid] = max;
            __syncthreads();

            reduceSharedData_<THREADS>(tid, s_mins, s_maxs, tmp_mins + blockIdx.x, tmp_maxs + blockIdx.x);
        }

        uint getBlocks_(uint rows) {
            constexpr uint MAX_BLOCKS = 512;
            constexpr uint WARPS = BLOCK_SIZE.y;
            uint blocks = (rows + (WARPS - 1)) / WARPS;
            return noa::math::min(blocks, MAX_BLOCKS);
        }

        template<typename T>
        void launch_(const T* input, uint pitch, T* tmp_mins, T* tmp_maxs, uint2_t shape, uint blocks,
                     cudaStream_t stream) {
            dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
            bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
            if (two_by_two) {
                reduce_<true><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, shape);
            } else {
                reduce_<false><<<blocks, threads, 0, stream>>>(input, pitch, tmp_mins, tmp_maxs, shape);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    namespace final_ {
        uint getThreads_(size_t elements) {
            uint threads = noa::math::nextPowerOf2((elements + 1) / 2); // compute at least 2 elements.
            return noa::math::clamp(threads, 32U, 256U);
        }

        // Kernel to reduce small arrays (one array per block). Computes 2 values per batch.
        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void reduce_(const T* inputs, uint elements, T* output_mins, T* output_maxs) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_mins[BLOCK_SIZE * 2];
            T* s_maxs = s_mins + BLOCK_SIZE;

            uint tid = threadIdx.x;
            uint batch = blockIdx.x;
            inputs += elements * batch;

            T min = *inputs, max = min;
            for (uint idx = tid; idx < elements; idx += BLOCK_SIZE * 2) {
                inPlaceMinMax_(&min, &max, inputs[idx]);

                if constexpr (TWO_BY_TWO) {
                    inPlaceMinMax_(&min, &max, inputs[idx + BLOCK_SIZE]);
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        inPlaceMinMax_(&min, &max, inputs[idx + BLOCK_SIZE]);
                }
            }
            s_mins[tid] = min;
            s_maxs[tid] = max;
            __syncthreads();

            reduceSharedData_<BLOCK_SIZE>(tid, s_mins, s_maxs, output_mins + batch, output_maxs + batch);
        }

        template<typename T>
        void launch_(const T* inputs, T* output_mins, T* output_maxs,
                     size_t elements, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        reduce_<256, true><<<batches, 256, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 128:
                        reduce_<128, true><<<batches, 128, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 64:
                        reduce_<64, true><<<batches, 64, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 32:
                        reduce_<32, true><<<batches, 32, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        reduce_<256, false><<<batches, 256, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 128:
                        reduce_<128, false><<<batches, 128, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 64:
                        reduce_<64, false><<<batches, 64, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    case 32:
                        reduce_<32, false><<<batches, 32, 0, stream>>>(inputs, elements, output_mins, output_maxs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }

        // Kernel to reduce the intermediary results (2 input arrays, per block). Computes 2 values per batch.
        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void reduceIntermediary_(const T* tmp_mins, const T* tmp_maxs, uint tmp_elements,
                                            T* output_mins, T* output_maxs) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_mins[BLOCK_SIZE];
            __shared__ T s_maxs[BLOCK_SIZE];

            uint tid = threadIdx.x;
            uint batch = blockIdx.x;
            uint offset = tmp_elements * batch;
            tmp_mins += offset, tmp_maxs += offset;

            T min = *tmp_mins, max = *tmp_maxs;
            for (uint idx = tid; idx < tmp_elements; idx += BLOCK_SIZE * 2) {
                if (tmp_mins[idx] < min) min = tmp_mins[idx];
                if (max < tmp_maxs[idx]) max = tmp_maxs[idx];

                if constexpr (TWO_BY_TWO) {
                    if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                    if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
                } else {
                    if (idx + BLOCK_SIZE < tmp_elements) {
                        if (tmp_mins[idx + BLOCK_SIZE] < min) min = tmp_mins[idx + BLOCK_SIZE];
                        if (max < tmp_maxs[idx + BLOCK_SIZE]) max = tmp_maxs[idx + BLOCK_SIZE];
                    }
                }
            }
            s_mins[tid] = min;
            s_maxs[tid] = max;
            __syncthreads();

            reduceSharedData_<BLOCK_SIZE>(tid, s_mins, s_maxs, output_mins + batch, output_maxs + batch);
        }

        template<typename T>
        void launch_(T* tmp_mins, T* tmp_maxs, T* output_mins, T* output_maxs,
                     size_t tmp_elements, uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(tmp_elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        reduceIntermediary_<256, true><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                    output_mins, output_maxs);
                        break;
                    case 128:
                        reduceIntermediary_<128, true><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                    output_mins, output_maxs);
                        break;
                    case 64:
                        reduceIntermediary_<64, true><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                  output_mins, output_maxs);
                        break;
                    case 32:
                        reduceIntermediary_<32, true><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                  output_mins, output_maxs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with tmp_elements:{}", threads, tmp_elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        reduceIntermediary_<256, false><<<batches, 256, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                     output_mins, output_maxs);
                        break;
                    case 128:
                        reduceIntermediary_<128, false><<<batches, 128, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                     output_mins, output_maxs);
                        break;
                    case 64:
                        reduceIntermediary_<64, false><<<batches, 64, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                   output_mins, output_maxs);
                        break;
                    case 32:
                        reduceIntermediary_<32, false><<<batches, 32, 0, stream>>>(tmp_mins, tmp_maxs, tmp_elements,
                                                                                   output_mins, output_maxs);
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
    void minMax(const T* inputs, T* output_mins, T* output_maxs, size_t elements, uint batches, Stream& stream) {
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = final_::getThreads_(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    const T* input = inputs + batch * elements;
                    T* o_mins = output_mins + batch;
                    T* o_maxs = output_maxs + batch;
                    uint blocks = noa::math::min(batches - batch, 32768U);
                    final_::launch_(input, o_mins, o_maxs, elements, blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream);

        } else {
            uint blocks = contiguous_::getBlocks_(elements);
            memory::PtrDevice<T> tmp(blocks * 2 * batches); // all mins, then all maxs.
            T* tmp_mins, * tmp_maxs;
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + batch * elements;
                tmp_mins = tmp.get() + batch * blocks;
                tmp_maxs = tmp_mins + batches * blocks;
                contiguous_::launch_(input, tmp_mins, tmp_maxs, elements, blocks, stream.get());
            }
            uint threads = final_::getThreads_(blocks);
            tmp_mins = tmp.get();
            tmp_maxs = tmp_mins + batches * blocks;
            final_::launch_(tmp_mins, tmp_maxs, output_mins, output_maxs, blocks, batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void minMax(const T* inputs, size_t inputs_pitch, T* output_mins, T* output_maxs,
                size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements) {
            Stream::synchronize(stream);
            return;
        }

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d.y);
        memory::PtrDevice<T> tmp(blocks * 2 * batches); // all mins, then all maxs.
        T* tmp_mins, * tmp_maxs;
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + inputs_pitch * shape_2d.y * batch;
            tmp_mins = tmp.get() + batch * blocks;
            tmp_maxs = tmp_mins + batches * blocks;
            padded_::launch_(input, inputs_pitch, tmp_mins, tmp_maxs, shape_2d, blocks, stream.get());
        }
        uint threads = final_::getThreads_(blocks);
        tmp_mins = tmp.get();
        tmp_maxs = tmp_mins + batches * blocks;
        final_::launch_(tmp_mins, tmp_maxs, output_mins, output_maxs, blocks, batches, threads, stream.id());
        Stream::synchronize(stream);
    }

    #define NOA_INSTANTIATE_MIN_MAX_(T)                                 \
    template void minMax<T>(const T*, T*, T*, size_t, uint, Stream&);   \
    template void minMax<T>(const T*, size_t, T*, T*, size3_t, uint, Stream&)

    NOA_INSTANTIATE_MIN_MAX_(float);
    NOA_INSTANTIATE_MIN_MAX_(double);
    NOA_INSTANTIATE_MIN_MAX_(char);
    NOA_INSTANTIATE_MIN_MAX_(short);
    NOA_INSTANTIATE_MIN_MAX_(int);
    NOA_INSTANTIATE_MIN_MAX_(long);
    NOA_INSTANTIATE_MIN_MAX_(long long);
    NOA_INSTANTIATE_MIN_MAX_(unsigned char);
    NOA_INSTANTIATE_MIN_MAX_(unsigned short);
    NOA_INSTANTIATE_MIN_MAX_(unsigned int);
    NOA_INSTANTIATE_MIN_MAX_(unsigned long);
    NOA_INSTANTIATE_MIN_MAX_(unsigned long long);
}
