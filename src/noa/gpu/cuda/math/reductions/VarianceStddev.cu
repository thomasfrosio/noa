// Implementation of math::varianceStddev() and math::sumMeanVarianceStddev() for contiguous and padded layouts.

#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace {
    using namespace noa;

    // Sum reduces 2 adjacent warps to 1 element.
    // tid should be from 0 to 31. Final sum is at s_data[0].
    template<typename T>
    __device__ void warpSumReduce_(volatile T* s_data, uint tid) {
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

    // Sum reduces the 512 elements in s_data to one element. The block size is assumed to be 512.
    // tid      :   From 0 to 511
    // s_data   :   Shared memory to reduce.
    // tmp_sums :   Returned sum. Usually on device memory.
    template<typename T>
    __device__ void sumReduceSharedMemory_(int tid, T* s_data, T* tmp_sum) {
        if (tid < 256)
            s_data[tid] += s_data[tid + 256];
        __syncthreads();
        if (tid < 128)
            s_data[tid] += s_data[tid + 128];
        __syncthreads();
        if (tid < 64)
            s_data[tid] += s_data[tid + 64];
        __syncthreads();
        if (tid < 32)
            warpSumReduce_(s_data, tid);
        if (tid == 0)
            *tmp_sum = *s_data;
    }

    // Sum reduces the BLOCK_SIZE elements in s_data to one element and compute the resulting
    // variance and stddev using scale.
    //
    // BLOCK_SIZE       :   Elements in s_data to reduce.
    // tid              :   Thread ID. From 0 to BLOCK_SIZE - 1.
    // s_data           :   Data to reduce. Assumed to contain the sum of the squared difference.
    // scale            :   Scale to divide the final sum of the squared difference with.
    // output_variance  :   Returned variance. If nullptr, it is ignored.
    // output_stddev    :   Returned stddev. If nullptr, it is ignored.
    template<int BLOCK_SIZE, typename T>
    __device__ void sumReduceSharedMemory_(int tid, T* s_data, T scale, T* output_variance, T* output_stddev) {
        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128)
                s_data[tid] += s_data[tid + 128];
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64)
                s_data[tid] += s_data[tid + 64];
            __syncthreads();
        }

        // Reduce the last 2 warps to one element.
        if constexpr (BLOCK_SIZE >= 64) {
            if (tid < 32)
                warpSumReduce_(s_data, tid);
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < BLOCK_SIZE; ++i)
                    *s_data += s_data[i];
            }

            // At this point, s_data[0] contains the final sum regardless of BLOCK_SIZE.
            T variance = *s_data / scale;
            if (output_variance)
                *output_variance = variance;
            if (output_stddev)
                *output_stddev = noa::math::sqrt(variance);
        }
    }

    // Intermediary kernel to reduce large contiguous arrays to max 512 elements.
    // Computes sums of squared distances with the mean.
    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 512U;

        // Computes the sum of the squared difference from the mean. Outputs one sum per block.
        // TWO_BY_TWO   :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
        //                  This allows to check for out of bounds every two iterations during the first
        //                  reduction, as opposed to once per iteration.
        // T            :   Data type. float or double.
        // input        :   Input array to reduce. Should be at least @a elements elements.
        // mean         :   Mean used to compute the variance.
        // tmp_sums     :   Returned sums. One per block.
        // elements     :   Number of elements to reduce.
        template<bool TWO_BY_TWO, typename T>
        __global__ void sum_(const T* input, const T* mean, T* tmp_sums, uint elements) {
            __shared__ T s_data[BLOCK_SIZE];

            T tmp, sum_squared_distance = 0;
            for (uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x; idx < elements;
                 idx += BLOCK_SIZE * 2 * gridDim.x) {
                tmp = input[idx] - *mean;
                sum_squared_distance += tmp * tmp;

                if constexpr (TWO_BY_TWO) {
                    tmp = input[idx + BLOCK_SIZE] - *mean;
                    sum_squared_distance += tmp * tmp;
                } else {
                    if (idx + BLOCK_SIZE < elements) {
                        tmp = input[idx + BLOCK_SIZE] - *mean;
                        sum_squared_distance += tmp * tmp;
                    }
                }
            }

            s_data[threadIdx.x] = sum_squared_distance;
            __syncthreads();

            sumReduceSharedMemory_(threadIdx.x, s_data, tmp_sums + blockIdx.x);
        }

        // Given the condition that one thread should reduce at least 2 elements, computes the number of blocks of
        // BLOCK_SIZE threads needed to compute the entire array. The block count is maxed out to 512 since the kernel
        // will loop until the end is reached.
        uint getBlocks_(size_t elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        // Launches the kernel, which outputs one reduced element per block.
        template<typename T>
        void launch_(const T* input, const T* mean, T* tmp_sums, uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE * 2));
            if (two_by_two) {
                sum_<true><<<blocks, BLOCK_SIZE, 0, stream>>>(input, mean, tmp_sums, elements);
            } else {
                sum_<false><<<blocks, BLOCK_SIZE, 0, stream>>>(input, mean, tmp_sums, elements);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    // Intermediary kernel to reduce large padded arrays to max 512 elements.
    // Computes sums of squared distances with the mean.
    namespace padded_ {
        constexpr uint2_t BLOCK_SIZE(32, 16);
        constexpr uint THREADS = BLOCK_SIZE.x * BLOCK_SIZE.y;

        // Computes the sum of the squared difference from the mean. Outputs one sum per block.
        // TWO_BY_TWO   :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
        //                  This allows to check for out of bounds every two iterations during the first
        //                  reduction, as opposed to once per iteration.
        // T            :   Data type. float or double.
        // input        :   Input array to reduce.
        // pitch        :   Pitch of input, in elements.
        // mean         :   Mean used to compute the variance.
        // tmp_sums     :   Returned sums. One per block.
        // elements     :   Number of elements to reduce.
        template<bool TWO_BY_TWO, typename T>
        __global__ void sum_(const T* input, uint pitch, const T* mean, T* tmp_sums, uint2_t shape) {
            uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x; // linear index within the block.
            __shared__ T s_data[THREADS];

            uint offset;
            T tmp, reduced = 0;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                offset = row * pitch;
                if constexpr (TWO_BY_TWO) {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                        tmp = input[offset + idx] - *mean;
                        reduced += tmp * tmp;
                        tmp = input[offset + idx + BLOCK_SIZE.x] - *mean;
                        reduced += tmp * tmp;
                    }
                } else {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                        tmp = input[offset + idx] - *mean;
                        reduced += tmp * tmp;
                    }
                }
            }
            s_data[tid] = reduced;
            __syncthreads();

            sumReduceSharedMemory_(tid, s_data, tmp_sums + blockIdx.x);
        }

        // Returns the number of necessary blocks to compute an array with that many rows.
        uint getBlocks_(uint rows) {
            constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            uint blocks = (rows + (WARPS - 1)) / WARPS;
            return noa::math::min(blocks, MAX_BLOCKS);
        }

        // Launches the kernel, which outputs one element per block.
        template<typename T>
        void launch_(const T* input, uint pitch, const T* mean, T* tmp_sums, uint2_t shape, uint blocks,
                     cudaStream_t stream) {
            dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
            bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
            if (two_by_two) {
                sum_<true><<<blocks, threads, 0, stream>>>(input, pitch, mean, tmp_sums, shape);
            } else {
                sum_<false><<<blocks, threads, 0, stream>>>(input, pitch, mean, tmp_sums, shape);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    namespace final_ {
        // Kernel to reduce small arrays (one array per block). Computes one variance and one stddev per batch.
        uint getThreads_(size_t elements) {
            uint threads = noa::math::nextPowerOf2((elements + 1) / 2);
            return noa::math::clamp(threads, 32U, 256U);
        }

        // For each batch (i.e. block), computes the variance and stddev of a contiguous array.
        // Blocks are independent and can be seen as batches. Each batch reduces an array of elements.
        //
        // BLOCK_SIZE       :   Should be 32, 64, 128 or 256.
        // TWO_BY_TWO       :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
        //                      This allows to check for out of bounds every two iterations during the first
        //                      reduction, as opposed to once per iteration.
        // T                :   Data type. float or double.
        // inputs           :   Input array to reduce. One per block.
        // elements         :   Number of elements to reduce, per input array.
        // means            :   Mean used to compute the variance. One per batch.
        // scale            :   Scale to divide the sum of the squared difference with.
        // output_variances :   Returned variances. One per block. If nullptr, it is ignored.
        // output_stddevs   :   Returned stddevs. One per block. If nullptr, it is ignored.
        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void sumVarStddev_(const T* inputs, uint elements, const T* means, T scale,
                                      T* output_variances, T* output_stddevs) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_data[BLOCK_SIZE];

            uint batch = blockIdx.x;
            inputs += elements * batch;
            means += batch;

            T tmp, sum = 0;
            for (uint idx = threadIdx.x; idx < elements; idx += BLOCK_SIZE * 2) {
                tmp = inputs[idx] - *means;
                sum += tmp * tmp;
                if constexpr (TWO_BY_TWO) {
                    tmp = inputs[idx + BLOCK_SIZE] - *means;
                    sum += tmp * tmp;
                } else {
                    if (idx + BLOCK_SIZE < elements) {
                        tmp = inputs[idx + BLOCK_SIZE] - *means;
                        sum += tmp * tmp;
                    }
                }
            }
            s_data[threadIdx.x] = sum;
            __syncthreads();

            sumReduceSharedMemory_<BLOCK_SIZE>(threadIdx.x, s_data, scale,
                                               output_variances + batch, output_stddevs + batch);
        }

        template<typename T>
        void launch_(const T* input, size_t elements, const T* means, T scale, T* output_variances, T* output_stddevs,
                     uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        sumVarStddev_<256, true><<<batches, 256, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 128:
                        sumVarStddev_<128, true><<<batches, 128, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 64:
                        sumVarStddev_<64, true><<<batches, 64, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 32:
                        sumVarStddev_<32, true><<<batches, 32, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        sumVarStddev_<256, false><<<batches, 256, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 128:
                        sumVarStddev_<128, false><<<batches, 128, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 64:
                        sumVarStddev_<64, false><<<batches, 64, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    case 32:
                        sumVarStddev_<32, false><<<batches, 32, 0, stream>>>(
                                input, elements, means, scale, output_variances, output_stddevs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }

        // Kernel to reduce the intermediary results (tmp_sums is the sum of the squared distances).
        // Computes one variance and one stddev per batch.

        // For each batch (i.e. block), sum reduces a contiguous array and computes the variance and stddev.
        // BLOCK_SIZE       :   Should be 32, 64, 128 or 256.
        // TWO_BY_TWO       :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
        //                      This allows to check for out of bounds every two iterations during the first
        //                      reduction, as opposed to once per iteration.
        // T                :   Data type. float or double.
        // tmp_sums         :   Input array to sum reduce. One per block.
        //                      Should contain the partial sums of the squared distance.
        // elements         :   Number of elements to reduce, per input array.
        // scale            :   Scale to divide the final sum of the squared difference with.
        // output_variances :   Returned variances. One per block. If nullptr, it is ignored.
        // output_stddevs   :   Returned stddevs. One per block. If nullptr, it is ignored.
        template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
        __global__ void sumVarStddev_(const T* tmp_sums, uint elements, T scale,
                                      T* output_variances, T* output_stddevs) {
            static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
            __shared__ T s_data[BLOCK_SIZE];

            uint batch = blockIdx.x;
            tmp_sums += elements * batch;

            T sum = 0;
            for (uint idx = threadIdx.x; idx < elements; idx += BLOCK_SIZE * 2) {
                sum += tmp_sums[idx];
                if constexpr (TWO_BY_TWO) {
                    sum += tmp_sums[idx + BLOCK_SIZE];
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        sum += tmp_sums[idx + BLOCK_SIZE];
                }
            }
            s_data[threadIdx.x] = sum;
            __syncthreads();

            sumReduceSharedMemory_<BLOCK_SIZE>(threadIdx.x, s_data, scale,
                                               output_variances + batch, output_stddevs + batch);
        }

        // Launches the kernel. There's one block per batch.
        template<typename T>
        void launch_(const T* tmp_sums, size_t elements, T scale, T* output_variances, T* output_stddevs,
                     uint batches, uint threads, cudaStream_t stream) {
            bool two_by_two = !(elements % (threads * 2));
            if (two_by_two) {
                switch (threads) {
                    case 256:
                        sumVarStddev_<256, true><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 128:
                        sumVarStddev_<128, true><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 64:
                        sumVarStddev_<64, true><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 32:
                        sumVarStddev_<32, true><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            } else {
                switch (threads) {
                    case 256:
                        sumVarStddev_<256, false><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 128:
                        sumVarStddev_<128, false><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 64:
                        sumVarStddev_<64, false><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    case 32:
                        sumVarStddev_<32, false><<<batches, threads, 0, stream>>>(
                                tmp_sums, elements, scale, output_variances, output_stddevs);
                        break;
                    default:
                        NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                                  "got threads:{}, with elements:{}", threads, elements);
                }
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void varianceStddev(const T* inputs, const T* means, T* output_variances, T* output_stddevs,
                        size_t elements, uint batches, Stream& stream) {
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = final_::getThreads_(elements);
                auto scale = static_cast<T>(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    const T* input = inputs + batch * elements;
                    T* variances = output_variances == nullptr ? output_variances : output_variances + batch;
                    T* stddevs = output_stddevs == nullptr ? output_stddevs : output_stddevs + batch;
                    uint blocks = noa::math::min(batches - batch, 32768U);
                    final_::launch_(input, elements, means + batch, scale, variances, stddevs,
                                    blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream); // not necessary but easier for the user side.

        } else {
            uint blocks = contiguous_::getBlocks_(elements);
            memory::PtrDevice<T> tmp(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + batch * elements;
                T* tmp_sums = tmp.get() + batch * blocks;
                contiguous_::launch_(input, means + batch, tmp_sums, elements, blocks, stream.get());
            }
            uint threads = final_::getThreads_(blocks);
            auto scale = static_cast<T>(elements);
            final_::launch_(tmp.get(), blocks, scale, output_variances, output_stddevs,
                            batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void varianceStddev(const T* inputs, size_t inputs_pitch, const T* means, T* output_variances, T* output_stddevs,
                        size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements) {
            Stream::synchronize(stream);
            return;
        }

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d.y);
        memory::PtrDevice<T> tmp(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + inputs_pitch * shape_2d.y * batch;
            T* tmp_sums = tmp.get() + batch * blocks;
            padded_::launch_(input, inputs_pitch, means + batch, tmp_sums, shape_2d, blocks, stream.get());
        }
        uint threads = final_::getThreads_(blocks);
        auto scale = static_cast<T>(elements);
        final_::launch_(tmp.get(), blocks, scale, output_variances, output_stddevs,
                        batches, threads, stream.id());
        Stream::synchronize(stream);
    }

    template<typename T>
    void sumMeanVarianceStddev(const T* inputs,
                               T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size_t elements, uint batches, Stream& stream) {
        // On the current CUDA backend, there's no real advantage to this function as opposed to directly calling
        // sumMean and varianceStddev. This is mostly to keep a similar API with the CPU backend.
        if (output_means) {
            sumMean(inputs, output_sums, output_means, elements, batches, stream);
            varianceStddev(inputs, output_means, output_variances, output_stddevs, elements, batches, stream);
        } else {
            memory::PtrDevice<T> tmp(batches);
            sumMean(inputs, output_sums, tmp.get(), elements, batches, stream);
            varianceStddev(inputs, tmp.get(), output_variances, output_stddevs, elements, batches, stream);
        }
    }

    template<typename T>
    void sumMeanVarianceStddev(const T* inputs, size_t inputs_pitch,
                               T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size3_t shape, uint batches, Stream& stream) {
        if (output_means) {
            sumMean(inputs, inputs_pitch, output_sums, output_means, shape, batches, stream);
            varianceStddev(inputs, inputs_pitch, output_means, output_variances, output_stddevs,
                           shape, batches, stream);
        } else {
            memory::PtrDevice<T> tmp(batches);
            sumMean(inputs, inputs_pitch, output_sums, tmp.get(), shape, batches, stream);
            varianceStddev(inputs, inputs_pitch, tmp.get(), output_variances, output_stddevs, shape, batches, stream);
        }
    }

    template<typename T>
    void statistics(const T* inputs, T* output_mins, T* output_maxs, T* output_sums,
                    T* output_means, T* output_variances, T* output_stddevs,
                    size_t elements, uint batches, Stream& stream) {
        minMaxSumMean(inputs, output_mins, output_maxs, output_sums, output_means, elements, batches, stream);
        varianceStddev(inputs, output_means, output_variances, output_stddevs, elements, batches, stream);
    }

    template<typename T>
    void statistics(const T* inputs, size_t inputs_pitch,
                    T* output_mins, T* output_maxs, T* output_sums,
                    T* output_means, T* output_variances, T* output_stddevs,
                    size3_t shape, uint batches, Stream& stream) {
        minMaxSumMean(inputs, inputs_pitch,
                      output_mins, output_maxs, output_sums, output_means,
                      shape, batches, stream);
        varianceStddev(inputs, inputs_pitch, output_means, output_variances, output_stddevs, shape, batches, stream);
    }

    #define INSTANTIATE_VARIANCE_STDDEV(T)                                                              \
    template void varianceStddev<T>(const T*, const T*, T*, T*, size_t, uint, Stream&);                 \
    template void varianceStddev<T>(const T*, size_t, const T*, T*, T*, size3_t, uint, Stream&);        \
    template void sumMeanVarianceStddev<T>(const T*, T*, T*, T*, T*, size_t, uint, Stream&);            \
    template void sumMeanVarianceStddev<T>(const T*, size_t, T*, T*, T*, T*, size3_t, uint, Stream&);   \
    template void statistics<T>(const T*, T*, T*, T*, T*, T*, T*, size_t, uint, Stream&);               \
    template void statistics<T>(const T*, size_t, T*, T*, T*, T*, T*, T*, size3_t, uint, Stream&)

    INSTANTIATE_VARIANCE_STDDEV(float);
    INSTANTIATE_VARIANCE_STDDEV(double);
}
