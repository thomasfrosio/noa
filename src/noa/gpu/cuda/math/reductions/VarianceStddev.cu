// Implementation of math::varianceStddev() and math::sumMeanVarianceStddev() for contiguous and padded layouts.

#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
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
    __device__ void sumReduceSharedMemory_(int tid, T* __restrict__ s_data, T* __restrict__ tmp_sum) {
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

    // Sum reduces the THREADS elements in s_data to one element and compute the resulting
    // variance and stddev using scale.
    //
    // THREADS          :   Elements in s_data to reduce.
    // tid              :   Thread ID. From 0 to THREADS - 1.
    // s_data           :   Data to reduce. Assumed to contain the sum of the squared difference.
    // scale            :   Scale to divide the final sum of the squared difference with.
    // output_variance  :   Returned variance. If nullptr, it is ignored.
    // output_stddev    :   Returned stddev. If nullptr, it is ignored.
    template<int THREADS, typename T>
    __device__ void sumReduceSharedMemory_(int tid, T* __restrict__ s_data, T scale,
                                           T* __restrict__ output_variance, T* __restrict__ output_stddev) {
        if constexpr (THREADS >= 256) {
            if (tid < 128)
                s_data[tid] += s_data[tid + 128];
            __syncthreads();
        }
        if constexpr (THREADS >= 128) {
            if (tid < 64)
                s_data[tid] += s_data[tid + 64];
            __syncthreads();
        }

        // Reduce the last 2 warps to one element.
        if constexpr (THREADS >= 64) {
            if (tid < 32)
                warpSumReduce_(s_data, tid);
        }

        if (tid == 0) {
            if constexpr (THREADS == 32) {
                for (int i = 1; i < THREADS; ++i)
                    *s_data += s_data[i];
            }

            // At this point, s_data[0] contains the final sum regardless of THREADS.
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
        constexpr uint THREADS = 512U;

        // Computes the sum of the squared difference from the mean. Outputs one sum per block.
        // TWO_BY_TWO   :   If true, the number of elements is assumed to be a multiple of 2 * THREADS,
        //                  This allows to check for out of bounds every two iterations during the first
        //                  reduction, as opposed to once per iteration.
        // T            :   Data type. float or double.
        // input        :   Input array to reduce. Should be at least @a elements elements.
        // mean         :   Mean used to compute the variance.
        // tmp_sums     :   Returned sums. One per block.
        // elements     :   Number of elements to reduce.
        template<bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS)
        void sum_(const T* __restrict__ input, const T* __restrict__ mean, T* __restrict__ tmp_sums, uint elements) {
            __shared__ T s_data[THREADS];

            T tmp, sum_squared_distance = 0;
            for (uint idx = blockIdx.x * THREADS * 2 + threadIdx.x; idx < elements;
                 idx += THREADS * 2 * gridDim.x) {
                tmp = input[idx] - *mean;
                sum_squared_distance += tmp * tmp;

                if constexpr (TWO_BY_TWO) {
                    tmp = input[idx + THREADS] - *mean;
                    sum_squared_distance += tmp * tmp;
                } else {
                    if (idx + THREADS < elements) {
                        tmp = input[idx + THREADS] - *mean;
                        sum_squared_distance += tmp * tmp;
                    }
                }
            }

            s_data[threadIdx.x] = sum_squared_distance;
            __syncthreads();

            sumReduceSharedMemory_(threadIdx.x, s_data, tmp_sums + blockIdx.x);
        }

        // Given the condition that one thread should reduce at least 2 elements, computes the number of blocks of
        // THREADS threads needed to compute the entire array. The block count is maxed out to 512 since the kernel
        // will loop until the end is reached.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = noa::math::divideUp(elements, THREADS * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        // Launches the kernel, which outputs one reduced element per block.
        template<typename T>
        void launch_(const T* input, const T* mean, T* tmp_sums, uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (THREADS * 2));
            if (two_by_two) {
                sum_<true><<<blocks, THREADS, 0, stream>>>(input, mean, tmp_sums, elements);
            } else {
                sum_<false><<<blocks, THREADS, 0, stream>>>(input, mean, tmp_sums, elements);
            }
            NOA_THROW_IF(cudaGetLastError());
        }
    }

    // Intermediary kernel to reduce large padded arrays to max 512 elements.
    // Computes sums of squared distances with the mean.
    namespace padded_ {
        constexpr uint2_t THREADS(32, 16);

        // Computes the sum of the squared difference from the mean. Outputs one sum per block.
        // TWO_BY_TWO   :   If true, the number of elements is assumed to be a multiple of 2 * THREADS,
        //                  This allows to check for out of bounds every two iterations during the first
        //                  reduction, as opposed to once per iteration.
        // T            :   Data type. float or double.
        // input        :   Input array to reduce.
        // pitch        :   Pitch of input, in elements.
        // mean         :   Mean used to compute the variance.
        // tmp_sums     :   Returned sums. One per block.
        // elements     :   Number of elements to reduce.
        template<bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void sum_(const T* __restrict__ input, uint pitch,
                  const T* __restrict__ mean, T* __restrict__ tmp_sums, uint2_t shape) {
            uint tid = threadIdx.y * THREADS.x + threadIdx.x; // linear index within the block.
            __shared__ T s_data[THREADS.x * THREADS.y];

            uint offset;
            T tmp, reduced = 0;
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                offset = row * pitch;
                if constexpr (TWO_BY_TWO) {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x * 2) {
                        tmp = input[offset + idx] - *mean;
                        reduced += tmp * tmp;
                        tmp = input[offset + idx + THREADS.x] - *mean;
                        reduced += tmp * tmp;
                    }
                } else {
                    for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x) {
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
            // every warp processes at least one row.
            uint blocks = noa::math::divideUp(rows, THREADS.y);
            return noa::math::min(blocks, MAX_BLOCKS);
        }

        // Launches the kernel, which outputs one element per block.
        template<typename T>
        void launch_(const T* input, uint pitch, const T* mean, T* tmp_sums, uint2_t shape, uint blocks,
                     cudaStream_t stream) {
            dim3 threads(THREADS.x, THREADS.y);
            bool two_by_two = !(shape.x % (THREADS.x * 2));
            if (two_by_two) {
                sum_<true><<<blocks, threads, 0, stream>>>(input, pitch, mean, tmp_sums, shape);
            } else {
                sum_<false><<<blocks, threads, 0, stream>>>(input, pitch, mean, tmp_sums, shape);
            }
            NOA_THROW_IF(cudaGetLastError());
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
        // THREADS       :   Should be 32, 64, 128 or 256.
        // TWO_BY_TWO       :   If true, the number of elements is assumed to be a multiple of 2 * THREADS,
        //                      This allows to check for out of bounds every two iterations during the first
        //                      reduction, as opposed to once per iteration.
        // T                :   Data type. float or double.
        // inputs           :   Input array to reduce. One per block.
        // elements         :   Number of elements to reduce, per input array.
        // means            :   Mean used to compute the variance. One per batch.
        // scale            :   Scale to divide the sum of the squared difference with.
        // output_variances :   Returned variances. One per block. If nullptr, it is ignored.
        // output_stddevs   :   Returned stddevs. One per block. If nullptr, it is ignored.
        template<int THREADS, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(256)
        void sumVarStddev_(const T* __restrict__ inputs, uint elements, const T* __restrict__ means, T scale,
                           T* __restrict__ output_variances, T* __restrict__ output_stddevs) {
            static_assert(THREADS >= 32 && THREADS <= 256);
            __shared__ T s_data[THREADS];

            uint batch = blockIdx.x;
            inputs += elements * batch;
            means += batch;

            T tmp, sum = 0;
            for (uint idx = threadIdx.x; idx < elements; idx += THREADS * 2) {
                tmp = inputs[idx] - *means;
                sum += tmp * tmp;
                if constexpr (TWO_BY_TWO) {
                    tmp = inputs[idx + THREADS] - *means;
                    sum += tmp * tmp;
                } else {
                    if (idx + THREADS < elements) {
                        tmp = inputs[idx + THREADS] - *means;
                        sum += tmp * tmp;
                    }
                }
            }
            s_data[threadIdx.x] = sum;
            __syncthreads();

            sumReduceSharedMemory_<THREADS>(threadIdx.x, s_data, scale,
                                            output_variances + batch, output_stddevs + batch);
        }

        template<typename T>
        void launch_(const T* input, uint elements, const T* means, T scale, T* output_variances, T* output_stddevs,
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
            NOA_THROW_IF(cudaGetLastError());
        }

        // Kernel to reduce the intermediary results (tmp_sums is the sum of the squared distances).
        // Computes one variance and one stddev per batch.

        // For each batch (i.e. block), sum reduces a contiguous array and computes the variance and stddev.
        // THREADS       :   Should be 32, 64, 128 or 256.
        // TWO_BY_TWO       :   If true, the number of elements is assumed to be a multiple of 2 * THREADS,
        //                      This allows to check for out of bounds every two iterations during the first
        //                      reduction, as opposed to once per iteration.
        // T                :   Data type. float or double.
        // tmp_sums         :   Input array to sum reduce. One per block.
        //                      Should contain the partial sums of the squared distance.
        // elements         :   Number of elements to reduce, per input array.
        // scale            :   Scale to divide the final sum of the squared difference with.
        // output_variances :   Returned variances. One per block. If nullptr, it is ignored.
        // output_stddevs   :   Returned stddevs. One per block. If nullptr, it is ignored.
        template<int THREADS, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(256)
        void sumVarStddev_(const T* __restrict__ tmp_sums, uint elements, T scale,
                           T* __restrict__ output_variances, T* __restrict__ output_stddevs) {
            static_assert(THREADS >= 32 && THREADS <= 256);
            __shared__ T s_data[THREADS];

            uint batch = blockIdx.x;
            tmp_sums += elements * batch;

            T sum = 0;
            for (uint idx = threadIdx.x; idx < elements; idx += THREADS * 2) {
                sum += tmp_sums[idx];
                if constexpr (TWO_BY_TWO) {
                    sum += tmp_sums[idx + THREADS];
                } else {
                    if (idx + THREADS < elements)
                        sum += tmp_sums[idx + THREADS];
                }
            }
            s_data[threadIdx.x] = sum;
            __syncthreads();

            sumReduceSharedMemory_<THREADS>(threadIdx.x, s_data, scale,
                                            output_variances + batch, output_stddevs + batch);
        }

        // Launches the kernel. There's one block per batch.
        template<typename T>
        void launch_(const T* tmp_sums, uint elements, T scale, T* output_variances, T* output_stddevs,
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
            NOA_THROW_IF(cudaGetLastError());
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void varianceStddev(const T* inputs, const T* means, T* output_variances, T* output_stddevs,
                        size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = final_::getThreads_(elements);
                auto scale = static_cast<T>(elements);
                for (size_t batch = 0; batch < batches; batch += 32768U) {
                    const T* input = inputs + batch * elements;
                    T* variances = output_variances == nullptr ? output_variances : output_variances + batch;
                    T* stddevs = output_stddevs == nullptr ? output_stddevs : output_stddevs + batch;
                    uint blocks = noa::math::min(batches - batch, size_t{32768});
                    final_::launch_(input, elements, means + batch, scale, variances, stddevs,
                                    blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream); // not necessary but easier for the user side.

        } else {
            uint blocks = contiguous_::getBlocks_(elements);
            memory::PtrDevice<T> tmp(blocks * batches);
            for (size_t batch = 0; batch < batches; ++batch) {
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
                        size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        size_t elements = noa::elements(shape);
        if (!elements)
            return Stream::synchronize(stream);

        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d.y);
        memory::PtrDevice<T> tmp(blocks * batches);
        for (size_t batch = 0; batch < batches; ++batch) {
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
                               size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
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
                               size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
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
                    size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        minMaxSumMean(inputs, output_mins, output_maxs, output_sums, output_means, elements, batches, stream);
        varianceStddev(inputs, output_means, output_variances, output_stddevs, elements, batches, stream);
    }

    template<typename T>
    void statistics(const T* inputs, size_t inputs_pitch,
                    T* output_mins, T* output_maxs, T* output_sums,
                    T* output_means, T* output_variances, T* output_stddevs,
                    size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        minMaxSumMean(inputs, inputs_pitch,
                      output_mins, output_maxs, output_sums, output_means,
                      shape, batches, stream);
        varianceStddev(inputs, inputs_pitch, output_means, output_variances, output_stddevs, shape, batches, stream);
    }

    #define NOA_INSTANTIATE_VARIANCE_STDDEV_(T)                                                           \
    template void varianceStddev<T>(const T*, const T*, T*, T*, size_t, size_t, Stream&);                 \
    template void varianceStddev<T>(const T*, size_t, const T*, T*, T*, size3_t, size_t, Stream&);        \
    template void sumMeanVarianceStddev<T>(const T*, T*, T*, T*, T*, size_t, size_t, Stream&);            \
    template void sumMeanVarianceStddev<T>(const T*, size_t, T*, T*, T*, T*, size3_t, size_t, Stream&);   \
    template void statistics<T>(const T*, T*, T*, T*, T*, T*, T*, size_t, size_t, Stream&);               \
    template void statistics<T>(const T*, size_t, T*, T*, T*, T*, T*, T*, size3_t, size_t, Stream&)

    NOA_INSTANTIATE_VARIANCE_STDDEV_(float);
    NOA_INSTANTIATE_VARIANCE_STDDEV_(double);
}
