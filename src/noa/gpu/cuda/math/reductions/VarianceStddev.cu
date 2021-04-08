#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/memory/Shared.h"

// Implementation for Math::varianceStddev() and Math::sumMeanVarianceStddev() for contiguous and padded layouts.
// These kernels follow the same logic as Noa::CUDA::Math::sum(). See implementation in Min_Max_SumMean.cu for more details.

using namespace Noa;

// COMMON:
namespace Noa::CUDA::Math::Details {
    /// Sum reduces 2 adjacent warps to 1 element. @a tid should be from 0 to 31. Final sum is at @a s_data[0].
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

    /**
     * Sum reduces the 512 elements in @a s_data to one element.
     * @param tid       From 0 to 511
     * @param s_data    Data to reduce. Usually in shared memory.
     * @param output    Returned sum.
     */
    template<typename T>
    NOA_DEVICE void sumReduceSharedMemory(int tid, T* s_data, T* output) {
        if (tid < 256)
            s_data[tid] += s_data[tid + 256];
        __syncthreads();
        if (tid < 128)
            s_data[tid] += s_data[tid + 128];
        __syncthreads();
        if (tid < 64)
            s_data[tid] += s_data[tid + 64];
        __syncthreads();

        // Reduces the last 2 warps to one element.
        if constexpr (Noa::Traits::is_complex_v<T>) {
            if (tid == 0) {
                for (int i = 1; i < 64; ++i)
                    *s_data = s_data[i];
                *output = *s_data;
            }
        } else {
            if (tid < 32)
                warpSumReduce(s_data, tid);
            if (tid == 0)
                *output = *s_data;
        }
    }

    /**
     * Sum reduces the BLOCK_SIZE elements in @a s_data to one element and compute the resulting variance and stddev.
     * @tparam BLOCK_SIZE       Elements in @a s_data to reduce.
     * @param tid               Thread ID. Should be < BLOCK_SIZE.
     * @param s_data            Data to reduce. Assumed to contain the sum of the squared difference.
     * @param scale             Scale to divide the final sum of the squared difference with.
     * @param output_variance   Returned variance. If nullptr, it is ignored.
     * @param output_stddev     Returned stddev. If nullptr, it is ignored.
     */
    template<int BLOCK_SIZE, typename T>
    NOA_DEVICE void sumReduceSharedMemory(int tid, T* s_data, T scale, T* output_variance, T* output_stddev) {
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
            if constexpr (Noa::Traits::is_complex_v<T>) {
                if (tid == 0)
                    for (int i = 1; i < 64; ++i)
                        *s_data += s_data[i];
            } else {
                if (tid < 32)
                    Details::warpSumReduce(s_data, tid);
            }
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < BLOCK_SIZE; ++i)
                    *s_data += s_data[i];
            }

            // At this point, s_data[0] contains the final sum regardless of BLOCK_SIZE.
            T final_sum = *s_data;
            T variance = final_sum / scale;
            if (output_variance)
                *output_variance = variance;
            if (output_stddev)
                *output_stddev = Noa::Math::sqrt(variance);
        }
    }
}

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 512U;

    /**
     * Computes the sum of the squared difference with @a mean. Outputs one sum per block.
     * @tparam TWO_BY_TWO           If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                              This allows to check for out of bounds every two iterations during the first
     *                              reduction, as opposed to once per iteration.
     * @tparam T                    Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] input             Input array to reduce. Should be at least @a elements elements.
     * @param mean                  Mean used to compute the variance.
     * @param[out] outputs          Returned sums. One per block.
     * @param elements              Number of elements to reduce. At least 1025.
     *
     * @warning This kernel must be launched for arrays with more than 1024 elements.
     *          Use Reduce::Final::kernel if it is not the case.
     */
    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, T mean, T* outputs, uint elements) {
        uint tid = threadIdx.x;
        T* s_data = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) bytes.

        uint increment = BLOCK_SIZE * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
        T tmp, sum_squared_distance = 0;
        while (idx < elements) {
            tmp = input[idx] - mean;
            sum_squared_distance += tmp * tmp;

            if constexpr (TWO_BY_TWO) {
                tmp = input[idx + BLOCK_SIZE] - mean;
                sum_squared_distance += tmp * tmp;
            } else {
                if (idx + BLOCK_SIZE < elements) {
                    tmp = input[idx + BLOCK_SIZE] - mean;
                    sum_squared_distance += tmp * tmp;
                }
            }
            idx += increment;
        }

        s_data[tid] = sum_squared_distance;
        __syncthreads();

        sumReduceSharedMemory(tid, s_data, outputs + blockIdx.x);
    }

    /// Given an array with at least 1025 elements (see launch()) and given the condition that one thread should
    /// reduce at least 2 elements, computes the number of blocks of BLOCK_SIZE threads needed to compute the
    /// entire array. The block count is maxed out to 512 since the kernel will loop until the end is reached.
    NOA_HOST uint getBlocks(size_t elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    /// Launches the kernel, which outputs one reduced element per block.
    /// @warning elements should be at least 1025 elements, i.e. at least 2 blocks.
    template<typename T>
    NOA_IH void launch(T* input, T mean, T* output_sums, uint elements, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE * sizeof(T);
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<true><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, mean, output_sums, elements);
        } else {
            kernel<false><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, mean, output_sums, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16);

    /**
     * Computes the sum of the squared difference with @a mean. Outputs one sum per block.
     * @tparam TWO_BY_TWO           If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                              This allows to check for out of bounds every two iterations during the first
     *                              reduction, as opposed to once per iteration.
     * @tparam T                    Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] input             Input array to reduce. Should be at least @a elements elements.
     * @param pitch                 Pitch of @a input, in elements.
     * @param mean                  Mean used to compute the variance.
     * @param[out] outputs          Returned sums. One per block.
     * @param elements              Number of elements to reduce. At least 1025.
     *
     * @warning This kernel must be launched for arrays with more than 1024 elements.
     *          Use Reduce::Final::kernel if it is not the case.
     */
    template<bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, uint pitch, T mean, T* outputs, uint2_t shape) {
        uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x; // linear index within the block.
        T* s_data = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) bytes.

        uint offset;
        T tmp, reduced = 0;
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            offset = row * pitch;
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) {
                    tmp = input[offset + idx] - mean;
                    reduced += tmp * tmp;
                    tmp = input[offset + idx + BLOCK_SIZE.x] - mean;
                    reduced += tmp * tmp;
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    tmp = input[offset + idx] - mean;
                    reduced += tmp * tmp;
                }
            }
        }
        s_data[tid] = reduced;
        __syncthreads();

        sumReduceSharedMemory(tid, s_data, outputs + blockIdx.x);
    }

    /// Returns the number of necessary blocks to compute an array with @a rows rows.
    NOA_HOST uint getBlocks(uint rows) {
        constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
        uint blocks = (rows + (WARPS - 1)) / WARPS;
        return Noa::Math::min(blocks, MAX_BLOCKS);
    }

    /// Launches the kernel, which outputs one element per block.
    template<typename T>
    NOA_HOST void launch(T* input, uint pitch, T mean, T* output_sums,
                         uint2_t shape, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE.x * BLOCK_SIZE.y * sizeof(T);
        dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
        bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
        if (two_by_two) {
            kernel<true><<<blocks, threads, bytes_sh, stream>>>(input, pitch, mean, output_sums, shape);
        } else {
            kernel<false><<<blocks, threads, bytes_sh, stream>>>(input, pitch, mean, output_sums, shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// FINAL REDUCTION:
namespace Noa::CUDA::Math::Details::Final {
    /**
     * For each batch (i.e. block), computes the variance and stddev of a contiguous array.
     * @details Blocks are independent and can be seen as batches. Each batch reduces an array of @a elements elements.
     *          The BLOCK_SIZE should be 32, 64, 128 or 256.
     *          The allocated shared memory should be BLOCK_SIZE * sizeof(T).
     *
     * @tparam TWO_BY_TWO           If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                              This allows to check for out of bounds every two iterations during the first
     *                              reduction, as opposed to once per iteration.
     * @tparam T                    Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] inputs            Input array to reduce. One per block. Should be at least @a elements elements.
     * @param elements              Number of elements to reduce, per input array.
     * @param means                 Mean used to compute the variance. One per batch.
     * @param scale                 Scale to divide the sum of the squared difference with.
     * @param[out] output_variances Returned variances. One per block. If nullptr, it is ignored.
     * @param[out] output_stddevs   Returned stddevs. One per block. If nullptr, it is ignored.
     */
    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void varianceStddev(T* inputs, uint elements, T* means, U scale, T* output_variances,
                                   T* output_stddevs) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_data = Memory::Shared<T>::getBlockResource();

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;
        means += batch;

        T tmp, sum = 0;
        uint idx = tid;
        while (idx < elements) {
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
            idx += BLOCK_SIZE * 2;
        }
        s_data[tid] = sum;
        __syncthreads();

        sumReduceSharedMemory<BLOCK_SIZE>(tid, s_data, scale, output_variances + batch, output_stddevs + batch);
    }

    template<typename T, typename U>
    NOA_HOST void launch(T* input, size_t elements, T* means, U scale, T* output_variances, T* output_stddevs,
                         uint batches, uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T);
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    varianceStddev<256, true><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                      output_variances, output_stddevs);
                    break;
                case 128:
                    varianceStddev<128, true><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                      output_variances, output_stddevs);
                    break;
                case 64:
                    varianceStddev<64, true><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                     output_variances, output_stddevs);
                    break;
                case 32:
                    varianceStddev<32, true><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                     output_variances, output_stddevs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    varianceStddev<256, false><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                       output_variances,
                                                                                       output_stddevs);
                    break;
                case 128:
                    varianceStddev<128, false><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                       output_variances,
                                                                                       output_stddevs);
                    break;
                case 64:
                    varianceStddev<64, false><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                      output_variances, output_stddevs);
                    break;
                case 32:
                    varianceStddev<32, false><<<batches, threads, bytes_sm, stream>>>(input, elements, means, scale,
                                                                                      output_variances, output_stddevs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    /**
     * For each batch (i.e. block), sum reduces a contiguous array and computes the variance and stddev.
     * @details See the overload above for more details.
     *
     * @param[in] inputs            Input array to sum reduce. One per block. Should be at least @a elements elements.
     *                              Should contain the partial sums of the squared distance.
     * @param elements              Number of elements to reduce, per input array.
     * @param scale                 Scale to divide the final sum of the squared difference with.
     * @param[out] output_variances Returned variances. One per block. If nullptr, it is ignored.
     * @param[out] output_stddevs   Returned stddevs. One per block. If nullptr, it is ignored.
     */
    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void varianceStddev(T* inputs, uint elements, U scale, T* output_variances, T* output_stddevs) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_data = Memory::Shared<T>::getBlockResource();

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        T sum = 0;
        uint idx = tid;
        while (idx < elements) {
            sum += inputs[idx];
            if constexpr (TWO_BY_TWO) {
                sum += inputs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements) {
                    sum += inputs[idx + BLOCK_SIZE];
                }
            }
            idx += BLOCK_SIZE * 2;
        }
        s_data[tid] = sum;
        __syncthreads();

        sumReduceSharedMemory<BLOCK_SIZE>(tid, s_data, scale, output_variances + batch, output_stddevs + batch);
    }

    /// Launches the kernel, which outputs one sum and mean per batch. There's one block per batch.
    /// See the kernel for more details.
    template<typename T, typename U>
    NOA_HOST void launch(T* inputs, size_t elements, U scale, T* output_variances, T* output_stddevs,
                         uint batches, uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T);
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    varianceStddev<256, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                      output_variances,
                                                                                      output_stddevs);
                    break;
                case 128:
                    varianceStddev<128, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                      output_variances,
                                                                                      output_stddevs);
                    break;
                case 64:
                    varianceStddev<64, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                     output_variances,
                                                                                     output_stddevs);
                    break;
                case 32:
                    varianceStddev<32, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                     output_variances,
                                                                                     output_stddevs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    varianceStddev<256, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                       output_variances,
                                                                                       output_stddevs);
                    break;
                case 128:
                    varianceStddev<128, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                       output_variances,
                                                                                       output_stddevs);
                    break;
                case 64:
                    varianceStddev<64, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                      output_variances,
                                                                                      output_stddevs);
                    break;
                case 32:
                    varianceStddev<32, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, scale,
                                                                                      output_variances,
                                                                                      output_stddevs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    NOA_HOST uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T>
    void varianceStddev(T* inputs, T* means, T* output_variances, T* output_stddevs,
                        size_t elements, uint batches, Stream& stream) {
        if (elements <= 1024 || batches > 16) {
            if (!elements)
                return;

            uint threads = Details::Final::getThreads(elements);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            for (int batch = 0; batch < batches; batch += 32768U) {
                T* input = inputs + batch * elements;
                T* variances = output_variances == nullptr ? output_variances : output_variances + batch;
                T* stddevs = output_stddevs == nullptr ? output_stddevs : output_stddevs + batch;
                uint blocks = Noa::Math::min(batches - batch, 32768U);
                Details::Final::launch(input, elements, means + batch, scale, variances, stddevs,
                                       blocks, threads, stream.id());
            }
        } else {
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> partial_values(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* mean = means + batch;
                T* out_sums = partial_values.get() + batch * blocks;
                Details::Contiguous::launch(input, mean[batch], out_sums, elements, blocks, stream.get());
            }
            uint threads = Details::Final::getThreads(blocks);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            Details::Final::launch(partial_values.get(), blocks, scale, output_variances, output_stddevs,
                                   batches, threads, stream.id());
        }
        CUDA::Stream::synchronize(stream);
    }

    template<typename T>
    void varianceStddev(T* inputs, size_t pitch_inputs, T* means, T* output_variances, T* output_stddevs,
                        size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements)
            return;

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> partial_sums(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_inputs * shape_2d.y * batch;
            T* tmp_sums = partial_sums.get() + batch * blocks;
            Details::Padded::launch(input, pitch_inputs, means[batch], tmp_sums, shape_2d, blocks, stream.get());
        }
        uint threads = Details::Final::getThreads(blocks);
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
        Details::Final::launch(partial_sums.get(), blocks, scale, output_variances, output_stddevs,
                               batches, threads, stream.id());
        CUDA::Stream::synchronize(stream);
    }

    template<typename T>
    void sumMeanVarianceStddev(T* inputs,
                               T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size_t elements, uint batches, Stream& stream) {
        // On the current CUDA backend, there's no real advantage to this function as opposed to directly calling
        // sumMean and varianceStddev. This is mostly to keep a similar API with the CPU backend.
        if (output_means) {
            sumMean(inputs, output_sums, output_means, elements, batches, stream);
            varianceStddev(inputs, output_means, output_variances, output_stddevs, elements, batches, stream);
        } else {
            PtrDevice<T> tmp(batches);
            sumMean(inputs, output_sums, tmp.get(), elements, batches, stream);
            varianceStddev(inputs, tmp.get(), output_variances, output_stddevs, elements, batches, stream);
        }
    }

    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs, size_t pitch_inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size3_t shape, uint batches, Stream& stream) {
        if (output_means) {
            sumMean(inputs, pitch_inputs, output_sums, output_means, shape, batches, stream);
            varianceStddev(inputs, pitch_inputs, output_means, output_variances, output_stddevs,
                           shape, batches, stream);
        } else {
            PtrDevice<T> tmp(batches);
            sumMean(inputs, pitch_inputs, output_sums, tmp.get(), shape, batches, stream);
            varianceStddev(inputs, pitch_inputs, tmp.get(), output_variances, output_stddevs,
                           shape, batches, stream);
        }
    }

    template<typename T>
    NOA_HOST void statistics(T* inputs, T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size_t elements, uint batches, Stream& stream) {
        minMaxSumMean(inputs, output_mins, output_maxs, output_sums, output_means, elements, batches, stream);
        varianceStddev(inputs, output_means, output_variances, output_stddevs, elements, batches, stream);
    }

    template<typename T>
    NOA_HOST void statistics(T* inputs, size_t pitch_inputs,
                             T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size3_t shape, uint batches, Stream& stream) {
        minMaxSumMean(inputs, pitch_inputs,
                      output_mins, output_maxs, output_sums, output_means,
                      shape, batches, stream);
        varianceStddev(inputs, pitch_inputs, output_means, output_variances, output_stddevs, shape, batches, stream);
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_VARIANCE_STDDEV(T)                                                      \
    template void varianceStddev<T>(T*, T*, T*, T*, size_t, uint, Stream&);                     \
    template void varianceStddev<T>(T*, size_t, T*, T*, T*, size3_t, uint, Stream&);            \
    template void sumMeanVarianceStddev<T>(T*, T*, T*, T*, T*, size_t, uint, Stream&);          \
    template void sumMeanVarianceStddev<T>(T*, size_t, T*, T*, T*, T*, size3_t, uint, Stream&); \
    template void statistics<T>(T*, T*, T*, T*, T*, T*, T*, size_t, uint, Stream&);             \
    template void statistics<T>(T*, size_t, T*, T*, T*, T*, T*, T*, size3_t, uint, Stream&)

    INSTANTIATE_VARIANCE_STDDEV(float);
    INSTANTIATE_VARIANCE_STDDEV(double);
}
