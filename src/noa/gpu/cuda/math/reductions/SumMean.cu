#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/memory/Shared.h"

/*
 * These reduction kernels are adapted from different sources, but mostly come from:
 * https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 * TODO We don't use __shfl_down_sync, __reduce_add_sync or cooperative_groups reduce. Although these could be explored.
 * TODO The last reduction to compute the final sum uses a second kernel launch as synchronization barrier.
 *      1) With cooperative groups, grid synchronization is possible but forces CUDA 9.0 minimum.
 *      2) An atomic operation could directly add the reduction of each block to global memory.
 *         Hardware atomicAdd for double is __CUDA_ARCH__ >= 6, otherwise it should be fine.
 *      3) Warp reduction: ballot? shfl_down? For complex types it is even worse, since it requires many syncs...
 *
 * There's two kernels:
 *  -- sum: This kernel is meant to be used as the first step of the reduction for arrays with more than 1024 elements.
 *          Each block is 512 threads (thus the grid is at least composed of 2 blocks) and computes one reduced sum per
 *          block. These partial sums then need to be reduced to on single element using sumMean.
 *  -- sumMean: This kernel is meant to reduce vectors to one single value. Each block is 32, 64, 128 or 256 threads
 *              and reduces the entire array it was assigned to. The block index corresponds to the batch index.
 */

using namespace Noa;

namespace Noa::CUDA::Math::Details {
    // Given an array with at least 1025 elements and given that one thread reduces at least 2 elements, computes the
    // number of blocks of 512 threads needed to compute the entire array. The block count is maxed out to 512.
    NOA_HOST uint getBlocksForSum(size_t elements) {
        uint threads = 512U;
        uint blocks = (elements + (threads * 2 - 1)) / (threads * 2);
        blocks = Noa::Math::min(512U, blocks);
        return blocks;
    }

    // Given that one thread reduces at least 2 elements, how many threads should we assign to compute the entire array.
    // This is a power of 2 between 32 and 256.
    NOA_HOST uint getThreadsForSumMean(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    // Sum reduces 2 adjacent warps to 1 element. Should be executed by one warp only (tid < 32).
    // Use volatile to prevent caching in registers and re-use the value instead of accessing the memory address.
    // No __syncthreads() necessary since we are in the same warp.
    template<typename T>
    NOA_DEVICE void warpSumReduce(volatile T* s_data, uint tid) {
        T t = s_data[tid]; // https://stackoverflow.com/a/12737522
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
}

namespace Noa::CUDA::Math::Kernels {
    /*
     * This kernel must be launched for arrays with more than 1024 elements. The block size is fixed to 512, which
     * therefore guarantees that there will be at least 2 blocks in the grid. Each block computes a final sum and
     * stores it in output_sums[blockIdx]. The output_sums should then be reduced using the sumFinal kernel.
     */
    template<bool IS_POWER_OF_2, typename T>
    __global__ void sum(T* input, T* output_sums, uint elements) {
        constexpr uint BLOCK_SIZE = 512U;
        T* s_data = Memory::Shared<T>(); // 512 * sizeof(T) bytes.
        uint tid = threadIdx.x;
        uint grid_size = BLOCK_SIZE * 2 * gridDim.x;

        // Firstly, the block reduces the elements to 512 elements. To do so, each one of the 512 threads reads from
        // global memory the element at idx and the element at idx + 512 and adds them to its local sum. Loop until the
        // end by incrementing to the next grid. More blocks will result in a larger grid and therefore fewer elements
        // per thread.
        T sum = 0;
        size_t idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
        while (idx < elements) {
            sum += input[idx];
            // If elements is a power of two, we know we are within the boundaries because elements > 1024.
            if constexpr (IS_POWER_OF_2) {
                sum += input[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements)
                    sum += input[idx + BLOCK_SIZE];
            }
            idx += grid_size;
        }

        s_data[tid] = sum;
        __syncthreads();

        // Once the initial sum is done, parallel reduce the shared array of 512 elements to one single element.
        // Since element should be > 1024, there will be at least 2 final sums (i.e. one per grid).
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
            if (tid < 2)
                for (int i = 0; i < 32; ++i)
                    s_data[tid] += s_data[tid + 32 * tid + i];
            if (tid == 0) {
                s_data[0] += s_data[1];
                output_sums[blockIdx.x] = s_data[0];
            }
        } else {
            if (tid < 32)
                Details::warpSumReduce(s_data, tid);
            if (tid == 0)
                output_sums[blockIdx.x] = s_data[0];
        }
    }

    // This kernel computes the sum and mean of the elements in inputs and stores the result in output_sums.
    // The BLOCK_SIZE should be 32, 64, 128 or 256 and one block will compute one batch. As such, the block
    // index is assumed to be the batch index.
    template<int BLOCK_SIZE, bool IS_POWER_OF_2, typename T, typename U>
    __global__ void sumMean(T* inputs, uint elements, T* output_sums, T* output_means, U scale) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_data = Memory::Shared<T>(); // BLOCK_SIZE * sizeof(T) bytes.

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch; // offset to desired batch

        // Firstly, the block reduces the elements to BLOCK_SIZE elements. To do so, each one of the BLOCK_SIZE threads
        // reads from global memory the element at threadIdx and the element at threadIdx + BLOCK_SIZE and adds them to
        // its local sum. Loop until the end by incrementing by BLOCK_SIZE * 2 to the next elements.
        T thread_sum = 0;
        uint idx = tid;
        while (idx < elements) {
            thread_sum += inputs[idx];
            // IS_POWER_OF_2: true if elements is a power of 2 AND elements >= 64.
            // In this case, we know that idx + BLOCK_SIZE is within boundaries.
            if constexpr (IS_POWER_OF_2) {
                thread_sum += inputs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements)
                    thread_sum += inputs[idx + BLOCK_SIZE];
            }
            idx += BLOCK_SIZE * 2;
        }

        s_data[threadIdx.x] = thread_sum;
        __syncthreads();

        // Once the initial sum is done, parallel reduce the shared array of BLOCK_SIZE elements to one single element.
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
        if constexpr (BLOCK_SIZE >= 64) {
            // If BLOCK_SIZE == 32, warpSumReduce will read out of bounds (even if we allocate at least 64 elements
            // it will read to uninitialized memory), so instead thread 0 will reduce the warp itself (see below).
            if constexpr (Noa::Traits::is_complex_v<T>) {
                if (tid < 2)
                    for (int i = 0; i < 32; ++i)
                        s_data[tid] += s_data[tid + 32 * tid + i];
                if (tid == 0)
                    s_data[0] += s_data[1];
            } else {
                if (tid < 32)
                    Details::warpSumReduce(s_data, tid);
            }
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < elements; ++i)
                    s_data[0] += s_data[i];
            }

            // At this point, s_data[0] contains the final sum regardless of BLOCK_SIZE.
            T final_sum = s_data[0];
            if (output_sums)
                output_sums[batch] = final_sum;
            if (output_means)
                output_means[batch] = final_sum * scale;
        }
    }
}

namespace Noa::CUDA::Math::Details {
    template<typename T>
    NOA_IH void launchSum(T* input, T* partial_sums, uint elements, uint blocks, cudaStream_t stream) {
        if (Noa::isPowerOf2(elements)) {
            Kernels::sum<true><<<blocks, 512U, 512U * sizeof(T), stream>>>(input, partial_sums, elements);
        } else {
            Kernels::sum<false><<<blocks, 512U, 512U * sizeof(T), stream>>>(input, partial_sums, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    NOA_HOST void launchSumMean(T* input,
                                T* sums, T* means,
                                size_t elements, U scale,
                                uint batches, uint threads,
                                cudaStream_t stream) {
        // If there's less than 64 elements, threads = 32 and sumMean needs to check for out of bounds even
        // if elements is a power of two.
        if (elements >= 64 && Noa::isPowerOf2(elements)) {
            switch (threads) {
                case 256:
                    Kernels::sumMean<256, true>
                    <<<batches, threads, 256 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 128:
                    Kernels::sumMean<128, true>
                    <<<batches, threads, 128 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 64:
                    Kernels::sumMean<64, true>
                    <<<batches, threads, 64 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 32:
                    Kernels::sumMean<32, true>
                    <<<batches, threads, 32 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    Kernels::sumMean<256, false>
                    <<<batches, threads, 256 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 128:
                    Kernels::sumMean<128, false>
                    <<<batches, threads, 128 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 64:
                    Kernels::sumMean<64, false>
                    <<<batches, threads, 64 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                case 32:
                    Kernels::sumMean<32, false>
                    <<<batches, threads, 32 * sizeof(T), stream>>>(input, elements, sums, means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace Noa::CUDA::Math {
    template<typename T>
    void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches, Stream& stream) {
        // In the case where there is more than 10 batches and/or there's less than 1280 elements per batch, we assume
        // that it will be better to launch one block of 128 threads per batch with a grid size of at most 32768 blocks.
        // Otherwise, use the traditional
        if (elements <= 1024 || batches > 8) {
            if (!elements)
                return;

            uint threads = Details::getThreadsForSumMean(elements);
            auto scale = 1 / static_cast<Noa::Traits::value_type_t<T>>(elements);
            for (int batch = 0; batch < batches; batch += 32768U) {
                T* input = inputs + batch * elements;
                T* sums = output_sums == nullptr ? output_sums : output_sums + batch;
                T* means = output_means == nullptr ? output_means : output_means + batch;
                uint blocks = Noa::Math::min(batches - batch, 32768U);
                Details::launchSumMean(input, sums, means, elements, scale, blocks, threads, stream.id());
            }
        } else {
            // For arrays with more than 1024 elements, first reduce the array to one element per block.
            // Then use sumMean to reduce these partial sums to one final sum.
            uint blocks = Details::getBlocksForSum(elements);
            PtrDevice<T> partial_sums(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                Details::launchSum(input, partial_sums.get() + batch * blocks, elements, blocks, stream.get());
            }
            // Now the number of blocks is the number of elements per batch.
            uint threads = Details::getThreadsForSumMean(blocks);
            auto scale = 1 / static_cast<Noa::Traits::value_type_t<T>>(elements);
            Details::launchSumMean(partial_sums.get(), output_sums, output_means,
                                   blocks, scale, batches, threads, stream.id());
        }
    }
}

// INSTANTIATE:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_SUM_MEAN(T) \
    template void sumMean<T>(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches, Stream& stream)

    INSTANTIATE_SUM_MEAN(float);
    INSTANTIATE_SUM_MEAN(double);
    INSTANTIATE_SUM_MEAN(int);
    INSTANTIATE_SUM_MEAN(uint);
    INSTANTIATE_SUM_MEAN(cfloat_t);
    INSTANTIATE_SUM_MEAN(cdouble_t);
}
