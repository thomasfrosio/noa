#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/memory/Shared.h"

// Implementation for Math::min(), Math::max() and Math::sumMean() for contiguous and padded layouts.
// The over reductions like Math::minMaxSumMean() or Math::varianceStddev() follow the same logic.

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
 *      3) Warp reduction: ballot? shfl_down?
 */

using namespace Noa;

// COMMON:
namespace Noa::CUDA::Math::Details {
    /**
     * Sum reduces 2 adjacent warps to 1 element.
     * @tparam T                Any integer or floating-point. Cannot be complex.
     * @param[in,out] s_data    Shared memory to reduce. The reduced sum is saved at s_data[0].
     * @param tid               Thread index, from 0 to 31.
     */
    template<typename T>
    NOA_DEVICE void warpSumReduce(volatile T* s_data, uint tid) {
        // No __syncthreads() required since this is executed within a warp.
        // Use volatile to prevent caching in registers and re-use the value instead of accessing the memory address.
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

    template<int REDUCTION, typename T>
    NOA_FD void warpReduce(volatile T* s_data, uint tid) {
        if constexpr (REDUCTION == Details::REDUCTION_SUM) {
            warpSumReduce(s_data, tid);
        } else if constexpr (REDUCTION == Details::REDUCTION_MIN) {
            warpMinReduce(s_data, tid);
        } else if constexpr (REDUCTION == Details::REDUCTION_MAX) {
            warpMaxReduce(s_data, tid);
        }
    }

    template<int REDUCTION, typename T>
    NOA_FD void inPlace(T* current, T candidate) {
        if constexpr (REDUCTION == Details::REDUCTION_SUM) {
            *current += candidate;
        } else if constexpr (REDUCTION == Details::REDUCTION_MIN) {
            if (candidate < *current) *current = candidate; // TODO Noa::Math::min();
        } else if constexpr (REDUCTION == Details::REDUCTION_MAX) {
            if (*current < candidate) *current = candidate;
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }

    // Once the initial reduction is done, parallel reduce the shared array of 512 elements to one single element.
    // Since element should be > 1024, there will be at least 2 final reduced elements (i.e. one per grid).
    template<int REDUCTION, typename T>
    NOA_DEVICE void reduceSharedMemory(int tid, T* s_data, T* output) {
        T* s_data_tid = s_data + tid;
        if (tid < 256)
            inPlace<REDUCTION>(s_data_tid, s_data_tid[256]);
        __syncthreads();
        if (tid < 128)
            inPlace<REDUCTION>(s_data_tid, s_data_tid[128]);
        __syncthreads();
        if (tid < 64)
            inPlace<REDUCTION>(s_data_tid, s_data_tid[64]);
        __syncthreads();

        // Reduces the last 2 warps to one element.
        if constexpr (Noa::Traits::is_complex_v<T>) {
            if (tid == 0) {
                for (int i = 1; i < 64; ++i)
                    inPlace<REDUCTION>(s_data, s_data[i]);
                *output = *s_data;
            }
        } else {
            if (tid < 32)
                warpReduce<REDUCTION>(s_data, tid);
            if (tid == 0)
                *output = *s_data;
        }
    }
}

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 512U; // This should not be changed.

    /**
     * Reduces a contiguous array to some partial reduced elements (one per block).
     * @tparam REDUCTION            Type of reduction: NOA_REDUCTION_SUM, NOA_REDUCTION_MIN or NOA_REDUCTION_MAX.
     * @tparam TWO_BY_TWO           If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                              This allows to check for out of bounds every two iterations during the first
     *                              reduction, as opposed to once per iteration.
     * @tparam T                    Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] input             Input array to reduce. Should be at least @a elements elements.
     * @param[out] outputs          Returned reduced elements. One per block.
     * @param elements              Number of elements to reduce. At least 1025.
     *
     * @warning This kernel must be launched for arrays with more than 1024 elements.
     *          Use Reduce::Final::kernel if it is not the case.
     */
    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, T* outputs, uint elements) {
        uint tid = threadIdx.x;
        T* s_data = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) bytes.
        T* s_data_tid = s_data + tid;

        // First, the block reduces the elements to 512 elements. Each threads reduce 2 elements at a time until the end
        // of the array is reached. More blocks will result in a larger grid and therefore fewer elements per thread.
        T reduced;
        if constexpr (REDUCTION == REDUCTION_SUM)
            reduced = 0;
        else
            reduced = *input;

        uint increment = BLOCK_SIZE * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
        while (idx < elements) {
            inPlace<REDUCTION>(&reduced, input[idx]);

            // If elements is a multiple of BLOCK_SIZE * 2, no need to check for out of bounds.
            if constexpr (TWO_BY_TWO) {
                inPlace<REDUCTION>(&reduced, input[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<REDUCTION>(&reduced, input[idx + BLOCK_SIZE]);
            }
            idx += increment;
        }

        *s_data_tid = reduced;
        __syncthreads();

        reduceSharedMemory<REDUCTION>(tid, s_data, outputs + blockIdx.x);
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
    /// @warning elements should be at least 1025 elements, i.e. at least 2 blocks. Use Reduce::Final otherwise.
    template<int REDUCTION, typename T>
    NOA_IH void launch(T* input, T* partially_reduced, uint elements, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE * sizeof(T);
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<REDUCTION, true><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, partially_reduced, elements);
        } else {
            kernel<REDUCTION, false><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, partially_reduced, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16); // This could be changed to (32, 8) or (32, 4).

    /**
     * Sum reduces an array with a given pitch to some partial sums (one per block).
     * @tparam TWO_BY_TWO       If true, the number of logical elements per row is assumed to be a multiple of 64.
     *                          This allows to check for out of bounds every two iteration, as opposed to once per iteration.
     * @tparam T                Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] input         Input array to reduce. Should be at least `pitch * shape.y` elements.
     * @param pitch             Pitch of @a input, in elements.
     * @param[out] outputs      Returned sum. One per block.
     * @param shape             Logical {fast, medium} shape of @a input. For a 3D array, shape.y should be y * z.
     */
    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, uint pitch, T* outputs, uint2_t shape) {
        uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x; // linear index within the block.
        T* s_data = Memory::Shared<T>::getBlockResource(); // 512 * sizeof(T) bytes.
        T* s_data_tid = s_data + tid;

        // Reduces elements from global memory to 512 elements.
        uint offset;
        T reduced;
        if constexpr (REDUCTION == REDUCTION_SUM)
            reduced = 0;
        else
            reduced = *input;
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            offset = row * pitch; // offset to starting element for that warp.
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) { // jump 2 warps at a time
                    inPlace<REDUCTION>(&reduced, input[offset + idx]);
                    inPlace<REDUCTION>(&reduced, input[offset + idx + BLOCK_SIZE.x]);
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) // jump 1 warp at a time
                    inPlace<REDUCTION>(&reduced, input[offset + idx]);
            }
        }
        *s_data_tid = reduced;
        __syncthreads();

        reduceSharedMemory<REDUCTION>(tid, s_data, outputs + blockIdx.x);
    }

    /// Returns the number of necessary blocks to compute an array with @a rows rows.
    NOA_HOST uint getBlocks(uint rows) {
        constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
        uint blocks = (rows + (WARPS - 1)) / WARPS;
        return Noa::Math::min(blocks, MAX_BLOCKS);
    }

    /// Launches the kernel, which outputs one element per block.
    template<int REDUCTION, typename T>
    NOA_HOST void launch(T* input, uint pitch, T* output_partial, uint2_t shape, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE.x * BLOCK_SIZE.y * sizeof(T);
        dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
        bool two_by_two = !(shape.x % (BLOCK_SIZE.x * 2));
        if (two_by_two) {
            kernel<REDUCTION, true><<<blocks, threads, bytes_sh, stream>>>(input, pitch, output_partial, shape);
        } else {
            kernel<REDUCTION, false><<<blocks, threads, bytes_sh, stream>>>(input, pitch, output_partial, shape);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// FINAL REDUCTION:
namespace Noa::CUDA::Math::Details::Final::SumMean {
    /**
     * For each batch (i.e. block), reduces a contiguous array to one final sum and mean.
     * @details This is optimized for small arrays and often used to compute the sum of the partial sums from
     *          SumContiguous or SumPadded. Blocks are independent and can be seen as batches. Each batch reduces
     *          an array of @a elements elements. The BLOCK_SIZE should be 32, 64, 128 or 256.
     *
     * @tparam TWO_BY_TWO           If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                              This allows to check for out of bounds every two iterations during the first
     *                              reduction, as opposed to once per iteration.
     * @tparam T                    Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * @param[in] input             Input array to reduce. Should be at least @a elements elements.
     * @param elements              Number of elements to reduce, per block.
     * @param[out] output_sums      Returned sum. One per block. If nullptr, ignores it.
     * @param[out] output_means     Returned mean. One per block. If nullptr, ignores it.
     * @param scale                 Value used to compute the mean (sum / value).
     *                              If @a output_means is nullptr, it is ignored.
     */
    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void kernel(T* inputs, uint elements, T* output_sums, T* output_means, U scale) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        T* s_data = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE * sizeof(T) bytes.

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch; // offset to desired batch

        // First, the block reduces the elements to BLOCK_SIZE elements. Each threads sums 2 elements at a time until
        // the end of the array is reached. More blocks results in a larger grid, thus fewer elements per thread.
        T sum = 0;
        uint idx = tid;
        while (idx < elements) {
            sum += inputs[idx];
            if constexpr (TWO_BY_TWO) {
                sum += inputs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements)
                    sum += inputs[idx + BLOCK_SIZE];
            }
            idx += BLOCK_SIZE * 2;
        }
        s_data[tid] = sum;
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

        // Reduce the last 2 warps to one element.
        if constexpr (BLOCK_SIZE >= 64) {
            // If BLOCK_SIZE == 32, warpSumReduce will read out of bounds (even if we allocate at least 64 elements
            // it will read uninitialized memory), so instead thread 0 will reduce the warp itself (see below).
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
                // Reduce the last warp to one element.
                for (int i = 1; i < BLOCK_SIZE; ++i)
                    *s_data += s_data[i];
            }

            // At this point, s_data[0] contains the final sum regardless of BLOCK_SIZE.
            T final_sum = *s_data;
            if (output_sums)
                output_sums[batch] = final_sum;
            if (output_means)
                output_means[batch] = final_sum / scale;
        }
    }

    /// One block works on one array and there's one array per block.
    /// Given that one thread reduces at least 2 elements, how many threads should we assign to compute the entire array.
    /// This is either 32, 64, 128 or 256.
    NOA_HOST uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    /// Launches the kernel, which outputs one sum and mean per batch. There's one block per batch.
    /// See the kernel for more details.
    template<typename T, typename U>
    NOA_HOST void launch(T* input, T* sums, T* means,
                         size_t elements, U scale, uint batches,
                         uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T);
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 128:
                    kernel<128, true><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 64:
                    kernel<64, true><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 32:
                    kernel<32, true><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 128:
                    kernel<128, false><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 64:
                    kernel<64, false><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                case 32:
                    kernel<32, false><<<batches, threads, bytes_sm, stream>>>(input, elements, sums, means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// FINAL REDUCTION:
namespace Noa::CUDA::Math::Details::Final::MinOrMax {
    template<int REDUCTION, int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* inputs, uint elements, T* outputs) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        T* s_data = Memory::Shared<T>::getBlockResource();
        T* s_data_tid = s_data + tid;

        T reduced = *inputs;
        uint idx = tid;
        while (idx < elements) {
            inPlace<REDUCTION>(&reduced, inputs[idx]);
            if constexpr (TWO_BY_TWO) {
                inPlace<REDUCTION>(&reduced, inputs[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<REDUCTION>(&reduced, inputs[idx + BLOCK_SIZE]);
            }
            idx += BLOCK_SIZE * 2;
        }
        *s_data_tid = reduced;
        __syncthreads();

        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128)
                inPlace<REDUCTION>(s_data_tid, s_data[tid + 128]);
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64)
                inPlace<REDUCTION>(s_data_tid, s_data[tid + 64]);
            __syncthreads();
        }

        if constexpr (BLOCK_SIZE >= 64) {
            if constexpr (Noa::Traits::is_complex_v<T>) {
                if (tid == 0)
                    for (int i = 1; i < 64; ++i)
                        inPlace<REDUCTION>(s_data, s_data[i]);
            } else {
                if (tid < 32)
                    warpReduce<REDUCTION>(s_data, tid);
            }
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < BLOCK_SIZE; ++i)
                    *s_data += s_data[i];
            }
            outputs[batch] = *s_data;
        }
    }

    NOA_HOST uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    template<int REDUCTION, typename T>
    NOA_HOST void launch(T* inputs, T* outputs, size_t elements, uint batches, uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T);
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<REDUCTION, 256, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 128:
                    kernel<REDUCTION, 128, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 64:
                    kernel<REDUCTION, 64, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 32:
                    kernel<REDUCTION, 32, true><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<REDUCTION, 256, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 128:
                    kernel<REDUCTION, 128, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 64:
                    kernel<REDUCTION, 64, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
                    break;
                case 32:
                    kernel<REDUCTION, 32, false><<<batches, threads, bytes_sm, stream>>>(inputs, elements, outputs);
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
    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, T* output_values, size_t elements, uint batches, Stream& stream) {
        static_assert(REDUCTION == Details::REDUCTION_MIN || REDUCTION == Details::REDUCTION_MAX);

        // In this case, we assume that it will be faster to directly launch the Final kernel,
        // which is optimized for large batches and small vectors.
        if (elements <= 1024 || batches > 16) {
            if (!elements)
                return;

            uint threads = Details::Final::MinOrMax::getThreads(elements);
            for (int batch = 0; batch < batches; batch += 32768U) {
                T* input = inputs + batch * elements;
                T* mins = output_values + batch;
                uint blocks = Noa::Math::min(batches - batch, 32768U);
                Details::Final::MinOrMax::launch<REDUCTION>(input, mins, elements,
                                                            blocks, threads, stream.id());
            }
        } else {
            // For arrays with more than 1024 elements, first reduce the array to one element per block.
            // Then use the Final reduction to compute the final element.
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> partial_values(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* outputs = partial_values.get() + batch * blocks;
                Details::Contiguous::launch<REDUCTION>(input, outputs, elements, blocks, stream.get());
            }
            // Now the number of blocks is the number of elements per batch.
            uint threads = Details::Final::SumMean::getThreads(blocks);
            Details::Final::MinOrMax::launch<REDUCTION>(partial_values.get(), output_values,
                                                        blocks, batches, threads, stream.id());
        }
        // In the second case, wait before destructing partial_sums. The first case doesn't need a sync, but
        // it is simpler on the user side to always have a sync and say this function always synchronizes the stream.
        CUDA::Stream::synchronize(stream);
    }

    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, size_t pitch_input, T* output_values, size3_t shape, uint batches, Stream& stream) {
        static_assert(REDUCTION == Details::REDUCTION_MIN || REDUCTION == Details::REDUCTION_MAX);
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> partial_values(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_input * shape_2d.y * batch;
            T* outputs = partial_values.get() + batch * blocks;
            Details::Padded::launch<REDUCTION>(input, pitch_input, outputs, shape_2d, blocks, stream.get());
        }
        // Now the number of blocks is the number of elements per batch.
        uint threads = Details::Final::MinOrMax::getThreads(blocks);
        Details::Final::MinOrMax::launch<REDUCTION>(partial_values.get(), output_values,
                                                    blocks, batches, threads, stream.id());
        CUDA::Stream::synchronize(stream); // wait before destructing partial_sums.
    }

    template<typename T>
    void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches, Stream& stream) {
        if (elements <= 1024 || batches > 16) {
            if (!elements)
                return;

            uint threads = Details::Final::SumMean::getThreads(elements);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            for (int batch = 0; batch < batches; batch += 32768U) {
                T* input = inputs + batch * elements;
                T* sums = output_sums == nullptr ? output_sums : output_sums + batch;
                T* means = output_means == nullptr ? output_means : output_means + batch;
                uint blocks = Noa::Math::min(batches - batch, 32768U);
                Details::Final::SumMean::launch(input, sums, means, elements, scale, blocks, threads, stream.id());
            }
        } else {
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> partial_sums(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* sums = partial_sums.get() + batch * blocks;
                Details::Contiguous::launch<Details::REDUCTION_SUM>(input, sums, elements, blocks, stream.get());
            }
            uint threads = Details::Final::SumMean::getThreads(blocks);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            Details::Final::SumMean::launch(partial_sums.get(), output_sums, output_means,
                                            blocks, scale, batches, threads, stream.id());
        }
        CUDA::Stream::synchronize(stream);
    }

    template<typename T>
    void sumMean(T* inputs, size_t pitch_input, T* output_sums, T* output_means,
                 size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements)
            return;

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> partial_sums(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_input * shape_2d.y * batch;
            T* sums = partial_sums.get() + batch * blocks;
            Details::Padded::launch<Details::REDUCTION_SUM>(input, pitch_input, sums, shape_2d, blocks, stream.get());
        }
        uint threads = Details::Final::SumMean::getThreads(blocks);
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
        Details::Final::SumMean::launch(partial_sums.get(), output_sums, output_means,
                                        blocks, scale, batches, threads, stream.id());
        CUDA::Stream::synchronize(stream);
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_SUM_MEAN(T)                                     \
    template void sumMean<T>(T*, T*, T*, size_t, uint, Stream&);        \
    template void sumMean<T>(T*, size_t, T*, T*, size3_t, uint, Stream&)

    INSTANTIATE_SUM_MEAN(float);
    INSTANTIATE_SUM_MEAN(double);
    INSTANTIATE_SUM_MEAN(int);
    INSTANTIATE_SUM_MEAN(uint);
    INSTANTIATE_SUM_MEAN(cfloat_t);
    INSTANTIATE_SUM_MEAN(cdouble_t);

    #define INSTANTIATE_MIN_OR_MAX(T)                                                           \
    template void minOrMax<Details::REDUCTION_MIN, T>(T*, T*, size_t, uint, Stream&);           \
    template void minOrMax<Details::REDUCTION_MIN, T>(T*, size_t, T*, size3_t, uint, Stream&);  \
    template void minOrMax<Details::REDUCTION_MAX, T>(T*, T*, size_t, uint, Stream&);           \
    template void minOrMax<Details::REDUCTION_MAX, T>(T*, size_t, T*, size3_t, uint, Stream&)

    INSTANTIATE_MIN_OR_MAX(float);
    INSTANTIATE_MIN_OR_MAX(double);
    INSTANTIATE_MIN_OR_MAX(int);
    INSTANTIATE_MIN_OR_MAX(uint);
    INSTANTIATE_MIN_OR_MAX(char);
    INSTANTIATE_MIN_OR_MAX(unsigned char);

}
