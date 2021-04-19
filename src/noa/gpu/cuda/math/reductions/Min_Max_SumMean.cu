// Implementation for Math::min(), Math::max() and Math::sumMean() for contiguous and padded layouts.

#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"
#include "noa/gpu/cuda/memory/Shared.h"

// -------------------------- //
// -- FORWARD DECLARATIONS -- //
// -------------------------- //

// Intermediary kernel to reduce large contiguous arrays to max 512 elements. Used by min(), max() and sumMean().
namespace Noa::CUDA::Math::Details::Contiguous {
    static uint getBlocks(uint elements);

    template<int REDUCTION, typename T>
    static void launch(T* input, T* tmp_outputs, uint elements, uint blocks, cudaStream_t stream);

    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* input, T* tmp_outputs, uint elements);
}

// Intermediary kernel to reduce large padded arrays to max 512 elements.
namespace Noa::CUDA::Math::Details::Padded {
    static uint getBlocks(uint rows);

    template<int REDUCTION, typename T>
    static void launch(T* input, uint pitch, T* output_partial, uint2_t shape, uint blocks, cudaStream_t stream);

    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* input, uint pitch, T* outputs, uint2_t shape);
}

// Kernel to sum reduce small arrays (one array per block). Computes one value per batch.
namespace Noa::CUDA::Math::Details::Final::SumMean {
    static uint getThreads(size_t elements);

    template<typename T, typename U>
    static void launch(T* input, T* sums, T* means, size_t elements, U scale,
                       uint batches, uint threads, cudaStream_t stream);

    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    static __global__ void kernel(T* inputs, uint elements, T* output_sums, T* output_means, U scale);
}

// Kernel to min or max reduce small arrays (one array per block). Computes one value per batch.
namespace Noa::CUDA::Math::Details::Final::MinOrMax {
    static uint getThreads(size_t elements);

    template<int REDUCTION, typename T>
    static void launch(T* inputs, T* outputs, size_t elements, uint batches, uint threads, cudaStream_t stream);

    template<int REDUCTION, int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* inputs, uint elements, T* outputs);
}

// ----------------- //
// -- DEFINITIONS -- //
// ----------------- //

namespace Noa::CUDA::Math::Details {
    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, T* output_values, size_t elements, uint batches, Stream& stream) {
        static_assert(REDUCTION == Details::REDUCTION_MIN || REDUCTION == Details::REDUCTION_MAX);
        // On my setup, anything below 65536 elements is faster if reduced directly by the Final::SumMean kernel.
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = Details::Final::MinOrMax::getThreads(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    T* input = inputs + batch * elements;
                    T* mins = output_values + batch;
                    uint blocks = Noa::Math::min(batches - batch, 32768U);
                    Details::Final::MinOrMax::launch<REDUCTION>(input, mins, elements,
                                                                blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream); // not necessary, but it is simpler on the user side to always have a sync.

        } else {
            // First reduce the array to one element per block.
            // Then use the Final reduction to compute the final element.
            uint blocks = Details::Contiguous::getBlocks(elements); // at least 65 blocks.
            PtrDevice<T> tmp_intermediary(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* outputs = tmp_intermediary.get() + batch * blocks;
                Details::Contiguous::launch<REDUCTION>(input, outputs, elements, blocks, stream.get());
            }
            // Now the number of blocks is the number of elements per batch.
            uint threads = Details::Final::MinOrMax::getThreads(blocks);
            Details::Final::MinOrMax::launch<REDUCTION>(tmp_intermediary.get(), output_values,
                                                        blocks, batches, threads, stream.id());
            Stream::synchronize(stream); // wait before destructing tmp_intermediary.
        }
    }

    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, size_t pitch_input, T* output_values, size3_t shape, uint batches, Stream& stream) {
        static_assert(REDUCTION == Details::REDUCTION_MIN || REDUCTION == Details::REDUCTION_MAX);
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> tmp_intermediary(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_input * shape_2d.y * batch;
            T* outputs = tmp_intermediary.get() + batch * blocks;
            Details::Padded::launch<REDUCTION>(input, pitch_input, outputs, shape_2d, blocks, stream.get());
        }
        // Now the number of blocks is the number of elements per batch.
        uint threads = Details::Final::MinOrMax::getThreads(blocks);
        Details::Final::MinOrMax::launch<REDUCTION>(tmp_intermediary.get(), output_values,
                                                    blocks, batches, threads, stream.id());
        Stream::synchronize(stream); // wait before destructing tmp_intermediary.
    }
}

namespace Noa::CUDA::Math {
    template<typename T>
    void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches, Stream& stream) {
        if (elements <= 65536 || batches > 16) {
            if (elements) {
                uint threads = Details::Final::SumMean::getThreads(elements);
                auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
                for (int batch = 0; batch < batches; batch += 32768U) {
                    T* input = inputs + batch * elements;
                    T* sums = output_sums == nullptr ? output_sums : output_sums + batch;
                    T* means = output_means == nullptr ? output_means : output_means + batch;
                    uint blocks = Noa::Math::min(batches - batch, 32768U);
                    Details::Final::SumMean::launch(input, sums, means, elements, scale, blocks, threads, stream.id());
                }
            }
            Stream::synchronize(stream);

        } else {
            uint blocks = Details::Contiguous::getBlocks(elements);
            PtrDevice<T> tmp_sums(blocks * batches);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* tmp = tmp_sums.get() + batch * blocks;
                Details::Contiguous::launch<Details::REDUCTION_SUM>(input, tmp, elements, blocks, stream.get());
            }
            uint threads = Details::Final::SumMean::getThreads(blocks);
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
            Details::Final::SumMean::launch(tmp_sums.get(), output_sums, output_means,
                                            blocks, scale, batches, threads, stream.id());
            Stream::synchronize(stream);
        }
    }

    template<typename T>
    void sumMean(T* inputs, size_t pitch_input, T* output_sums, T* output_means,
                 size3_t shape, uint batches, Stream& stream) {
        size_t elements = getElements(shape);
        if (!elements) {
            Stream::synchronize(stream);
            return;
        }

        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d.y);
        PtrDevice<T> tmp_sums(blocks * batches);
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + pitch_input * shape_2d.y * batch;
            T* tmp = tmp_sums.get() + batch * blocks;
            Details::Padded::launch<Details::REDUCTION_SUM>(input, pitch_input, tmp, shape_2d, blocks, stream.get());
        }
        uint threads = Details::Final::SumMean::getThreads(blocks);
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
        Details::Final::SumMean::launch(tmp_sums.get(), output_sums, output_means,
                                        blocks, scale, batches, threads, stream.id());
        Stream::synchronize(stream);
    }
}

// -------------------- //
// -- IMPLEMENTATION -- //
// -------------------- //

// COMMON:
namespace Noa::CUDA::Math::Details {
    // Sum reduces 2 adjacent warps to 1 element.
    // T:       Any integer or floating-point. Cannot be complex.
    // s_data:  Shared memory to reduce. The reduced sum is saved at s_data[0].
    // tid      Thread index, from 0 to 31.
    template<typename T>
    static NOA_DEVICE void warpSumReduce(volatile T* s_data_tid) {
        // No __syncthreads() required since this is executed within a warp.
        // Use volatile to prevent caching in registers and re-use the value instead of accessing the memory address.
        T t = *s_data_tid; // https://stackoverflow.com/a/12737522
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

    template<int REDUCTION, typename T>
    static NOA_FD void warpReduce(volatile T* s_data_tid) {
        if constexpr (REDUCTION == Details::REDUCTION_SUM) {
            warpSumReduce(s_data_tid);
        } else if constexpr (REDUCTION == Details::REDUCTION_MIN) {
            warpMinReduce(s_data_tid);
        } else if constexpr (REDUCTION == Details::REDUCTION_MAX) {
            warpMaxReduce(s_data_tid);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }

    template<int REDUCTION, typename T>
    static NOA_FD void inPlace(T* current, T candidate) {
        if constexpr (REDUCTION == Details::REDUCTION_SUM) {
            *current += candidate;
        } else if constexpr (REDUCTION == Details::REDUCTION_MIN) {
            if (candidate < *current) *current = candidate;
        } else if constexpr (REDUCTION == Details::REDUCTION_MAX) {
            if (*current < candidate) *current = candidate;
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }

    // ! Block size should be 512 !
    // Once the initial reduction is done, parallel reduce the shared array of 512 elements to one single element.
    template<int REDUCTION, typename T>
    static NOA_DEVICE void reduceSharedMemory(int tid, T* s_data, T* output) {
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
                warpReduce<REDUCTION>(s_data_tid);
            if (tid == 0)
                *output = *s_data;
        }
    }
}

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 512U;

    /*
     * Reduces a contiguous array to some partial reduced elements (one per block).
     * REDUCTION    :   Type of reduction: NOA_REDUCTION_SUM, NOA_REDUCTION_MIN or NOA_REDUCTION_MAX.
     * TWO_BY_TWO   :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE, This allows
     *                  to check for out of bounds every two iterations during the first reduction, as opposed to
     *                  once per iteration.
     * T            :   Data type. Usually (u)int, float, double, cfloat_t or cdouble_t.
     * input        :   Input array to reduce. Should be at least @a elements elements.
     * outputs      :   Returned reduced elements. One per block.
     * elements     :   Number of elements to reduce.
     */
    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, T* tmp_outputs, uint elements) {
        uint tid = threadIdx.x;
        T* s_data = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE * sizeof(T) bytes.
        T* s_data_tid = s_data + tid;

        T reduced;
        if constexpr (REDUCTION == REDUCTION_SUM)
            reduced = 0;
        else
            reduced = *input;

        // First, the block reduces the elements to 512 elements.
        // Each threads reduce 2 elements at a time until the end of the array is reached.
        // More blocks will result in a larger grid and therefore fewer elements per thread.
        for (uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x; idx < elements; idx += BLOCK_SIZE * 2 * gridDim.x) {
            inPlace<REDUCTION>(&reduced, input[idx]);
            if constexpr (TWO_BY_TWO) {
                inPlace<REDUCTION>(&reduced, input[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<REDUCTION>(&reduced, input[idx + BLOCK_SIZE]);
            }
        }

        *s_data_tid = reduced;
        __syncthreads();

        reduceSharedMemory<REDUCTION>(tid, s_data, tmp_outputs + blockIdx.x);
    }

    // Given the condition that one thread should reduce at least 2 elements, computes the number of blocks of
    // BLOCK_SIZE threads needed to compute the entire array. The block count is maxed out to 512 since the kernel
    // will loop until the end is reached.
    uint getBlocks(uint elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE * 2 - 1)) / (BLOCK_SIZE * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    // Launches the kernel, which outputs one reduced element per block.
    template<int REDUCTION, typename T>
    void launch(T* input, T* tmp_outputs, uint elements, uint blocks, cudaStream_t stream) {
        constexpr int bytes_sh = BLOCK_SIZE * sizeof(T);
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<REDUCTION, true><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, tmp_outputs, elements);
        } else {
            kernel<REDUCTION, false><<<blocks, BLOCK_SIZE, bytes_sh, stream>>>(input, tmp_outputs, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16);

    /*
     * Sum reduces an array with a given pitch to some partial sums (one per block).
     * REDUCTION    :   Type of reduction: NOA_REDUCTION_SUM, NOA_REDUCTION_MIN or NOA_REDUCTION_MAX.
     * TWO_BY_TWO   :   If true, the number of logical elements per row is assumed to be a multiple of 64.
     *                  This allows to check for out of bounds every two iteration, as opposed to once per iteration.
     * T            :   Data type. (u)int, float, double, cfloat_t or cdouble_t.
     * input        :   Input array to reduce. Should be at least `pitch * shape.y` elements.
     * pitch        :   Pitch of @a input, in elements.
     * outputs      :   Returned sum. One per block.
     * shape        :   Logical {fast, medium} shape of @a input. For a 3D array, shape.y should be y * z.
     */
    template<int REDUCTION, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* input, uint pitch, T* outputs, uint2_t shape) {
        uint tid = threadIdx.y * BLOCK_SIZE.x + threadIdx.x; // linear index within the block.
        T* s_data = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE.x * BLOCK_SIZE.y * sizeof(T) bytes.
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

    // Returns the number of necessary blocks to compute an array with that many rows.
    uint getBlocks(uint rows) {
        constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
        uint blocks = (rows + (WARPS - 1)) / WARPS;
        return Noa::Math::min(blocks, MAX_BLOCKS);
    }

    // Launches the kernel, which outputs one element per block.
    template<int REDUCTION, typename T>
    void launch(T* input, uint pitch, T* output_partial, uint2_t shape, uint blocks, cudaStream_t stream) {
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
    /*
     * For each batch (i.e. block), reduces a contiguous array to one final sum and mean.
     * This is optimized for small arrays (one block of max 256 threads per array).
     * Blocks are independent and can be seen as batches. Each batch reduces an array of @a elements elements.
     *
     * BLOCK_SIZE       :   Should be 32, 64, 128 or 256.
     * TWO_BY_TWO       :   If true, the number of elements is assumed to be a multiple of 2 * BLOCK_SIZE,
     *                      This allows to check for out of bounds every two iterations during the first
     *                      reduction, as opposed to once per iteration.
     * T                :   Data type. (u)int, float, double, cfloat_t or cdouble_t.
     * input            :   Input array to reduce.
     * elements         :   Number of elements to reduce in an input array.
     * output_sums      :   Returned sum. One per block. If nullptr, ignores it.
     * output_means     :   Returned mean. One per block. If nullptr, ignores it.
     * scale            :   Value used to compute the mean (sum / value). This is used when the input is the
     *                      intermediary sums of the kernels above. If output_means is nullptr, it is ignored.
     */
    template<int BLOCK_SIZE, bool TWO_BY_TWO, typename T, typename U>
    __global__ void kernel(T* inputs, uint elements, T* output_sums, T* output_means, U scale) {
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        T* s_data = Memory::Shared<T>::getBlockResource(); // BLOCK_SIZE * sizeof(T) bytes.
        T* s_data_tid = s_data + tid;

        // First, the block reduces the elements to BLOCK_SIZE elements.
        // Each threads sums 2 elements at a time until the end of the array is reached.
        // More blocks results in a larger grid, thus fewer elements per thread.
        T sum = 0;
        for (uint idx = tid; idx < elements; idx += BLOCK_SIZE * 2) {
            sum += inputs[idx];
            if constexpr (TWO_BY_TWO) {
                sum += inputs[idx + BLOCK_SIZE];
            } else {
                if (idx + BLOCK_SIZE < elements)
                    sum += inputs[idx + BLOCK_SIZE];
            }
        }
        *s_data_tid = sum;
        __syncthreads();

        // Once the initial sum is done, parallel reduce the shared array of BLOCK_SIZE elements to one single element.
        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128)
                *s_data_tid += s_data_tid[128];
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64)
                *s_data_tid += s_data_tid[64];
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
                    Details::warpSumReduce(s_data_tid);
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

    // One block works on one array and there's one array per block.
    // Given that one thread reduces at least 2 elements, how many threads should we assign to compute the entire array?
    // This is either 32, 64, 128 or 256.
    uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    // Launches the kernel, which outputs one sum and mean per batch. There's one block per batch.
    template<typename T, typename U>
    void launch(T* input, T* output_sums, T* output_means,
                size_t elements, U scale, uint batches,
                uint threads, cudaStream_t stream) {
        int bytes_sm = threads * sizeof(T);
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<256, true><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                              output_sums, output_means, scale);
                    break;
                case 128:
                    kernel<128, true><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                              output_sums, output_means, scale);
                    break;
                case 64:
                    kernel<64, true><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                             output_sums, output_means, scale);
                    break;
                case 32:
                    kernel<32, true><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                             output_sums, output_means, scale);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<256, false><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                               output_sums, output_means, scale);
                    break;
                case 128:
                    kernel<128, false><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                               output_sums, output_means, scale);
                    break;
                case 64:
                    kernel<64, false><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                              output_sums, output_means, scale);
                    break;
                case 32:
                    kernel<32, false><<<batches, threads, bytes_sm, stream>>>(input, elements,
                                                                              output_sums, output_means, scale);
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
    // This is very much similar to the kernel above but for min or max, i.e. it does not compute the mean.
    template<int REDUCTION, int BLOCK_SIZE, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* inputs, uint elements, T* outputs) {
        static_assert(REDUCTION == REDUCTION_MIN || REDUCTION == REDUCTION_MAX);
        static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 256);

        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        __shared__ T s_data[BLOCK_SIZE];
        T* s_data_tid = s_data + tid;

        T reduced = *inputs;
        for (uint idx = tid; idx < elements; idx += BLOCK_SIZE * 2) {
            inPlace<REDUCTION>(&reduced, inputs[idx]);
            if constexpr (TWO_BY_TWO) {
                inPlace<REDUCTION>(&reduced, inputs[idx + BLOCK_SIZE]);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<REDUCTION>(&reduced, inputs[idx + BLOCK_SIZE]);
            }
        }
        *s_data_tid = reduced;
        __syncthreads();

        if constexpr (BLOCK_SIZE >= 256) {
            if (tid < 128)
                inPlace<REDUCTION>(s_data_tid, s_data_tid[128]);
            __syncthreads();
        }
        if constexpr (BLOCK_SIZE >= 128) {
            if (tid < 64)
                inPlace<REDUCTION>(s_data_tid, s_data_tid[64]);
            __syncthreads();
        }

        if constexpr (BLOCK_SIZE >= 64) {
            if (tid < 32)
                warpReduce<REDUCTION>(s_data_tid);
        }

        if (tid == 0) {
            if constexpr (BLOCK_SIZE == 32) {
                for (int i = 1; i < BLOCK_SIZE; ++i)
                    inPlace<REDUCTION>(s_data, s_data[i]);
            }
            outputs[batch] = *s_data;
        }
    }

    uint getThreads(size_t elements) {
        uint threads = Noa::Math::nextPowerOf2((elements + 1) / 2);
        return Noa::Math::clamp(threads, 32U, 256U);
    }

    template<int REDUCTION, typename T>
    void launch(T* inputs, T* outputs, size_t elements, uint batches, uint threads, cudaStream_t stream) {
        bool two_by_two = !(elements % (threads * 2));
        if (two_by_two) {
            switch (threads) {
                case 256:
                    kernel<REDUCTION, 256, true><<<batches, 256, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 128:
                    kernel<REDUCTION, 128, true><<<batches, 128, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 64:
                    kernel<REDUCTION, 64, true><<<batches, 64, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 32:
                    kernel<REDUCTION, 32, true><<<batches, 32, 0, stream>>>(inputs, elements, outputs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        } else {
            switch (threads) {
                case 256:
                    kernel<REDUCTION, 256, false><<<batches, 256, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 128:
                    kernel<REDUCTION, 128, false><<<batches, 128, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 64:
                    kernel<REDUCTION, 64, false><<<batches, 64, 0, stream>>>(inputs, elements, outputs);
                    break;
                case 32:
                    kernel<REDUCTION, 32, false><<<batches, 32, 0, stream>>>(inputs, elements, outputs);
                    break;
                default:
                    NOA_THROW("DEV: block size should be 32, 64, 128 or 256, "
                              "got threads:{}, with elements:{}", threads, elements);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
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
    template void Details::minOrMax<Details::REDUCTION_MIN, T>(T*, T*, size_t, uint, Stream&);           \
    template void Details::minOrMax<Details::REDUCTION_MIN, T>(T*, size_t, T*, size3_t, uint, Stream&);  \
    template void Details::minOrMax<Details::REDUCTION_MAX, T>(T*, T*, size_t, uint, Stream&);           \
    template void Details::minOrMax<Details::REDUCTION_MAX, T>(T*, size_t, T*, size3_t, uint, Stream&)

    INSTANTIATE_MIN_OR_MAX(float);
    INSTANTIATE_MIN_OR_MAX(double);
    INSTANTIATE_MIN_OR_MAX(int);
    INSTANTIATE_MIN_OR_MAX(uint);
    INSTANTIATE_MIN_OR_MAX(char);
    INSTANTIATE_MIN_OR_MAX(unsigned char);
}
