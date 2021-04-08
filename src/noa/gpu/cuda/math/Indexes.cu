#include "noa/gpu/cuda/math/Indexes.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/PtrDevice.h"

namespace Noa::CUDA::Math::Details {
    template<int FIND, typename T>
    NOA_FD void inPlace(T* current_value, uint* current_index, T candidate_value, uint canditate_index) {
        if constexpr (FIND == FIRST_MIN) {
            if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = canditate_index;
            }
        } else if constexpr (FIND == FIRST_MAX) {
            if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = canditate_index;
            }
        } else if constexpr (FIND == LAST_MIN) {
            if (*current_value <= candidate_value) {
                *current_value = candidate_value;
                *current_index = canditate_index;
            }
        } else if constexpr (FIND == LAST_MAX) {
            if (*current_value <= candidate_value) {
                *current_value = candidate_value;
                *current_index = canditate_index;
            }
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }

    template<int FIND, typename T>
    NOA_DEVICE void warpReduce(volatile T* s_values_tid, volatile uint* s_indexes_tid) {
        for (int idx = 32; idx >= 1; idx /= 2) {
            if constexpr (FIND == FIRST_MIN) {
                if (s_values_tid[idx] < *s_values_tid) {
                    *s_values_tid = s_values_tid[idx];
                    *s_indexes_tid = s_indexes_tid[idx];
                }
            } else if constexpr (FIND == FIRST_MAX) {
                if (*s_values_tid < s_values_tid[idx]) {
                    *s_values_tid = s_values_tid[idx];
                    *s_indexes_tid = s_indexes_tid[idx];
                }
            } else if constexpr (FIND == LAST_MIN) {
                if (s_values_tid[idx] <= *s_values_tid) {
                    *s_values_tid = s_values_tid[idx];
                    *s_indexes_tid = s_indexes_tid[idx];
                }
            } else if constexpr (FIND == LAST_MAX) {
                if (*s_values_tid <= s_values_tid[idx]) {
                    *s_values_tid = s_values_tid[idx];
                    *s_indexes_tid = s_indexes_tid[idx];
                }
            } else {
                static_assert(Noa::Traits::always_false_v<T>);
            }
        }
    }
}

namespace Noa::CUDA::Math::Details::TwoSteps {
    static constexpr uint BLOCK_SIZE_1 = 512U;
    static constexpr uint BLOCK_SIZE_2 = 128U;

    template<int FIND, bool TWO_BY_TWO, typename T>
    __global__ void kernel1(T* input, T* output_values, uint* output_indexes, uint elements) {
        uint tid = threadIdx.x;
        __shared__ T s_values[BLOCK_SIZE_1];
        __shared__ uint s_indexes[BLOCK_SIZE_1];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *input;
        uint current_idx = 0;
        uint increment = BLOCK_SIZE_1 * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE_1 * 2 + threadIdx.x;
        while (idx < elements) {
            inPlace<FIND>(&current_value, &current_idx, input[idx], idx);

            if constexpr (TWO_BY_TWO) {
                inPlace<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
            } else {
                if (idx + BLOCK_SIZE_1 < elements)
                    inPlace<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
            }
            idx += increment;
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 256)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[256], tid + 256);
        __syncthreads();
        if (tid < 128)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[128], tid + 128);
        __syncthreads();
        if (tid < 64)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], tid + 64);
        __syncthreads();

        // Reduces the last 2 warps to one element.
        if (tid < 32)
            warpReduce<FIND>(s_values_tid, s_indexes_tid);
        if (tid == 0) {
            output_values[blockIdx.x] = *s_values;
            output_indexes[blockIdx.x] = *s_indexes;
        }
    }

    // Given an array with at least 1025 elements (see launch()) and given the condition that one thread should
    // reduce at least 2 elements, computes the number of blocks of BLOCK_SIZE threads needed to compute the
    // entire array. The block count is maxed out to 512 since the kernel will loop until the end is reached.
    NOA_HOST uint getBlocks(size_t elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE_1 * 2 - 1)) / (BLOCK_SIZE_1 * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    // Launches the kernel, which outputs one reduced element per block.
    // Should be at least 1025 elements, i.e. at least 2 blocks. Use Details::Final otherwise.
    template<int FIND, typename T>
    NOA_IH void launch1(T* input, T* output_values, uint* output_indexes,
                        uint elements, uint blocks, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE_1 * 2));
        if (two_by_two) {
            kernel1<FIND, true><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, output_values, output_indexes, elements);
        } else {
            kernel1<FIND, false><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, output_values, output_indexes, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int FIND, bool TWO_BY_TWO, typename T>
    __global__ void kernel2(T* input_values, uint* input_indexes, uint elements, size_t* output_indexes) {
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        input_values += elements * batch;
        input_indexes += elements * batch;

        __shared__ T s_values[BLOCK_SIZE_2];
        __shared__ uint s_indexes[BLOCK_SIZE_2];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *input_values;
        uint current_idx = *input_indexes;
        uint increment = BLOCK_SIZE_2 * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE_2 * 2 + threadIdx.x;
        while (idx < elements) {
            inPlace<FIND>(&current_value, &current_idx, input_values[idx], input_indexes[idx]);

            if constexpr (TWO_BY_TWO) {
                inPlace<FIND>(&current_value, &current_idx,
                              input_values[idx + BLOCK_SIZE_2], input_indexes[idx + BLOCK_SIZE_2]);
            } else {
                if (idx + BLOCK_SIZE_2 < elements)
                    inPlace<FIND>(&current_value, &current_idx,
                                  input_values[idx + BLOCK_SIZE_2], input_indexes[idx + BLOCK_SIZE_2]);
            }
            idx += increment;
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 64)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], tid + 64);
        __syncthreads();

        if (tid < 32)
            warpReduce<FIND>(s_values_tid, s_indexes_tid);
        if (tid == 0)
            output_indexes[batch] = static_cast<size_t>(*s_indexes);
    }

    template<int FIND, typename T>
    NOA_HOST void launch2(T* input_values, uint* input_indexes, size_t elements, size_t* outputs_indexes,
                          uint batches, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE_2 * 2));
        if (two_by_two) {
            kernel2<FIND, true><<<batches, BLOCK_SIZE_2, 0, stream>>>(input_values, input_indexes,
                                                                      elements, outputs_indexes);
        } else {
            kernel2<FIND, false><<<batches, BLOCK_SIZE_2, 0, stream>>>(input_values, input_indexes,
                                                                       elements, outputs_indexes);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace Noa::CUDA::Math::Details::OneStep {
    static constexpr uint BLOCK_SIZE = 128U;

    template<int FIND, bool TWO_BY_TWO, typename T>
    __global__ void kernel(T* inputs, uint elements, size_t* output_indexes) {
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        __shared__ T s_values[BLOCK_SIZE];
        __shared__ uint s_indexes[BLOCK_SIZE];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *inputs;
        uint current_idx = 0;
        uint increment = BLOCK_SIZE * 2 * gridDim.x;
        uint idx = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
        while (idx < elements) {
            inPlace<FIND>(&current_value, &current_idx, inputs[idx], idx);

            if constexpr (TWO_BY_TWO) {
                inPlace<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
            }
            idx += increment;
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 128)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[128], tid + 128);
        __syncthreads();
        if (tid < 64)
            inPlace<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], tid + 64);
        __syncthreads();

        if (tid < 32)
            warpReduce<FIND>(s_values_tid, s_indexes_tid);
        if (tid == 0)
            output_indexes[batch] = static_cast<size_t>(*s_indexes);
    }

    template<int FIND, typename T>
    NOA_HOST void launch(T* inputs, size_t* outputs, size_t elements, uint batches, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE * 2));
        if (two_by_two) {
            kernel<FIND, true><<<batches, BLOCK_SIZE, 0, stream>>>(inputs, elements, outputs);
        } else {
            kernel<FIND, false><<<batches, BLOCK_SIZE, 0, stream>>>(inputs, elements, outputs);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace Noa::CUDA::Math::Details {
    template<int SEARCH_FOR, typename T>
    void find(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        if (elements <= 4096 || batches > 8) {
            if (!elements)
                return;
            Details::OneStep::launch<SEARCH_FOR>(inputs, output_indexes, elements, batches, stream.id());

        } else {
            uint blocks = Details::TwoSteps::getBlocks(elements);
            uint total_blocks = blocks * batches;
            PtrDevice<T> d_tmp_values(total_blocks);
            PtrDevice<uint> d_tmp_indexes(total_blocks);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* intermediary_values = d_tmp_values.get() + batch * blocks;
                uint* intermediary_indexes = d_tmp_indexes.get() + batch * blocks;
                Details::TwoSteps::launch1<SEARCH_FOR>(input, intermediary_values, intermediary_indexes,
                                                       elements, blocks, stream.get());
            }
            Details::TwoSteps::launch2<SEARCH_FOR>(d_tmp_values.get(), d_tmp_indexes.get(), blocks, output_indexes,
                                                   batches, stream.id());
        }
        CUDA::Stream::synchronize(stream);
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math::Details {
    #define INSTANTIATE_FIND(T)                                                     \
    template void find<Details::FIRST_MIN, T>(T*, size_t*, size_t, uint, Stream&);  \
    template void find<Details::FIRST_MAX, T>(T*, size_t*, size_t, uint, Stream&);  \
    template void find<Details::LAST_MIN, T>(T*, size_t*, size_t, uint, Stream&);   \
    template void find<Details::LAST_MAX, T>(T*, size_t*, size_t, uint, Stream&)

    INSTANTIATE_FIND(float);
    INSTANTIATE_FIND(double);
    INSTANTIATE_FIND(int32_t);
    INSTANTIATE_FIND(uint32_t);
    INSTANTIATE_FIND(char);
    INSTANTIATE_FIND(unsigned char);
    INSTANTIATE_FIND(int16_t);
    INSTANTIATE_FIND(uint16_t);
}
