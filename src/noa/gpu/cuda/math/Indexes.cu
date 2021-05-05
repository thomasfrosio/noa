// Implementation for Math::firstMin() and Math::firstMax(), Math::lastMin(), Math::lastMin().

#include "noa/gpu/cuda/math/Indexes.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace Noa::CUDA::Math::Details {
    // This assumes that current_index is smaller than candidate_index.
    // This is true in the first reduction from global memory.
    template<int FIND, typename T>
    static NOA_FD void inPlace(T* current_value, uint* current_index, T candidate_value, uint candidate_index) {
        if constexpr (FIND == FIRST_MIN) {
            if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == FIRST_MAX) {
            if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == LAST_MIN) {
            if (candidate_value <= *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == LAST_MAX) {
            if (*current_value <= candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }

    // This takes into account that that current_index is not necessarily smaller than candidate_index.
    // This is what happen when the shared memory is being reduced.
    template<int FIND, typename T>
    static NOA_ID void inPlaceNonOrdered(T* current_value, uint* current_index,
                                         T candidate_value, uint candidate_index) {
        if constexpr (FIND == FIRST_MIN) {
            if (candidate_value == *current_value) {
                if (candidate_index < *current_index)
                    *current_index = candidate_index;
            } else if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == FIRST_MAX) {
            if (candidate_value == *current_value) {
                if (candidate_index < *current_index)
                    *current_index = candidate_index;
            } else if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == LAST_MIN) {
            if (candidate_value == *current_value) {
                if (*current_index < candidate_index)
                    *current_index = candidate_index;
            } else if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == LAST_MAX) {
            if (candidate_value == *current_value) {
                if (*current_index < candidate_index)
                    *current_index = candidate_index;
            } else if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        }
    }
}

namespace Noa::CUDA::Math::Details::TwoSteps {
    static constexpr uint BLOCK_SIZE_1 = 512U;
    static constexpr uint BLOCK_SIZE_2 = 128U;

    template<int FIND, bool TWO_BY_TWO, typename T>
    static __global__ void kernel1(T* input, T* tmp_values, uint* tmp_indexes, uint elements) {
        uint tid = threadIdx.x;
        __shared__ T s_values[BLOCK_SIZE_1];
        __shared__ uint s_indexes[BLOCK_SIZE_1];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *input;
        uint current_idx = 0;
        uint increment = BLOCK_SIZE_1 * 2 * gridDim.x;
        for (uint idx = blockIdx.x * BLOCK_SIZE_1 * 2 + threadIdx.x; idx < elements; idx += increment) {
            inPlace<FIND>(&current_value, &current_idx, input[idx], idx);
            if constexpr (TWO_BY_TWO) {
                inPlace<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
            } else {
                if (idx + BLOCK_SIZE_1 < elements)
                    inPlace<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
            }
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 256)
            inPlaceNonOrdered<FIND>(s_values_tid, s_indexes_tid, s_values_tid[256], s_indexes_tid[256]);
        __syncthreads();
        if (tid < 128)
            inPlaceNonOrdered<FIND>(s_values_tid, s_indexes_tid, s_values_tid[128], s_indexes_tid[128]);
        __syncthreads();
        if (tid < 64)
            inPlaceNonOrdered<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
        __syncthreads();

        // Reduces the last 2 warps to one element.
        if (tid == 0) {
            for (int idx = 0; idx < 64; ++idx)
                inPlaceNonOrdered<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
            tmp_values[blockIdx.x] = *s_values;
            tmp_indexes[blockIdx.x] = *s_indexes;
        }
    }

    // Given an array with at least 1025 elements (see launch()) and given the condition that one thread should
    // reduce at least 2 elements, computes the number of blocks of BLOCK_SIZE threads needed to compute the
    // entire array. The block count is maxed out to 512 since the kernel will loop until the end is reached.
    static uint getBlocks(size_t elements) {
        constexpr uint MAX_BLOCKS = 512U;
        uint blocks = (elements + (BLOCK_SIZE_1 * 2 - 1)) / (BLOCK_SIZE_1 * 2);
        return Noa::Math::min(MAX_BLOCKS, blocks);
    }

    // Launches the kernel, which outputs one reduced element per block.
    // Should be at least 1025 elements, i.e. at least 2 blocks. Use Details::Final otherwise.
    template<int FIND, typename T>
    static void launch1(T* input, T* tmp_values, uint* tmp_indexes,
                        uint elements, uint blocks, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE_1 * 2));
        if (two_by_two) {
            kernel1<FIND, true><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
        } else {
            kernel1<FIND, false><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int FIND, bool TWO_BY_TWO, typename T>
    static __global__ void kernel2(T* tmp_values, uint* tmp_indexes, uint tmps, size_t* output_indexes) {
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        tmp_values += tmps * batch;
        tmp_indexes += tmps * batch;

        __shared__ T s_values[BLOCK_SIZE_2];
        __shared__ uint s_indexes[BLOCK_SIZE_2];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *tmp_values;
        uint current_idx = *tmp_indexes;
        for (uint idx = tid; idx < tmps; idx += BLOCK_SIZE_2 * 2) {
            inPlaceNonOrdered<FIND>(&current_value, &current_idx, tmp_values[idx], tmp_indexes[idx]);

            if constexpr (TWO_BY_TWO) {
                inPlaceNonOrdered<FIND>(&current_value, &current_idx,
                                        tmp_values[idx + BLOCK_SIZE_2], tmp_indexes[idx + BLOCK_SIZE_2]);
            } else {
                if (idx + BLOCK_SIZE_2 < tmps)
                    inPlaceNonOrdered<FIND>(&current_value, &current_idx,
                                            tmp_values[idx + BLOCK_SIZE_2], tmp_indexes[idx + BLOCK_SIZE_2]);
            }
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 64)
            inPlaceNonOrdered<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
        __syncthreads();

        if (tid == 0) {
            for (int idx = 0; idx < 64; ++idx)
                inPlaceNonOrdered<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
            output_indexes[batch] = static_cast<size_t>(*s_indexes);
        }
    }

    template<int FIND, typename T>
    static void launch2(T* tmp_values, uint* tmp_indexes, size_t elements, size_t* outputs_indexes,
                        uint batches, cudaStream_t stream) {
        bool two_by_two = !(elements % (BLOCK_SIZE_2 * 2));
        if (two_by_two) {
            kernel2<FIND, true><<<batches, BLOCK_SIZE_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                      elements, outputs_indexes);
        } else {
            kernel2<FIND, false><<<batches, BLOCK_SIZE_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                       elements, outputs_indexes);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace Noa::CUDA::Math::Details::OneStep {
    static constexpr uint BLOCK_SIZE = 128U;

    template<int FIND, bool TWO_BY_TWO, typename T>
    static __global__ void kernel(T* inputs, uint elements, size_t* output_indexes) {
        uint tid = threadIdx.x;
        uint batch = blockIdx.x;
        inputs += elements * batch;

        __shared__ T s_values[BLOCK_SIZE];
        __shared__ uint s_indexes[BLOCK_SIZE];
        T* s_values_tid = s_values + tid;
        uint* s_indexes_tid = s_indexes + tid;

        T current_value = *inputs;
        uint current_idx = 0;
        for (uint idx = tid; idx < elements; idx += BLOCK_SIZE * 2) {
            inPlace<FIND>(&current_value, &current_idx, inputs[idx], idx);

            if constexpr (TWO_BY_TWO) {
                inPlace<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
            } else {
                if (idx + BLOCK_SIZE < elements)
                    inPlace<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
            }
        }
        *s_values_tid = current_value;
        *s_indexes_tid = current_idx;
        __syncthreads();

        if (tid < 64)
            inPlaceNonOrdered<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
        __syncthreads();

        if (tid == 0) {
            for (int idx = 0; idx < 64; ++idx)
                inPlaceNonOrdered<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
            output_indexes[batch] = static_cast<size_t>(*s_indexes);
        }
    }

    template<int FIND, typename T>
    static void launch(T* inputs, size_t* outputs, size_t elements, uint batches, cudaStream_t stream) {
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
            if (elements)
                Details::OneStep::launch<SEARCH_FOR>(inputs, output_indexes, elements, batches, stream.id());
            Stream::synchronize(stream);

        } else {
            uint blocks = Details::TwoSteps::getBlocks(elements);
            uint total_blocks = blocks * batches;
            Memory::PtrDevice<T> d_tmp_values(total_blocks);
            Memory::PtrDevice<uint> d_tmp_indexes(total_blocks);
            for (uint batch = 0; batch < batches; ++batch) {
                T* input = inputs + batch * elements;
                T* intermediary_values = d_tmp_values.get() + batch * blocks;
                uint* intermediary_indexes = d_tmp_indexes.get() + batch * blocks;
                Details::TwoSteps::launch1<SEARCH_FOR>(input, intermediary_values, intermediary_indexes,
                                                       elements, blocks, stream.id());
            }
            Details::TwoSteps::launch2<SEARCH_FOR>(d_tmp_values.get(), d_tmp_indexes.get(), blocks, output_indexes,
                                                   batches, stream.id());
            Stream::synchronize(stream);
        }
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math::Details {
    #define INSTANTIATE_FIND(T)                                                     \
    template void find<Details::FIRST_MIN, T>(T*, size_t*, size_t, uint, Stream&);  \
    template void find<Details::FIRST_MAX, T>(T*, size_t*, size_t, uint, Stream&);  \
    template void find<Details::LAST_MIN, T>(T*, size_t*, size_t, uint, Stream&);   \
    template void find<Details::LAST_MAX, T>(T*, size_t*, size_t, uint, Stream&)

    INSTANTIATE_FIND(int32_t);
    INSTANTIATE_FIND(uint32_t);
    INSTANTIATE_FIND(char);
    INSTANTIATE_FIND(unsigned char);
    INSTANTIATE_FIND(int16_t);
    INSTANTIATE_FIND(uint16_t);
}
