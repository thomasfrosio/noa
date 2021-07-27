#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Indexes.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

namespace {
    using namespace noa;

    // This assumes that current_index is smaller than candidate_index.
    // This is true in the first reduction from global memory.
    template<int FIND, typename T>
    __forceinline__ __device__ void inPlace_(T* current_value, uint* current_index,
                                             T candidate_value, uint candidate_index) {
        if constexpr (FIND == cuda::math::details::FIRST_MIN) {
            if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::FIRST_MAX) {
            if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::LAST_MIN) {
            if (candidate_value <= *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::LAST_MAX) {
            if (*current_value <= candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    // This takes into account that that current_index is not necessarily smaller than candidate_index.
    // This is what happen when the shared memory is being reduced.
    template<int FIND, typename T>
    inline __device__ void inPlaceNonOrdered_(T* current_value, uint* current_index,
                                              T candidate_value, uint candidate_index) {
        if constexpr (FIND == cuda::math::details::FIRST_MIN) {
            if (candidate_value == *current_value) {
                if (candidate_index < *current_index)
                    *current_index = candidate_index;
            } else if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::FIRST_MAX) {
            if (candidate_value == *current_value) {
                if (candidate_index < *current_index)
                    *current_index = candidate_index;
            } else if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::LAST_MIN) {
            if (candidate_value == *current_value) {
                if (*current_index < candidate_index)
                    *current_index = candidate_index;
            } else if (candidate_value < *current_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        } else if constexpr (FIND == cuda::math::details::LAST_MAX) {
            if (candidate_value == *current_value) {
                if (*current_index < candidate_index)
                    *current_index = candidate_index;
            } else if (*current_value < candidate_value) {
                *current_value = candidate_value;
                *current_index = candidate_index;
            }
        }
    }

    namespace two_steps_ {
        constexpr uint BLOCK_SIZE_1 = 512U;
        constexpr uint BLOCK_SIZE_2 = 128U;

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ void kernel1_(const T* input, T* tmp_values, uint* tmp_indexes, uint elements) {
            uint tid = threadIdx.x;
            __shared__ T s_values[BLOCK_SIZE_1];
            __shared__ uint s_indexes[BLOCK_SIZE_1];
            T* s_values_tid = s_values + tid;
            uint* s_indexes_tid = s_indexes + tid;

            T current_value = *input;
            uint current_idx = 0;
            uint increment = BLOCK_SIZE_1 * 2 * gridDim.x;
            for (uint idx = blockIdx.x * BLOCK_SIZE_1 * 2 + threadIdx.x; idx < elements; idx += increment) {
                inPlace_<FIND>(&current_value, &current_idx, input[idx], idx);
                if constexpr (TWO_BY_TWO) {
                    inPlace_<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
                } else {
                    if (idx + BLOCK_SIZE_1 < elements)
                        inPlace_<FIND>(&current_value, &current_idx, input[idx + BLOCK_SIZE_1], idx + BLOCK_SIZE_1);
                }
            }
            *s_values_tid = current_value;
            *s_indexes_tid = current_idx;
            __syncthreads();

            if (tid < 256)
                inPlaceNonOrdered_<FIND>(s_values_tid, s_indexes_tid, s_values_tid[256], s_indexes_tid[256]);
            __syncthreads();
            if (tid < 128)
                inPlaceNonOrdered_<FIND>(s_values_tid, s_indexes_tid, s_values_tid[128], s_indexes_tid[128]);
            __syncthreads();
            if (tid < 64)
                inPlaceNonOrdered_<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
            __syncthreads();

            // Reduces the last 2 warps to one element.
            if (tid == 0) {
                for (int idx = 0; idx < 64; ++idx)
                    inPlaceNonOrdered_<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
                tmp_values[blockIdx.x] = *s_values;
                tmp_indexes[blockIdx.x] = *s_indexes;
            }
        }

        // Given an array with at least 1025 elements (see launch()) and given the condition that one thread should
        // reduce at least 2 elements, computes the number of blocks of BLOCK_SIZE threads needed to compute the
        // entire array. The block count is maxed out to 512 since the kernel will loop until the end is reached.
        uint getBlocks_(size_t elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = (elements + (BLOCK_SIZE_1 * 2 - 1)) / (BLOCK_SIZE_1 * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        // Launches the kernel, which outputs one reduced element per block.
        // Should be at least 1025 elements, i.e. at least 2 blocks. Use details::Final otherwise.
        template<int FIND, typename T>
        void launch1_(const T* input, T* tmp_values, uint* tmp_indexes,
                      uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE_1 * 2));
            if (two_by_two) {
                kernel1_<FIND, true><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
            } else {
                kernel1_<FIND, false><<<blocks, BLOCK_SIZE_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ void kernel2_(T* tmp_values, uint* tmp_indexes, uint tmps, size_t* output_indexes) {
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
                inPlaceNonOrdered_<FIND>(&current_value, &current_idx, tmp_values[idx], tmp_indexes[idx]);

                if constexpr (TWO_BY_TWO) {
                    inPlaceNonOrdered_<FIND>(&current_value, &current_idx,
                                             tmp_values[idx + BLOCK_SIZE_2], tmp_indexes[idx + BLOCK_SIZE_2]);
                } else {
                    if (idx + BLOCK_SIZE_2 < tmps)
                        inPlaceNonOrdered_<FIND>(&current_value, &current_idx,
                                                 tmp_values[idx + BLOCK_SIZE_2], tmp_indexes[idx + BLOCK_SIZE_2]);
                }
            }
            *s_values_tid = current_value;
            *s_indexes_tid = current_idx;
            __syncthreads();

            if (tid < 64)
                inPlaceNonOrdered_<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
            __syncthreads();

            if (tid == 0) {
                for (int idx = 0; idx < 64; ++idx)
                    inPlaceNonOrdered_<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
                output_indexes[batch] = static_cast<size_t>(*s_indexes);
            }
        }

        template<int FIND, typename T>
        void launch2_(T* tmp_values, uint* tmp_indexes, size_t elements, size_t* outputs_indexes,
                      uint batches, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE_2 * 2));
            if (two_by_two) {
                kernel2_<FIND, true><<<batches, BLOCK_SIZE_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                           elements, outputs_indexes);
            } else {
                kernel2_<FIND, false><<<batches, BLOCK_SIZE_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                            elements, outputs_indexes);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }

    namespace one_step_ {
        constexpr uint BLOCK_SIZE = 128U;

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ void kernel_(const T* inputs, uint elements, size_t* output_indexes) {
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
                inPlace_<FIND>(&current_value, &current_idx, inputs[idx], idx);

                if constexpr (TWO_BY_TWO) {
                    inPlace_<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
                } else {
                    if (idx + BLOCK_SIZE < elements)
                        inPlace_<FIND>(&current_value, &current_idx, inputs[idx + BLOCK_SIZE], idx + BLOCK_SIZE);
                }
            }
            *s_values_tid = current_value;
            *s_indexes_tid = current_idx;
            __syncthreads();

            if (tid < 64)
                inPlaceNonOrdered_<FIND>(s_values_tid, s_indexes_tid, s_values_tid[64], s_indexes_tid[64]);
            __syncthreads();

            if (tid == 0) {
                for (int idx = 0; idx < 64; ++idx)
                    inPlaceNonOrdered_<FIND>(s_values, s_indexes, s_values_tid[idx], s_indexes_tid[idx]);
                output_indexes[batch] = static_cast<size_t>(*s_indexes);
            }
        }

        template<int FIND, typename T>
        void launch_(const T* inputs, size_t* outputs, size_t elements, uint batches, cudaStream_t stream) {
            bool two_by_two = !(elements % (BLOCK_SIZE * 2));
            if (two_by_two) {
                kernel_<FIND, true><<<batches, BLOCK_SIZE, 0, stream>>>(inputs, elements, outputs);
            } else {
                kernel_<FIND, false><<<batches, BLOCK_SIZE, 0, stream>>>(inputs, elements, outputs);
            }
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }
}

namespace noa::cuda::math::details {
    template<int SEARCH_FOR, typename T>
    void find(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        if (elements <= 4096 || batches > 8) {
            if (elements)
                one_step_::launch_<SEARCH_FOR>(inputs, output_indexes, elements, batches, stream.id());
            Stream::synchronize(stream);

        } else {
            uint blocks = two_steps_::getBlocks_(elements);
            uint total_blocks = blocks * batches;
            memory::PtrDevice<T> d_tmp_values(total_blocks);
            memory::PtrDevice<uint> d_tmp_indexes(total_blocks);
            for (uint batch = 0; batch < batches; ++batch) {
                const T* input = inputs + batch * elements;
                T* intermediary_values = d_tmp_values.get() + batch * blocks;
                uint* intermediary_indexes = d_tmp_indexes.get() + batch * blocks;
                two_steps_::launch1_<SEARCH_FOR>(input, intermediary_values, intermediary_indexes,
                                                elements, blocks, stream.id());
            }
            two_steps_::launch2_<SEARCH_FOR>(d_tmp_values.get(), d_tmp_indexes.get(), blocks, output_indexes,
                                            batches, stream.id());
            Stream::synchronize(stream);
        }
    }

    #define NOA_INSTANTIATE_FIND_(T)                                                        \
    template void find<details::FIRST_MIN, T>(const T*, size_t*, size_t, uint, Stream&);    \
    template void find<details::FIRST_MAX, T>(const T*, size_t*, size_t, uint, Stream&);    \
    template void find<details::LAST_MIN, T>(const T*, size_t*, size_t, uint, Stream&);     \
    template void find<details::LAST_MAX, T>(const T*, size_t*, size_t, uint, Stream&)

    NOA_INSTANTIATE_FIND_(char);
    NOA_INSTANTIATE_FIND_(short);
    NOA_INSTANTIATE_FIND_(int);
    NOA_INSTANTIATE_FIND_(long);
    NOA_INSTANTIATE_FIND_(long long);
    NOA_INSTANTIATE_FIND_(unsigned char);
    NOA_INSTANTIATE_FIND_(unsigned short);
    NOA_INSTANTIATE_FIND_(unsigned int);
    NOA_INSTANTIATE_FIND_(unsigned long);
    NOA_INSTANTIATE_FIND_(unsigned long long);
}
