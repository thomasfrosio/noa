#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

// TODO This is an old piece of code. Update this to support more reductions, better launch configs
//      and vectorized loads/stores.

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
        constexpr uint THREADS_1 = 512U;
        constexpr uint THREADS_2 = 128U;

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS_1)
        void kernel1_(const T* __restrict__ input,
                      T* __restrict__ tmp_values, uint* __restrict__ tmp_indexes,
                      uint elements) {
            uint tid = threadIdx.x;
            __shared__ T s_values[THREADS_1];
            __shared__ uint s_indexes[THREADS_1];
            T* s_values_tid = s_values + tid;
            uint* s_indexes_tid = s_indexes + tid;

            T current_value = *input;
            uint current_idx = 0;
            uint increment = THREADS_1 * 2 * gridDim.x;
            for (uint idx = blockIdx.x * THREADS_1 * 2 + threadIdx.x; idx < elements; idx += increment) {
                inPlace_<FIND>(&current_value, &current_idx, input[idx], idx);
                if constexpr (TWO_BY_TWO) {
                    inPlace_<FIND>(&current_value, &current_idx, input[idx + THREADS_1], idx + THREADS_1);
                } else {
                    if (idx + THREADS_1 < elements)
                        inPlace_<FIND>(&current_value, &current_idx, input[idx + THREADS_1], idx + THREADS_1);
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
        // reduce at least 2 elements, computes the number of blocks of THREADS threads needed to compute the
        // entire array. The block count is maxed out to 512 since the kernel will loop until the end is reached.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 512U;
            uint blocks = math::divideUp(elements, THREADS_1 * 2);
            return noa::math::min(MAX_BLOCKS, blocks);
        }

        // Launches the kernel, which outputs one reduced element per block.
        // Should be at least 1025 elements, i.e. at least 2 blocks. Use details::Final otherwise.
        template<int FIND, typename T>
        void launch1_(const T* input, T* tmp_values, uint* tmp_indexes,
                      uint elements, uint blocks, cudaStream_t stream) {
            bool two_by_two = !(elements % (THREADS_1 * 2));
            if (two_by_two) {
                kernel1_<FIND, true><<<blocks, THREADS_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
            } else {
                kernel1_<FIND, false><<<blocks, THREADS_1, 0, stream>>>(input, tmp_values, tmp_indexes, elements);
            }
            NOA_THROW_IF(cudaGetLastError());
        }

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS_2)
        void kernel2_(T* __restrict__ tmp_values, uint* __restrict__ tmp_indexes, uint tmps,
                      size_t* __restrict__ output_indexes) {
            uint tid = threadIdx.x;
            uint batch = blockIdx.x;
            tmp_values += tmps * batch;
            tmp_indexes += tmps * batch;

            __shared__ T s_values[THREADS_2];
            __shared__ uint s_indexes[THREADS_2];
            T* s_values_tid = s_values + tid;
            uint* s_indexes_tid = s_indexes + tid;

            T current_value = *tmp_values;
            uint current_idx = *tmp_indexes;
            for (uint idx = tid; idx < tmps; idx += THREADS_2 * 2) {
                inPlaceNonOrdered_<FIND>(&current_value, &current_idx, tmp_values[idx], tmp_indexes[idx]);

                if constexpr (TWO_BY_TWO) {
                    inPlaceNonOrdered_<FIND>(&current_value, &current_idx,
                                             tmp_values[idx + THREADS_2], tmp_indexes[idx + THREADS_2]);
                } else {
                    if (idx + THREADS_2 < tmps)
                        inPlaceNonOrdered_<FIND>(&current_value, &current_idx,
                                                 tmp_values[idx + THREADS_2], tmp_indexes[idx + THREADS_2]);
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
                      size_t batches, cudaStream_t stream) {
            bool two_by_two = !(elements % (THREADS_2 * 2));
            if (two_by_two) {
                kernel2_<FIND, true><<<batches, THREADS_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                        elements, outputs_indexes);
            } else {
                kernel2_<FIND, false><<<batches, THREADS_2, 0, stream>>>(tmp_values, tmp_indexes,
                                                                         elements, outputs_indexes);
            }
            NOA_THROW_IF(cudaGetLastError());
        }
    }

    namespace one_step_ {
        constexpr uint THREADS = 128U;

        template<int FIND, bool TWO_BY_TWO, typename T>
        __global__ __launch_bounds__(THREADS)
        void kernel_(const T* __restrict__ inputs, uint elements, size_t* __restrict__ output_indexes) {
            uint tid = threadIdx.x;
            uint batch = blockIdx.x;
            inputs += elements * batch;

            __shared__ T s_values[THREADS];
            __shared__ uint s_indexes[THREADS];
            T* s_values_tid = s_values + tid;
            uint* s_indexes_tid = s_indexes + tid;

            T current_value = *inputs;
            uint current_idx = 0;
            for (uint idx = tid; idx < elements; idx += THREADS * 2) {
                inPlace_<FIND>(&current_value, &current_idx, inputs[idx], idx);

                if constexpr (TWO_BY_TWO) {
                    inPlace_<FIND>(&current_value, &current_idx, inputs[idx + THREADS], idx + THREADS);
                } else {
                    if (idx + THREADS < elements)
                        inPlace_<FIND>(&current_value, &current_idx, inputs[idx + THREADS], idx + THREADS);
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
        void launch_(const T* inputs, size_t* outputs, size_t elements, size_t batches, cudaStream_t stream) {
            bool two_by_two = !(elements % (THREADS * 2));
            if (two_by_two) {
                kernel_<FIND, true><<<batches, THREADS, 0, stream>>>(inputs, elements, outputs);
            } else {
                kernel_<FIND, false><<<batches, THREADS, 0, stream>>>(inputs, elements, outputs);
            }
            NOA_THROW_IF(cudaGetLastError());
        }
    }
}

namespace noa::cuda::math::details {
    template<int SEARCH_FOR, typename T>
    void find(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream) {
        if (elements <= 4096 || batches > 8) {
            if (elements)
                one_step_::launch_<SEARCH_FOR>(inputs, output_indexes, elements, batches, stream.id());
            Stream::synchronize(stream);

        } else {
            size_t blocks = two_steps_::getBlocks_(elements);
            size_t total_blocks = blocks * batches;
            memory::PtrDevice<T> d_tmp_values(total_blocks);
            memory::PtrDevice<uint> d_tmp_indexes(total_blocks);
            for (size_t batch = 0; batch < batches; ++batch) {
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
    template void find<details::FIRST_MIN, T>(const T*, size_t*, size_t, size_t, Stream&);  \
    template void find<details::FIRST_MAX, T>(const T*, size_t*, size_t, size_t, Stream&);  \
    template void find<details::LAST_MIN, T>(const T*, size_t*, size_t, size_t, Stream&);   \
    template void find<details::LAST_MAX, T>(const T*, size_t*, size_t, size_t, Stream&)

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
