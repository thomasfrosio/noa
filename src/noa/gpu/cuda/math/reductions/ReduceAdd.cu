// Implementation of math::reduceAdd(), math::reduceMean() and math::reduceMeanWeighted() for contiguous and padded layouts.

#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Reductions.h"

namespace {
    using namespace noa;

    namespace contiguous_ {
        // Get blocks and threads, given that each thread should compute at least 2 elements.
        void getLaunchConfig_(uint elements, uint* output_blocks, uint* output_threads) {
            constexpr uint MAX_THREADS = 256U, MAX_BLOCKS = 512U;
            *output_threads = noa::math::nextMultipleOf((elements + 1) / 2, 32U);
            *output_threads = noa::math::clamp(*output_threads, 32U, MAX_THREADS);
            *output_blocks = (elements + (*output_threads * 2 - 1)) / (*output_threads * 2);
            *output_blocks = noa::math::min(*output_blocks, MAX_BLOCKS);
        }

        template<typename T>
        __global__ void reduceAdd_(const T* inputs, T* outputs, uint elements, int vectors) {
            inputs += blockIdx.y * vectors * elements;
            outputs += blockIdx.y * elements;

            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
                T sum = 0;
                for (int vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vector + idx];
                outputs[idx] = sum;
            }
        }

        template<typename T>
        __global__ void reduceMean_(const T* inputs, T* outputs, uint elements, int vectors) {
            inputs += blockIdx.y * vectors * elements;
            outputs += blockIdx.y * elements;
            auto scale = static_cast<noa::traits::value_type_t<T>>(vectors);
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
                T sum = 0;
                for (int vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vector + idx];
                outputs[idx] = sum / scale;
            }
        }

        template<typename T, typename U>
        __global__ void reduceMeanWeighted_(const T* inputs, const U* weights, T* outputs, uint elements, int vectors) {
            inputs += blockIdx.y * vectors * elements;
            outputs += blockIdx.y * elements;

            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
                T sum = 0;
                U sum_weight = 0;
                for (int vector = 0; vector < vectors; ++vector) {
                    uint offset = elements * vector + idx;
                    U weight = weights[offset];
                    sum_weight += weight;
                    sum += inputs[offset] * weight;
                }
                if (sum_weight != 0)
                    outputs[idx] = sum / sum_weight;
                else
                    outputs[idx] = 0;
            }
        }
    }

    namespace padded_ {
        constexpr uint2_t BLOCK_SIZE(32, 16);

        void getLaunchConfig_(uint2_t shape_2d, uint batches, dim3* output_blocks, dim3* output_threads) {
            constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.

            output_blocks->x = noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
            output_blocks->y = batches; // requires batches < 65535
            output_blocks->z = 1;

            output_threads->x = BLOCK_SIZE.x;
            output_threads->y = BLOCK_SIZE.y;
            output_threads->z = 1;
        }

        template<typename T>
        __global__ void reduceAdd_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                   int vectors, uint2_t shape) {
            uint size_input = inputs_pitch * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * outputs_pitch * shape.y;

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset = row * inputs_pitch; // offset to starting element for that warp.
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[vector * size_input + offset + idx];
                    outputs[outputs_pitch * row + idx] = sum;
                }
            }
        }

        template<typename T>
        __global__ void reduceMean_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                                    int vectors, uint2_t shape) {
            uint size_input = inputs_pitch * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * outputs_pitch * shape.y;
            auto scale = static_cast<noa::traits::value_type_t<T>>(vectors);

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset = row * inputs_pitch;
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[vector * size_input + offset + idx];
                    outputs[outputs_pitch * row + idx] = sum / scale;
                }
            }
        }

        template<typename T, typename U>
        __global__ void reduceMeanWeighted_(const T* inputs, uint inputs_pitch,
                                            const U* weights, uint weights_pitch,
                                            T* outputs, uint outputs_pitch,
                                            int vectors, uint2_t shape) {
            uint size_input = inputs_pitch * shape.y;
            uint size_weight = weights_pitch * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * outputs_pitch * shape.y;

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset_inputs = row * inputs_pitch;
                uint offset_weights = row * weights_pitch;
                uint offset_outputs = row * outputs_pitch;

                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T sum = 0;
                    U sum_weight = 0;
                    for (int vector = 0; vector < vectors; ++vector) {
                        U weight = weights[size_weight * vector + offset_weights + idx];
                        sum_weight += weight;
                        sum += inputs[size_input * vector + offset_inputs + idx] * weight;
                    }
                    if (sum_weight != 0)
                        outputs[offset_outputs + idx] = sum / sum_weight;
                    else
                        outputs[offset_outputs + idx] = 0;
                }
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void reduceAdd(const T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream) {
        uint blocks, threads;
        contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        contiguous_::reduceAdd_<<<dim3(blocks, batches), threads, 0, stream.id()>>>(
                inputs, outputs, elements, vectors);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void reduceAdd(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                   size3_t shape, uint nb_to_reduce, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        padded_::reduceAdd_<<<blocks, threads, 0, stream.id()>>>(
                inputs, inputs_pitch, outputs, outputs_pitch, nb_to_reduce, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void reduceMean(const T* inputs, T* outputs, size_t elements, uint nb_to_reduce, uint batches, Stream& stream) {
        uint blocks, threads;
        contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        contiguous_::reduceMean_<<<dim3(blocks, batches), threads, 0, stream.id()>>>(
                inputs, outputs, elements, nb_to_reduce);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void reduceMean(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                    size3_t shape, uint nb_to_reduce, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        padded_::reduceMean_<<<blocks, threads, 0, stream.id()>>>(
                inputs, inputs_pitch, outputs, outputs_pitch, nb_to_reduce, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void reduceMeanWeighted(const T* inputs, const U* weights, T* outputs,
                            size_t elements, uint nb_to_reduce, uint batches, Stream& stream) {
        uint blocks, threads;
        contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        contiguous_::reduceMeanWeighted_<<<dim3(blocks, batches), threads, 0, stream.id()>>>(
                inputs, weights, outputs, elements, nb_to_reduce);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void reduceMeanWeighted(const T* inputs, size_t inputs_pitch,
                            const U* weights, size_t weights_pitch,
                            T* outputs, size_t outputs_pitch,
                            size3_t shape, uint nb_to_reduce, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        padded_::reduceMeanWeighted_<<<blocks, threads, 0, stream.id()>>>(
                inputs, inputs_pitch, weights, weights_pitch, outputs, outputs_pitch, nb_to_reduce, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_ADD_MEAN_(T)                                                    \
    template void reduceAdd<T>(const T*, T*, size_t, uint, uint, Stream&);                  \
    template void reduceAdd<T>(const T*, size_t, T*, size_t, size3_t, uint, uint, Stream&); \
    template void reduceMean<T>(const T*, T*, size_t, uint, uint, Stream&);                 \
    template void reduceMean<T>(const T*, size_t, T*, size_t, size3_t, uint, uint, Stream&)

    NOA_INSTANTIATE_ADD_MEAN_(int);
    NOA_INSTANTIATE_ADD_MEAN_(long);
    NOA_INSTANTIATE_ADD_MEAN_(long long);
    NOA_INSTANTIATE_ADD_MEAN_(unsigned int);
    NOA_INSTANTIATE_ADD_MEAN_(unsigned long);
    NOA_INSTANTIATE_ADD_MEAN_(unsigned long long);
    NOA_INSTANTIATE_ADD_MEAN_(float);
    NOA_INSTANTIATE_ADD_MEAN_(double);
    NOA_INSTANTIATE_ADD_MEAN_(cfloat_t);
    NOA_INSTANTIATE_ADD_MEAN_(cdouble_t);

    #define NOA_INSTANTIATE_WEIGHTED_(T, U)                                                         \
    template void reduceMeanWeighted<T, U>(const T*, const U*, T*, size_t, uint, uint, Stream&);    \
    template void reduceMeanWeighted<T, U>(const T*, size_t, const U*, size_t, T*, size_t, size3_t, uint, uint, Stream&)

    NOA_INSTANTIATE_WEIGHTED_(int, int);
    NOA_INSTANTIATE_WEIGHTED_(long, long);
    NOA_INSTANTIATE_WEIGHTED_(long long, long long);
    NOA_INSTANTIATE_WEIGHTED_(unsigned int, unsigned int);
    NOA_INSTANTIATE_WEIGHTED_(unsigned long, unsigned long);
    NOA_INSTANTIATE_WEIGHTED_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_WEIGHTED_(float, float);
    NOA_INSTANTIATE_WEIGHTED_(double, double);
    NOA_INSTANTIATE_WEIGHTED_(cfloat_t, float);
    NOA_INSTANTIATE_WEIGHTED_(cdouble_t, double);
}
