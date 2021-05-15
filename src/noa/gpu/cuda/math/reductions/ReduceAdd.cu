// Implementation of Math::reduceAdd(), Math::reduceMean() and Math::reduceMeanWeighted() for contiguous and padded layouts.

#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace {
    using namespace Noa;

    namespace Contiguous_ {
        // Get blocks and threads, given that each thread should compute at least 2 elements.
        void getLaunchConfig_(uint elements, uint* output_blocks, uint* output_threads) {
            constexpr uint MAX_THREADS = 256U, MAX_BLOCKS = 512U;
            *output_threads = Noa::Math::nextMultipleOf((elements + 1) / 2, 32U);
            *output_threads = Noa::Math::clamp(*output_threads, 32U, MAX_THREADS);
            *output_blocks = (elements + (*output_threads * 2 - 1)) / (*output_threads * 2);
            *output_blocks = Noa::Math::min(*output_blocks, MAX_BLOCKS);
        }

        template<typename T>
        __global__ void reduceAdd_(T* inputs, T* outputs, uint elements, int vectors) {
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
        __global__ void reduceMean_(T* inputs, T* outputs, uint elements, int vectors) {
            inputs += blockIdx.y * vectors * elements;
            outputs += blockIdx.y * elements;
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(vectors);
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
                T sum = 0;
                for (int vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vector + idx];
                outputs[idx] = sum / scale;
            }
        }

        template<typename T, typename U>
        __global__ void reduceMeanWeighted_(T* inputs, U* weights, T* outputs, uint elements, int vectors) {
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

    namespace Padded_ {
        constexpr uint2_t BLOCK_SIZE(32, 16);

        void getLaunchConfig_(uint2_t shape_2d, uint batches, dim3* output_blocks, dim3* output_threads) {
            constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.

            output_blocks->x = Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
            output_blocks->y = batches; // requires batches < 65535
            output_blocks->z = 1;

            output_threads->x = BLOCK_SIZE.x;
            output_threads->y = BLOCK_SIZE.y;
            output_threads->z = 1;
        }

        template<typename T>
        __global__ void reduceAdd_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                   int vectors, uint2_t shape) {
            uint size_input = pitch_inputs * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * pitch_outputs * shape.y;

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset = row * pitch_inputs; // offset to starting element for that warp.
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[vector * size_input + offset + idx];
                    outputs[pitch_outputs * row + idx] = sum;
                }
            }
        }

        template<typename T>
        __global__ void reduceMean_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                    int vectors, uint2_t shape) {
            uint size_input = pitch_inputs * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            auto scale = static_cast<Noa::Traits::value_type_t<T>>(vectors);

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset = row * pitch_inputs;
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[vector * size_input + offset + idx];
                    outputs[pitch_outputs * row + idx] = sum / scale;
                }
            }
        }

        template<typename T, typename U>
        __global__ void reduceMeanWeighted_(T* inputs, uint pitch_inputs,
                                            U* weights, uint pitch_weights,
                                            T* outputs, uint pitch_outputs,
                                            int vectors, uint2_t shape) {
            uint size_input = pitch_inputs * shape.y;
            uint size_weight = pitch_weights * shape.y;
            inputs += blockIdx.y * vectors * size_input;
            outputs += blockIdx.y * pitch_outputs * shape.y;

            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                uint offset_inputs = row * pitch_inputs;
                uint offset_weights = row * pitch_weights;
                uint offset_outputs = row * pitch_outputs;

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

namespace Noa::CUDA::Math {
    template<typename T>
    void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream) {
        uint blocks, threads;
        Contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Contiguous_::reduceAdd_,
                        inputs, outputs, elements, vectors);
    }

    template<typename T>
    void reduceAdd(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                   size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        Padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        Padded_::reduceAdd_,
                        inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d);
    }

    template<typename T>
    void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream) {
        uint blocks, threads;
        Contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Contiguous_::reduceMean_,
                        inputs, outputs, elements, vectors);
    }

    template<typename T>
    void reduceMean(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                    size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        Padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        Padded_::reduceMean_,
                        inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d);
    }

    template<typename T, typename U>
    void reduceMeanWeighted(T* inputs, U* weights, T* outputs,
                            size_t elements, uint vectors, uint batches, Stream& stream) {
        uint blocks, threads;
        Contiguous_::getLaunchConfig_(elements, &blocks, &threads);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Contiguous_::reduceMeanWeighted_,
                        inputs, weights, outputs, elements, vectors);
    }

    template<typename T, typename U>
    void reduceMeanWeighted(T* inputs, size_t pitch_inputs,
                            U* weights, size_t pitch_weights,
                            T* outputs, size_t pitch_outputs,
                            size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks, threads;
        Padded_::getLaunchConfig_(shape_2d, batches, &blocks, &threads);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        Padded_::reduceMeanWeighted_,
                        inputs, pitch_inputs, weights, pitch_weights, outputs, pitch_outputs, vectors, shape_2d);
    }

    #define INSTANTIATE_ADD_MEAN(T)                                                     \
    template void reduceAdd<T>(T*, T*, size_t, uint, uint, Stream&);                    \
    template void reduceAdd<T>(T*, size_t, T*, size_t, size3_t, uint, uint, Stream&);   \
    template void reduceMean<T>(T*, T*, size_t, uint, uint, Stream&);                   \
    template void reduceMean<T>(T*, size_t, T*, size_t, size3_t, uint, uint, Stream&)

    INSTANTIATE_ADD_MEAN(int);
    INSTANTIATE_ADD_MEAN(uint);
    INSTANTIATE_ADD_MEAN(float);
    INSTANTIATE_ADD_MEAN(double);
    INSTANTIATE_ADD_MEAN(cfloat_t);
    INSTANTIATE_ADD_MEAN(cdouble_t);

    #define INSTANTIATE_WEIGHTED(T, U)                                                                      \
    template void reduceMeanWeighted<T, U>(T*, U*, T*, size_t, uint, uint, Stream&);                        \
    template void reduceMeanWeighted<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, uint, Stream&)

    INSTANTIATE_WEIGHTED(int, int);
    INSTANTIATE_WEIGHTED(uint, uint);
    INSTANTIATE_WEIGHTED(float, float);
    INSTANTIATE_WEIGHTED(double, double);
    INSTANTIATE_WEIGHTED(cfloat_t, float);
    INSTANTIATE_WEIGHTED(cdouble_t, double);
}
