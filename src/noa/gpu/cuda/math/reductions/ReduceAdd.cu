#include "noa/gpu/cuda/math/Reductions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

// Implementation of Math::reduceAdd(), Math::reduceMean() and Math::reduceMeanWeighted() for contiguous and padded layouts.
// These kernels follow the same logic as Noa::CUDA::Math::sum(). See implementation in Min_Max_SumMean.cu for more details.

using namespace Noa;

// CONTIGUOUS LAYOUT:
namespace Noa::CUDA::Math::Details::Contiguous {
    template<typename T>
    __global__ void reduceAdd(T* inputs, T* outputs, uint elements, int vectors) {
        inputs += blockIdx.y * vectors * elements;
        outputs += blockIdx.y * elements;

        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
            T sum = 0;
            for (int vector = 0; vector < vectors; ++vectors)
                sum += inputs[elements * vector + idx];
            outputs[idx] = sum;
        }
    }

    template<typename T, typename U>
    __global__ void reduceMean(T* inputs, T* outputs, uint elements, int vectors, U scale) {
        inputs += blockIdx.y * vectors * elements;
        outputs += blockIdx.y * elements;

        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
            T sum = 0;
            for (int vector = 0; vector < vectors; ++vectors)
                sum += inputs[elements * vector + idx];
            outputs[idx] = sum / scale;
        }
    }

    template<typename T, typename U>
    __global__ void reduceMeanWeighted(T* inputs, U* weights, T* outputs, uint elements, int vectors) {
        inputs += blockIdx.y * vectors * elements;
        outputs += blockIdx.y * elements;

        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += gridDim.x * blockDim.x) {
            T sum = 0;
            U sum_weight = 0;
            for (int vector = 0; vector < vectors; ++vectors) {
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

    // Get blocks and threads, given that each thread should compute at least 2 elements.
    NOA_HOST std::tuple<uint, uint> getLaunchConfig(uint elements) {
        constexpr uint MAX_THREADS = 256U, MAX_BLOCKS = 512U;
        uint threads = Noa::Math::nextMultipleOf((elements + 1) / 2, 32U);
        threads = Noa::Math::clamp(threads, 32U, MAX_THREADS);
        uint blocks = (elements + (threads * 2 - 1)) / (threads * 2);
        blocks = Noa::Math::min(blocks, MAX_BLOCKS);
        return {blocks, threads};
    }
}

// PADDED LAYOUT:
namespace Noa::CUDA::Math::Details::Padded {
    static constexpr uint2_t BLOCK_SIZE(32, 16);

    template<int TWO_BY_TWO, typename T>
    __global__ void reduceAdd(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                              int vectors, uint2_t shape) {
        inputs += blockIdx.y * vectors * pitch_inputs * shape.y;
        outputs += blockIdx.y * pitch_outputs * shape.y;

        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            uint offset = row * pitch_inputs; // offset to starting element for that warp.
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) { // jump 2 warps at a time
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[offset + idx] + inputs[offset + idx + BLOCK_SIZE.x];
                    outputs[pitch_outputs * row + idx] = sum;
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) { // jump 1 warp at a time
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[offset + idx];
                    outputs[pitch_outputs * row + idx] = sum;
                }
            }
        }
    }

    template<int TWO_BY_TWO, typename T, typename U>
    __global__ void reduceMean(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                               int vectors, uint2_t shape, U scale) {
        inputs += blockIdx.y * vectors * pitch_inputs * shape.y;
        outputs += blockIdx.y * pitch_outputs * shape.y;

        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            uint offset = row * pitch_inputs; // offset to starting element for that warp.
            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) { // jump 2 warps at a time
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[offset + idx] + inputs[offset + idx + BLOCK_SIZE.x];
                    outputs[pitch_outputs * row + idx] = sum / scale;
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) { // jump 1 warp at a time
                    T sum = 0;
                    for (int vector = 0; vector < vectors; ++vector)
                        sum += inputs[offset + idx];
                    outputs[pitch_outputs * row + idx] = sum / scale;
                }
            }
        }
    }

    template<int TWO_BY_TWO, typename T, typename U>
    __global__ void reduceMeanWeighted(T* inputs, uint pitch_inputs,
                                       U* weights, uint pitch_weights,
                                       T* outputs, uint pitch_outputs,
                                       int vectors, uint2_t shape) {
        inputs += blockIdx.y * vectors * pitch_inputs * shape.y;
        weights += blockIdx.y * vectors * pitch_weights * shape.y;
        outputs += blockIdx.y * pitch_outputs * shape.y;

        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            uint offset_inputs = row * pitch_inputs;
            uint offset_weights = row * pitch_weights;
            uint offset_outputs = row * pitch_outputs;

            if constexpr (TWO_BY_TWO) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x * 2) { // jump 2 warps at a time
                    T sum = 0;
                    U sum_weight = 0;
                    for (int vector = 0; vector < vectors; ++vector) {
                        uint tmp = offset_weights * vector + idx;
                        U weight = weights[tmp] + weights[tmp + BLOCK_SIZE.x];
                        sum_weight += weight;
                        tmp = offset_inputs * vector + idx;
                        sum += inputs[tmp] + inputs[tmp + BLOCK_SIZE.x];
                    }
                    if (sum_weight != 0)
                        outputs[offset_outputs + idx] = sum / sum_weight;
                    else
                        outputs[offset_outputs + idx] = 0;
                }
            } else {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) { // jump 1 warp at a time
                    T sum = 0;
                    U sum_weight = 0;
                    for (int vector = 0; vector < vectors; ++vector) {
                        U weight = weights[offset_weights * vector + idx];
                        sum_weight += weight;
                        sum += inputs[offset_inputs * vector + idx];
                    }
                    if (sum_weight != 0)
                        outputs[offset_outputs + idx] = sum / sum_weight;
                    else
                        outputs[offset_outputs + idx] = 0;
                }
            }
        }
    }

    std::tuple<dim3, dim3, bool> getLaunchConfig(uint2_t shape_2d, uint batches) {
        constexpr uint MAX_BLOCKS = 512; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.

        dim3 blocks(Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS),
                    batches); // requires batches < 65535
        dim3 threads(BLOCK_SIZE.x, BLOCK_SIZE.y);
        bool two_by_two = !(shape_2d.x % (BLOCK_SIZE.x * 2));
        return {blocks, threads, two_by_two};
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T>
    void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream) {
        auto[blocks, threads] = Details::Contiguous::getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Details::Contiguous::reduceAdd,
                        inputs, outputs, elements, vectors);
    }

    template<typename T>
    void reduceAdd(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                   size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        auto[blocks, threads, two_by_two] = Details::Padded::getLaunchConfig(shape_2d, batches);
        if (two_by_two) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceAdd<true>,
                            inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceAdd<false>,
                            inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d);
        }
    }

    template<typename T>
    void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream) {
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(elements);
        auto[blocks, threads] = Details::Contiguous::getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Details::Contiguous::reduceMean,
                        inputs, outputs, elements, vectors, scale);
    }

    template<typename T>
    void reduceMean(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                    size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        auto[blocks, threads, two_by_two] = Details::Padded::getLaunchConfig(shape_2d, batches);
        auto scale = static_cast<Noa::Traits::value_type_t<T>>(getElements(shape_2d));

        if (two_by_two) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceMean<true>,
                            inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d, scale);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceMean<false>,
                            inputs, pitch_inputs, outputs, pitch_outputs, vectors, shape_2d, scale);
        }
    }

    template<typename T, typename U>
    void reduceMeanWeighted(T* inputs, U* weights, T* outputs,
                            size_t elements, uint vectors, uint batches, Stream& stream) {
        auto[blocks, threads] = Details::Contiguous::getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), threads, 0, stream.id(),
                        Details::Contiguous::reduceMeanWeighted,
                        inputs, weights, outputs, elements, vectors);
    }

    template<typename T, typename U>
    void reduceMeanWeighted(T* inputs, size_t pitch_inputs,
                            U* weights, size_t pitch_weights,
                            T* outputs, size_t pitch_outputs,
                            size3_t shape, uint vectors, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        auto[blocks, threads, two_by_two] = Details::Padded::getLaunchConfig(shape_2d, batches);

        if (two_by_two) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceMeanWeighted<true>,
                            inputs, pitch_inputs, weights, pitch_weights, outputs, pitch_outputs, vectors, shape_2d);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            Details::Padded::reduceMeanWeighted<false>,
                            inputs, pitch_inputs, weights, pitch_weights, outputs, pitch_outputs, vectors, shape_2d);
        }
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
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
