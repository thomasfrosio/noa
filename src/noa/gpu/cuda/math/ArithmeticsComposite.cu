#include "noa/gpu/cuda/math/ArithmeticsComposite.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace {
    using namespace Noa;

    namespace Contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        // One block computes its elements and go to the corresponding elements
        // in next grid, until the end, for each batch.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 32768;
            return Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
        }

        template<typename T>
        __global__ void multiplyAddArray_(T* inputs, T* multipliers, T* addends, T* outputs, uint elements) {
            T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = tmp_in[idx] * multipliers[idx] + addends[idx];
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(T* inputs, T* values, T* outputs, size_t elements) {
            T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            T value = values[blockIdx.y];
            T distance;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                distance = tmp_in[idx] - value;
                tmp_out[idx] = distance * distance;
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromArray_(T* inputs, T* array, T* outputs, size_t elements) {
            T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            T distance;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                distance = tmp_in[idx] - array[idx];
                tmp_out[idx] = distance * distance;
            }
        }
    }

    namespace Padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<typename T>
        __global__ void multiplyAddArray_(T* inputs, uint pitch_inputs,
                                          T* multipliers, uint pitch_multipliers,
                                          T* addends, uint pitch_addends,
                                          T* outputs, uint pitch_outputs,
                                          uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    outputs[row * pitch_outputs + idx] = inputs[row * pitch_inputs + idx] *
                                                         multipliers[row * pitch_multipliers + idx] +
                                                         addends[row * pitch_addends + idx];
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(T* inputs, uint pitch_inputs, T* values,
                                                  T* outputs, uint pitch_outputs,
                                                  uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            T value = values[blockIdx.y];
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T distance = inputs[row * pitch_inputs + idx] - value;
                    outputs[row * pitch_outputs + idx] = distance * distance;
                }
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromArray_(T* inputs, uint pitch_inputs,
                                                  T* array, uint pitch_array,
                                                  T* outputs, uint pitch_outputs,
                                                  uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T distance = inputs[row * pitch_inputs + idx] - array[row * pitch_array + idx];
                    outputs[row * pitch_outputs + idx] = distance * distance;
                }
            }
        }
    }
}

namespace Noa::CUDA::Math {
    /* ------------------------ */
    /* --- multiplyAddArray --- */
    /* ------------------------ */

    template<typename T>
    void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs,
                          size_t elements, uint batches, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::multiplyAddArray_,
                        inputs, multipliers, addends, outputs, elements);
    }

    template<typename T>
    void multiplyAddArray(T* inputs, size_t pitch_inputs,
                          T* multipliers, size_t pitch_multipliers,
                          T* addends, size_t pitch_addends,
                          T* outputs, size_t pitch_outputs,
                          size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::multiplyAddArray_,
                        inputs, pitch_inputs, multipliers, pitch_multipliers, addends, pitch_addends,
                        outputs, pitch_outputs, shape_2d);
    }

    /* ------------------------ */
    /* --- Squared distance --- */
    /* ------------------------ */

    template<typename T>
    void squaredDistanceFromValue(T* inputs, T* values, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::squaredDistanceFromValue_,
                        inputs, values, outputs, elements);
    }

    template<typename T>
    void squaredDistanceFromValue(T* inputs, size_t pitch_inputs, T* values,
                                  T* outputs, size_t pitch_outputs,
                                  size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::squaredDistanceFromValue_,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape_2d);
    }

    template<typename T>
    void squaredDistanceFromArray(T* inputs, T* array, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::squaredDistanceFromArray_,
                        inputs, array, outputs, elements);
    }

    template<typename T>
    void squaredDistanceFromArray(T* inputs, size_t pitch_inputs,
                                  T* array, size_t pitch_array,
                                  T* outputs, size_t pitch_outputs,
                                  size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::squaredDistanceFromArray_,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape_2d);
    }

    #define INSTANTIATE_COMPOSITES(T, U)                                                                        \
    template void multiplyAddArray<T>(T*, T*, T*, T*, size_t, uint, Stream&);                                   \
    template void multiplyAddArray<T>(T*, size_t, T*, size_t, T*, size_t, T*, size_t, size3_t, uint, Stream&);  \
    template void squaredDistanceFromValue<T>(T*, T*, T*, size_t, uint, Stream&);                               \
    template void squaredDistanceFromValue<T>(T*, size_t, T*, T*, size_t, size3_t, uint, Stream&);              \
    template void squaredDistanceFromArray<T>(T*, T*, T*, size_t, uint, Stream&);                               \
    template void squaredDistanceFromArray<T>(T*, size_t, T* array, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_COMPOSITES(float, float);
    INSTANTIATE_COMPOSITES(double, double);
    INSTANTIATE_COMPOSITES(int, int);
    INSTANTIATE_COMPOSITES(uint, uint);
}
