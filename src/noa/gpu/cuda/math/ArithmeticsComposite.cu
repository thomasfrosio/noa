#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/ArithmeticsComposite.h"

namespace {
    using namespace noa;

    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        // One block computes its elements and go to the corresponding elements
        // in next grid, until the end, for each batch.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 32768;
            return noa::math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
        }

        template<typename T>
        __global__ void multiplyAddArray_(const T* inputs, const T* multipliers, const T* addends,
                                          T* outputs, uint elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = tmp_in[idx] * multipliers[idx] + addends[idx];
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(const T* inputs, const T* values, T* outputs, size_t elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            T value = values[blockIdx.y];
            T distance;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                distance = tmp_in[idx] - value;
                tmp_out[idx] = distance * distance;
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(const T* inputs, T value, T* outputs, size_t elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            T distance;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                distance = tmp_in[idx] - value;
                tmp_out[idx] = distance * distance;
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromArray_(const T* inputs, const T* array, T* outputs, size_t elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            T distance;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                distance = tmp_in[idx] - array[idx];
                tmp_out[idx] = distance * distance;
            }
        }
    }

    namespace padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            return noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<typename T>
        __global__ void multiplyAddArray_(const T* inputs, uint inputs_pitch,
                                          const T* multipliers, uint multipliers_pitch,
                                          const T* addends, uint addends_pitch,
                                          T* outputs, uint outputs_pitch,
                                          uint2_t shape) {
            inputs += blockIdx.y * inputs_pitch * shape.y;
            outputs += blockIdx.y * outputs_pitch * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    outputs[row * outputs_pitch + idx] = inputs[row * inputs_pitch + idx] *
                                                         multipliers[row * multipliers_pitch + idx] +
                                                         addends[row * addends_pitch + idx];
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(const T* inputs, uint inputs_pitch, const T* values,
                                                  T* outputs, uint outputs_pitch,
                                                  uint2_t shape) {
            inputs += blockIdx.y * inputs_pitch * shape.y;
            outputs += blockIdx.y * outputs_pitch * shape.y;
            T value = values[blockIdx.y];
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T distance = inputs[row * inputs_pitch + idx] - value;
                    outputs[row * outputs_pitch + idx] = distance * distance;
                }
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromValue_(const T* inputs, uint inputs_pitch, T value,
                                                  T* outputs, uint outputs_pitch,
                                                  uint2_t shape) {
            inputs += blockIdx.y * inputs_pitch * shape.y;
            outputs += blockIdx.y * outputs_pitch * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T distance = inputs[row * inputs_pitch + idx] - value;
                    outputs[row * outputs_pitch + idx] = distance * distance;
                }
            }
        }

        template<typename T>
        __global__ void squaredDistanceFromArray_(const T* inputs, uint inputs_pitch,
                                                  const T* array, uint array_pitch,
                                                  T* outputs, uint outputs_pitch,
                                                  uint2_t shape) {
            inputs += blockIdx.y * inputs_pitch * shape.y;
            outputs += blockIdx.y * outputs_pitch * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T distance = inputs[row * inputs_pitch + idx] - array[row * array_pitch + idx];
                    outputs[row * outputs_pitch + idx] = distance * distance;
                }
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                          size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::multiplyAddArray_<<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, multipliers, addends, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void multiplyAddArray(const T* inputs, size_t inputs_pitch,
                          const T* multipliers, size_t multipliers_pitch,
                          const T* addends, size_t addends_pitch,
                          T* outputs, size_t outputs_pitch,
                          size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        dim3 blocks(padded_::getBlocks_(shape_2d), batches);
        padded_::multiplyAddArray_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, multipliers, multipliers_pitch, addends,
                        addends_pitch, outputs, outputs_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::squaredDistanceFromValue_<<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, values, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromValue(const T* inputs, size_t inputs_pitch, const T* values,
                                  T* outputs, size_t outputs_pitch,
                                  size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::squaredDistanceFromValue_<<<dim3(blocks, batches), padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, values, outputs, outputs_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromValue(const T* inputs, T value, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::squaredDistanceFromValue_<<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, value, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromValue(const T* inputs, size_t inputs_pitch, T value,
                                  T* outputs, size_t outputs_pitch,
                                  size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::squaredDistanceFromValue_<<<dim3(blocks, batches), padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, value, outputs, outputs_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::squaredDistanceFromArray_<<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, array, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void squaredDistanceFromArray(const T* inputs, size_t inputs_pitch,
                                  const T* array, size_t array_pitch,
                                  T* outputs, size_t outputs_pitch,
                                  size3_t shape, uint batches, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::squaredDistanceFromArray_<<<dim3(blocks, batches), padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(T)                                                                               \
    template void multiplyAddArray<T>(const T*, const T*, const T*, T*, size_t, uint, Stream&);                                     \
    template void multiplyAddArray<T>(const T*, size_t, const T*, size_t, const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void squaredDistanceFromValue<T>(const T*, const T*, T*, size_t, uint, Stream&);                                       \
    template void squaredDistanceFromValue<T>(const T*, size_t, const T*, T*, size_t, size3_t, uint, Stream&);                      \
    template void squaredDistanceFromValue<T>(const T*, T, T*, size_t, uint, Stream&);                                              \
    template void squaredDistanceFromValue<T>(const T*, size_t, T, T*, size_t, size3_t, uint, Stream&);                             \
    template void squaredDistanceFromArray<T>(const T*, const T*, T*, size_t, uint, Stream&);                                       \
    template void squaredDistanceFromArray<T>(const T*, size_t, const T*, size_t, T*, size_t, size3_t, uint, Stream&)

    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(int);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(long long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned int);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(unsigned long long);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(float);
    NOA_INSTANTIATE_ARITHMETICS_COMPOSITE_(double);
}
