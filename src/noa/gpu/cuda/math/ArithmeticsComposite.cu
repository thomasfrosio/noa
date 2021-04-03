#include "noa/gpu/cuda/math/ArithmeticsComposite.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

using namespace Noa;

static constexpr size_t max_threads_in_block = 256;
static constexpr size_t max_block_size = 32768;
static constexpr size_t warp_size = CUDA::Limits::warp_size;

// One block computes its elements and go to the corresponding elements in next grid, until the end, for each batch.
static NOA_HOST std::pair<size_t, size_t> getLaunchConfig(size_t elements) {
    size_t threads = max_threads_in_block;
    size_t total_blocks = Noa::Math::min((elements + threads - 1) / threads, max_block_size);
    return {total_blocks, threads};
}

// One block computes its row and go to the corresponding row in next grid, until the end, for each batch.
static NOA_HOST std::pair<size_t, size_t> getLaunchConfig(size3_t shape) {
    size_t threads = Noa::Math::min(max_threads_in_block, getNextMultipleOf(shape.x, warp_size)); // threads per row.
    size_t total_blocks = Noa::Math::min(Noa::getRows(shape), max_block_size);
    return {total_blocks, threads};
}

// KERNELS:
namespace Noa::CUDA::Math::Kernels {
    template<typename T>
    static __global__ void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs, uint elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            tmp_out[idx] = tmp_in[idx] * multipliers[idx] + addends[idx];
    }

    template<typename T>
    static __global__ void multiplyAddArray(T* inputs, uint pitch_inputs,
                                            T* multipliers, uint pitch_multipliers,
                                            T* addends, uint pitch_addends,
                                            T* outputs, uint pitch_outputs,
                                            uint elements_in_row, uint rows_per_batch) {
        T* tmp_in = inputs + blockIdx.y * pitch_inputs * rows_per_batch;
        T* tmp_out = outputs + blockIdx.y * pitch_outputs * rows_per_batch;
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x) {
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                tmp_out[row * pitch_outputs + idx] = tmp_in[row * pitch_inputs + idx] *
                                                     multipliers[row * pitch_multipliers + idx] +
                                                     addends[row * pitch_addends + idx];
        }
    }

    template<typename T>
    static __global__ void squaredDistanceFromValue(T* inputs, T* values, T* outputs, size_t elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        T value = values[blockIdx.y];
        T distance;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x) {
            distance = tmp_in[idx] - value;
            tmp_out[idx] = distance * distance;
        }
    }

    template<typename T>
    static __global__ void squaredDistanceFromValue(T* inputs, uint pitch_inputs, T* values,
                                                    T* outputs, uint pitch_outputs,
                                                    uint elements_in_row, uint rows_per_batch) {
        T* tmp_in = inputs + blockIdx.y * pitch_inputs * rows_per_batch;
        T* tmp_out = outputs + blockIdx.y * pitch_outputs * rows_per_batch;
        T value = values[blockIdx.y];
        T distance;
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x) {
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x) {
                distance = tmp_in[row * pitch_inputs + idx] - value;
                tmp_out[row * pitch_outputs + idx] = distance * distance;
            }
        }
    }

    template<typename T>
    static __global__ void squaredDistanceFromArray(T* inputs, T* array, T* outputs, size_t elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        T distance;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x) {
            distance = tmp_in[idx] - array[idx];
            tmp_out[idx] = distance * distance;
        }
    }

    template<typename T>
    static __global__ void squaredDistanceFromArray(T* inputs, uint pitch_inputs,
                                                    T* array, uint pitch_array,
                                                    T* outputs, uint pitch_outputs,
                                                    uint elements_in_row, uint rows_per_batch) {
        T* tmp_in = inputs + blockIdx.y * pitch_inputs * rows_per_batch;
        T* tmp_out = outputs + blockIdx.y * pitch_outputs * rows_per_batch;
        T distance;
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x) {
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x) {
                distance = tmp_in[row * pitch_inputs + idx] - array[row * pitch_array + idx];
                tmp_out[row * pitch_outputs + idx] = distance * distance;
            }
        }
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    /* ------------------------ */
    /* --- multiplyAddArray --- */
    /* ------------------------ */

    template<typename T>
    void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs,
                          size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::multiplyAddArray,
                        inputs, multipliers, addends, outputs, elements);
    }

    template<typename T>
    void multiplyAddArray(T* inputs, size_t pitch_inputs,
                          T* multipliers, size_t pitch_multipliers,
                          T* addends, size_t pitch_addends,
                          T* outputs, size_t pitch_outputs,
                          size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::multiplyAddArray,
                        inputs, pitch_inputs, multipliers, pitch_multipliers, addends, pitch_addends,
                        outputs, pitch_outputs, shape.x, getRows(shape));
    }

    /* ------------------------ */
    /* --- Squared distance --- */
    /* ------------------------ */

    template<typename T>
    void squaredDistanceFromValue(T* inputs, T* values, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::squaredDistanceFromValue,
                        inputs, values, outputs, elements);
    }

    template<typename T>
    void squaredDistanceFromValue(T* inputs, size_t pitch_inputs, T* values,
                                  T* outputs, size_t pitch_outputs,
                                  size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::squaredDistanceFromValue,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T>
    void squaredDistanceFromArray(T* inputs, T* array, T* outputs,
                                  size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::squaredDistanceFromArray,
                        inputs, array, outputs, elements);
    }

    template<typename T>
    void squaredDistanceFromArray(T* inputs, size_t pitch_inputs,
                                  T* array, size_t pitch_array,
                                  T* outputs, size_t pitch_outputs,
                                  size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::squaredDistanceFromArray,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }
}

namespace Noa::CUDA::Math {
    #define INSTANTIATE_COMPOSITES(T, U)                                                                        \
    template void multiplyAddArray<T>(T*, T*, T*, T*, size_t, uint, Stream&);                                   \
    template void multiplyAddArray<T>(T*, size_t, T*, size_t, T*, size_t, T*, size_t, size3_t, uint, Stream&);  \
    template void squaredDistanceFromValue<T>(T*, T*, T*, size_t, uint, Stream&);                               \
    template void squaredDistanceFromValue<T>(T*, size_t, T*, T*, size_t, size3_t, uint, Stream&);              \
    template void squaredDistanceFromArray<T>(T*, T*, T*, size_t, uint, Stream&);                               \
    template void squaredDistanceFromArray<T>(T*, size_t, T* array, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_COMPOSITES(float, float);
    INSTANTIATE_COMPOSITES(double, double);
    INSTANTIATE_COMPOSITES(int32_t, int32_t);
    INSTANTIATE_COMPOSITES(uint32_t, uint32_t);
    INSTANTIATE_COMPOSITES(cfloat_t, cfloat_t);
    INSTANTIATE_COMPOSITES(cdouble_t, cdouble_t);
}
