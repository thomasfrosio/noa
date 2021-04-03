#include "noa/gpu/cuda/math/Arithmetics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

#define NOA_OP_ADD 1
#define NOA_OP_SUBTRACT 2
#define NOA_OP_MULTIPLY 3
#define NOA_OP_DIVIDE 4
#define NOA_OP_DIVIDE_SAFE 5

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

template<int OPERATION, typename T, typename U>
static NOA_FD T computeArith(T lhs, U rhs) {
    T out;
    if constexpr (OPERATION == NOA_OP_ADD) {
        out = lhs + rhs;
    } else if constexpr (OPERATION == NOA_OP_SUBTRACT) {
        out = lhs - rhs;
    } else if constexpr (OPERATION == NOA_OP_MULTIPLY) {
        out = lhs * rhs;
    } else if constexpr (OPERATION == NOA_OP_DIVIDE) {
        out = lhs / rhs;
    } else if constexpr (OPERATION == NOA_OP_DIVIDE_SAFE) {
        out = Math::abs(rhs) < 1e-15 ? T(0) : lhs / rhs;
    } else {
        static_assert(Noa::Traits::always_false_v<T>);
    }
    return out; // https://stackoverflow.com/questions/64523302
}

// KERNELS:
namespace Noa::CUDA::Math::Kernels {
    template<int OPERATOR, typename T, typename U>
    static __global__ void computeSingleValue(T* input, U value, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = computeArith<OPERATOR>(input[idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeSingleValue(T* inputs, U* values, T* outputs, uint elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        U value = values[blockIdx.y];
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            tmp_out[idx] = computeArith<OPERATOR>(tmp_in[idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeElementWise(T* inputs, U* array, T* outputs, uint elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            tmp_out[idx] = computeArith<OPERATOR>(tmp_in[idx], array[idx]);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeSingleValue(T* input, uint pitch_input, U value,
                                              T* output, uint pitch_output,
                                              uint elements_in_row, uint rows_per_batch) {
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = computeArith<OPERATOR>(input[row * pitch_input + idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeSingleValue(T* inputs, uint pitch_inputs, U* values,
                                              T* outputs, uint pitch_outputs,
                                              uint elements_in_row, uint rows_per_batch) {
        T* tmp_in = inputs + blockIdx.y * pitch_inputs * rows_per_batch;
        T* tmp_out = outputs + blockIdx.y * pitch_outputs * rows_per_batch;
        U value = values[blockIdx.y];
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                tmp_out[row * pitch_outputs + idx] = computeArith<OPERATOR>(tmp_in[row * pitch_inputs + idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeElementWise(T* inputs, uint pitch_inputs,
                                              U* array, uint pitch_array,
                                              T* outputs, uint pitch_outputs,
                                              uint elements_in_row, uint rows_per_batch) {
        T* tmp_in = inputs + blockIdx.y * pitch_inputs * rows_per_batch;
        T* tmp_out = outputs + blockIdx.y * pitch_outputs * rows_per_batch;
        for (uint row = blockIdx.x; row < rows_per_batch; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                tmp_out[row * pitch_outputs + idx] = computeArith<OPERATOR>(tmp_in[row * pitch_inputs + idx],
                                                                            array[row * pitch_array + idx]);
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    /* ---------------- */
    /* --- Multiply --- */
    /* ---------------- */

    template<typename T, typename U>
    void multiplyByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_MULTIPLY>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void multiplyByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_MULTIPLY>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void multiplyByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_MULTIPLY>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void multiplyByValue(T* input, size_t pitch_input, U value,
                         T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_MULTIPLY>,
                        input, pitch_input, value, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void multiplyByValue(T* inputs, size_t pitch_inputs, U* values,
                         T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_MULTIPLY>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void multiplyByArray(T* inputs, size_t pitch_inputs,
                         U* array, size_t pitch_array,
                         T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_MULTIPLY>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    /* -------------- */
    /* --- Divide --- */
    /* -------------- */

    template<typename T, typename U>
    void divideByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_DIVIDE>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void divideByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_DIVIDE>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void divideByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_DIVIDE>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void divideByValue(T* input, size_t pitch_input, U value,
                       T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_DIVIDE>,
                        input, pitch_input, value, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void divideByValue(T* inputs, size_t pitch_inputs, U* values,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_DIVIDE>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void divideByArray(T* inputs, size_t pitch_inputs,
                       U* array, size_t pitch_array,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_DIVIDE>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void divideSafeByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches,
                           Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_DIVIDE_SAFE>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void divideSafeByArray(T* inputs, size_t pitch_inputs,
                           U* array, size_t pitch_array,
                           T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_DIVIDE_SAFE>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    /* ----------- */
    /* --- Add --- */
    /* ----------- */

    template<typename T, typename U>
    void addValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_ADD>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void addValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_ADD>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void addArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_ADD>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void addValue(T* input, size_t pitch_input, U value,
                  T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_ADD>,
                        input, pitch_input, value, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void addValue(T* inputs, size_t pitch_inputs, U* values,
                  T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_ADD>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void addArray(T* inputs, size_t pitch_inputs,
                  U* array, size_t pitch_array,
                  T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_ADD>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    /* ---------------- */
    /* --- Subtract --- */
    /* ---------------- */
    template<typename T, typename U>
    void subtractValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_SUBTRACT>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void subtractValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_SUBTRACT>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void subtractArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_SUBTRACT>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void subtractValue(T* input, size_t pitch_input, U value,
                       T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_SUBTRACT>,
                        input, pitch_input, value, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void subtractValue(T* inputs, size_t pitch_inputs, U* values,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeSingleValue<NOA_OP_SUBTRACT>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void subtractArray(T* inputs, size_t pitch_inputs,
                       U* array, size_t pitch_array,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(dim3(total_blocks, batches), threads_per_block, 0, stream.get(),
                        Kernels::computeElementWise<NOA_OP_SUBTRACT>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape.x, getRows(shape));
    }
}

// INSTANTIATE:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_ARITH_OPERATORS(T, U)                                                       \
    template void multiplyByValue<T, U>(T*, U, T*, size_t, Stream&);                                \
    template void multiplyByValue<T, U>(T*, U*, T*, size_t, uint, Stream&);                         \
    template void multiplyByValue<T, U>(T*, size_t, U, T*, size_t, size3_t, Stream&);               \
    template void multiplyByValue<T, U>(T*, size_t, U*, T*, size_t, size3_t, uint, Stream&);        \
    template void multiplyByArray<T, U>(T*, U*, T*, size_t, uint, Stream&);                         \
    template void multiplyByArray<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&);\
    template void divideByValue<T, U>(T*, U, T*, size_t, Stream&);                                  \
    template void divideByValue<T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void divideByValue<T, U>(T*, size_t, U, T*, size_t, size3_t, Stream&);                 \
    template void divideByValue<T, U>(T*, size_t, U*, T*, size_t, size3_t, uint, Stream&);          \
    template void divideByArray<T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void divideByArray<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&);  \
    template void addValue<T, U>(T*, U, T*, size_t, Stream&);                                       \
    template void addValue<T, U>(T*, U*, T*, size_t, uint, Stream&);                                \
    template void addValue<T, U>(T*, size_t, U, T*, size_t, size3_t, Stream&);                      \
    template void addValue<T, U>(T*, size_t, U*, T*, size_t, size3_t, uint, Stream&);               \
    template void addArray<T, U>(T*, U*, T*, size_t, uint, Stream&);                                \
    template void addArray<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&);       \
    template void subtractValue<T, U>(T*, U, T*, size_t, Stream&);                                  \
    template void subtractValue<T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void subtractValue<T, U>(T*, size_t, U, T*, size_t, size3_t, Stream&);                 \
    template void subtractValue<T, U>(T*, size_t, U*, T*, size_t, size3_t, uint, Stream&);          \
    template void subtractArray<T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void subtractArray<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_ARITH_OPERATORS(float, float);
    INSTANTIATE_ARITH_OPERATORS(double, double);
    INSTANTIATE_ARITH_OPERATORS(int32_t, int32_t);
    INSTANTIATE_ARITH_OPERATORS(uint32_t, uint32_t);
    INSTANTIATE_ARITH_OPERATORS(cfloat_t, cfloat_t);
    INSTANTIATE_ARITH_OPERATORS(cfloat_t, float);
    INSTANTIATE_ARITH_OPERATORS(cdouble_t, cdouble_t);
    INSTANTIATE_ARITH_OPERATORS(cdouble_t, double);

    #define INSTANTIATE_DIVIDE_SAFE(T, U)                                                               \
    template void divideSafeByArray<T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void divideSafeByArray<T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_DIVIDE_SAFE(float, float);
    INSTANTIATE_DIVIDE_SAFE(double, double);
    INSTANTIATE_DIVIDE_SAFE(cfloat_t, float);
    INSTANTIATE_DIVIDE_SAFE(cdouble_t, double);
}
