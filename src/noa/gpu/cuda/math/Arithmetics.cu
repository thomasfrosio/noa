#include "noa/gpu/cuda/math/Arithmetics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace Noa::CUDA::Math::Details {
    enum : int { ARITH_ADD, ARITH_SUBTRACT, ARITH_MULTIPLY, ARITH_DIVIDE, ARITH_DIVIDE_SAFE };

    template<int OPERATOR, typename T, typename U>
    NOA_FD T computeArith(T lhs, U rhs) {
        T out;
        if constexpr (OPERATOR == ARITH_ADD) {
            out = lhs + rhs;
        } else if constexpr (OPERATOR == ARITH_SUBTRACT) {
            out = lhs - rhs;
        } else if constexpr (OPERATOR == ARITH_MULTIPLY) {
            out = lhs * rhs;
        } else if constexpr (OPERATOR == ARITH_DIVIDE) {
            out = lhs / rhs;
        } else if constexpr (OPERATOR == ARITH_DIVIDE_SAFE) {
            out = Noa::Math::abs(rhs) < 1e-15 ? T(0) : lhs / rhs;
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out; // https://stackoverflow.com/questions/64523302
    }
}

namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 256;

    // One block computes its elements and go to the corresponding
    // elements in next grid, until the end, for each batch.
    uint getBlocks(uint elements) {
        constexpr uint MAX_BLOCKS = 32768;
        return Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    }

    template<int OPERATOR, typename T, typename U>
    __global__ void computeSingleValue(T* input, U value, T* output, uint elements) {
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
            output[idx] = computeArith<OPERATOR>(input[idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    __global__ void computeSingleValue(T* inputs, U* values, T* outputs, uint elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        U value = values[blockIdx.y];
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
            tmp_out[idx] = computeArith<OPERATOR>(tmp_in[idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    __global__ void computeElementWise(T* inputs, U* array, T* outputs, uint elements) {
        T* tmp_in = inputs + blockIdx.y * elements;
        T* tmp_out = outputs + blockIdx.y * elements;
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
            tmp_out[idx] = computeArith<OPERATOR>(tmp_in[idx], array[idx]);
    }
}

namespace Noa::CUDA::Math::Details::Padded {
    static constexpr dim3 BLOCK_SIZE(32, 8);

    uint getBlocks(uint2_t shape_2d) {
        constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
        return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
    }

    template<int OPERATOR, typename T, typename U>
    __global__ void computeSingleValue(T* input, uint pitch_input, U value,
                                       T* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = computeArith<OPERATOR>(input[row * pitch_input + idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    __global__ void computeSingleValue(T* inputs, uint pitch_inputs, U* values,
                                       T* outputs, uint pitch_outputs, uint2_t shape) {
        inputs += blockIdx.y * pitch_inputs * shape.y;
        outputs += blockIdx.y * pitch_outputs * shape.y;
        U value = values[blockIdx.y];
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                outputs[row * pitch_outputs + idx] = computeArith<OPERATOR>(inputs[row * pitch_inputs + idx], value);
    }

    template<int OPERATOR, typename T, typename U>
    static __global__ void computeElementWise(T* inputs, uint pitch_inputs, U* array, uint pitch_array,
                                              T* outputs, uint pitch_outputs, uint2_t shape) {
        inputs += blockIdx.y * pitch_inputs * shape.y;
        outputs += blockIdx.y * pitch_outputs * shape.y;
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                outputs[row * pitch_outputs + idx] = computeArith<OPERATOR>(inputs[row * pitch_inputs + idx],
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
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_MULTIPLY>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void multiplyByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_MULTIPLY>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void multiplyByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeElementWise<Details::ARITH_MULTIPLY>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void multiplyByValue(T* input, size_t pitch_input, U value,
                         T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_MULTIPLY>,
                        input, pitch_input, value, output, pitch_output, shape2d);
    }

    template<typename T, typename U>
    void multiplyByValue(T* inputs, size_t pitch_inputs, U* values,
                         T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_MULTIPLY>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape2d);
    }

    template<typename T, typename U>
    void multiplyByArray(T* inputs, size_t pitch_inputs,
                         U* array, size_t pitch_array,
                         T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeElementWise<Details::ARITH_MULTIPLY>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
    }

    /* -------------- */
    /* --- Divide --- */
    /* -------------- */

    template<typename T, typename U>
    void divideByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_DIVIDE>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void divideByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_DIVIDE>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void divideByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeElementWise<Details::ARITH_DIVIDE>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void divideByValue(T* input, size_t pitch_input, U value,
                       T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_DIVIDE>,
                        input, pitch_input, value, output, pitch_output, shape2d);
    }

    template<typename T, typename U>
    void divideByValue(T* inputs, size_t pitch_inputs, U* values,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_DIVIDE>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape2d);
    }

    template<typename T, typename U>
    void divideByArray(T* inputs, size_t pitch_inputs,
                       U* array, size_t pitch_array,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeElementWise<Details::ARITH_DIVIDE>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
    }

    template<typename T, typename U>
    void divideSafeByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches,
                           Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeElementWise<Details::ARITH_DIVIDE_SAFE>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void divideSafeByArray(T* inputs, size_t pitch_inputs,
                           U* array, size_t pitch_array,
                           T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeElementWise<Details::ARITH_DIVIDE_SAFE>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
    }

    /* ----------- */
    /* --- Add --- */
    /* ----------- */

    template<typename T, typename U>
    void addValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_ADD>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void addValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_ADD>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void addArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeElementWise<Details::ARITH_ADD>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void addValue(T* input, size_t pitch_input, U value,
                  T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_ADD>,
                        input, pitch_input, value, output, pitch_output, shape2d);
    }

    template<typename T, typename U>
    void addValue(T* inputs, size_t pitch_inputs, U* values,
                  T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_ADD>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape2d);
    }

    template<typename T, typename U>
    void addArray(T* inputs, size_t pitch_inputs,
                  U* array, size_t pitch_array,
                  T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeElementWise<Details::ARITH_ADD>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
    }

    /* ---------------- */
    /* --- Subtract --- */
    /* ---------------- */

    template<typename T, typename U>
    void subtractValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_SUBTRACT>,
                        input, value, output, elements);
    }

    template<typename T, typename U>
    void subtractValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeSingleValue<Details::ARITH_SUBTRACT>,
                        inputs, values, outputs, elements);
    }

    template<typename T, typename U>
    void subtractArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeElementWise<Details::ARITH_SUBTRACT>,
                        inputs, array, outputs, elements);
    }

    template<typename T, typename U>
    void subtractValue(T* input, size_t pitch_input, U value,
                       T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_SUBTRACT>,
                        input, pitch_input, value, output, pitch_output, shape2d);
    }

    template<typename T, typename U>
    void subtractValue(T* inputs, size_t pitch_inputs, U* values,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeSingleValue<Details::ARITH_SUBTRACT>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape2d);
    }

    template<typename T, typename U>
    void subtractArray(T* inputs, size_t pitch_inputs,
                       U* array, size_t pitch_array,
                       T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeElementWise<Details::ARITH_SUBTRACT>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
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
