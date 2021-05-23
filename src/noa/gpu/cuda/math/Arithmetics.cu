#include "noa/gpu/cuda/math/Arithmetics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace {
    using namespace Noa;

    template<int OPERATOR, typename T, typename U>
    __forceinline__ __device__ T getArith_(T lhs, U rhs) {
        T out;
        if constexpr (OPERATOR == CUDA::Math::Details::ARITH_ADD) {
            out = lhs + rhs;
        } else if constexpr (OPERATOR == CUDA::Math::Details::ARITH_SUBTRACT) {
            out = lhs - rhs;
        } else if constexpr (OPERATOR == CUDA::Math::Details::ARITH_MULTIPLY) {
            out = lhs * rhs;
        } else if constexpr (OPERATOR == CUDA::Math::Details::ARITH_DIVIDE) {
            out = lhs / rhs;
        } else if constexpr (OPERATOR == CUDA::Math::Details::ARITH_DIVIDE_SAFE) {
            out = Noa::Math::abs(rhs) < 1e-15 ? T(0) : lhs / rhs;
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out; // https://stackoverflow.com/questions/64523302
    }

    namespace Contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        // One block computes its elements and go to the corresponding
        // elements in next grid, until the end, for each batch.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 32768;
            return Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(T* input, U value, T* output, uint elements) {
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                output[idx] = getArith_<OPERATOR>(input[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(T* inputs, U* values, T* outputs, uint elements) {
            T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            U value = values[blockIdx.y];
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeElementWise_(T* inputs, U* array, T* outputs, uint elements) {
            T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], array[idx]);
        }
    }

    namespace Padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(T* input, uint pitch_input, U value,
                                            T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = getArith_<OPERATOR>(input[row * pitch_input + idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(T* inputs, uint pitch_inputs, U* values,
                                            T* outputs, uint pitch_outputs, uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            U value = values[blockIdx.y];
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    outputs[row * pitch_outputs + idx] = getArith_<OPERATOR>(inputs[row * pitch_inputs + idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        static __global__ void computeElementWise_(T* inputs, uint pitch_inputs, U* array, uint pitch_array,
                                                   T* outputs, uint pitch_outputs, uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    outputs[row * pitch_outputs + idx] = getArith_<OPERATOR>(inputs[row * pitch_inputs + idx],
                                                                             array[row * pitch_array + idx]);
        }
    }
}

namespace Noa::CUDA::Math::Details {
    template<int ARITH, typename T, typename U>
    void arithByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(blocks, Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeSingleValue_<ARITH>,
                        input, value, output, elements);
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeSingleValue_<ARITH>,
                        inputs, values, outputs, elements);
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeElementWise_<ARITH>,
                        inputs, array, outputs, elements);
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(T* input, size_t pitch_input, U value,
                      T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape2d);
        NOA_CUDA_LAUNCH(blocks, Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeSingleValue_<ARITH>,
                        input, pitch_input, value, output, pitch_output, shape2d);
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(T* inputs, size_t pitch_inputs, U* values,
                      T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeSingleValue_<ARITH>,
                        inputs, pitch_inputs, values, outputs, pitch_outputs, shape2d);
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(T* inputs, size_t pitch_inputs,
                      U* array, size_t pitch_array,
                      T* outputs, size_t pitch_outputs, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape2d);
        NOA_CUDA_LAUNCH(dim3(blocks, batches), Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeElementWise_<ARITH>,
                        inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs, shape2d);
    }

    #define INSTANTIATE_ARITH_OPERATORS(ARITH, T, U) \
    template void Details::arithByValue<ARITH, T, U>(T*, U, T*, size_t, Stream&);                                \
    template void Details::arithByValue<ARITH, T, U>(T*, U*, T*, size_t, uint, Stream&);                         \
    template void Details::arithByArray<ARITH, T, U>(T*, U*, T*, size_t, uint, Stream&);                         \
    template void Details::arithByValue<ARITH, T, U>(T*, size_t, U, T*, size_t, size3_t, Stream&);               \
    template void Details::arithByValue<ARITH, T, U>(T*, size_t, U*, T*, size_t, size3_t, uint, Stream&);        \
    template void Details::arithByArray<ARITH, T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&)

    #define INSTANTIATE_ARITH(T, U)                                 \
    INSTANTIATE_ARITH_OPERATORS(Details::ARITH_MULTIPLY, T, U);     \
    INSTANTIATE_ARITH_OPERATORS(Details::ARITH_DIVIDE, T, U);       \
    INSTANTIATE_ARITH_OPERATORS(Details::ARITH_ADD, T, U);          \
    INSTANTIATE_ARITH_OPERATORS(Details::ARITH_SUBTRACT, T, U)

    INSTANTIATE_ARITH(float, float);
    INSTANTIATE_ARITH(double, double);
    INSTANTIATE_ARITH(int, int);
    INSTANTIATE_ARITH(uint, uint);
    INSTANTIATE_ARITH(cfloat_t, cfloat_t);
    INSTANTIATE_ARITH(cfloat_t, float);
    INSTANTIATE_ARITH(cdouble_t, cdouble_t);
    INSTANTIATE_ARITH(cdouble_t, double);

    #define INSTANTIATE_DIVIDE_SAFE(T, U)                                                                                               \
    template void Details::arithByArray<Details::ARITH_DIVIDE_SAFE, T, U>(T*, U*, T*, size_t, uint, Stream&);                           \
    template void Details::arithByArray<Details::ARITH_DIVIDE_SAFE, T, U>(T*, size_t, U*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_DIVIDE_SAFE(float, float);
    INSTANTIATE_DIVIDE_SAFE(double, double);
    INSTANTIATE_DIVIDE_SAFE(cfloat_t, float);
    INSTANTIATE_DIVIDE_SAFE(cdouble_t, double);
}
