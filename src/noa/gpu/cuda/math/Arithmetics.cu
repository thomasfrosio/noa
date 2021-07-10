#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Arithmetics.h"

namespace {
    using namespace noa;

    template<int OPERATOR, typename T, typename U>
    __forceinline__ __device__ T getArith_(T lhs, U rhs) {
        T out;
        if constexpr (OPERATOR == cuda::math::details::ARITH_ADD) {
            out = lhs + rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_SUBTRACT) {
            out = lhs - rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_MULTIPLY) {
            out = lhs * rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_DIVIDE) {
            out = lhs / rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_DIVIDE_SAFE) {
            out = noa::math::abs(rhs) < 1e-15 ? T(0) : lhs / rhs;
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out; // https://stackoverflow.com/questions/64523302
    }

    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        // One block computes its elements and go to the corresponding
        // elements in next grid, until the end, for each batch.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 32768;
            return noa::math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(const T* input, U value, T* output, uint elements) {
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                output[idx] = getArith_<OPERATOR>(input[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(const T* inputs, const U* values, T* outputs, uint elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            U value = values[blockIdx.y];
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeElementWise_(const T* inputs, const U* array, T* outputs, uint elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], array[idx]);
        }
    }

    namespace padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            return noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(const T* input, uint pitch_input, U value,
                                            T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = getArith_<OPERATOR>(input[row * pitch_input + idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ void computeSingleValue_(const T* inputs, uint pitch_inputs, const U* values,
                                            T* outputs, uint pitch_outputs, uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            U value = values[blockIdx.y];
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    outputs[row * pitch_outputs + idx] = getArith_<OPERATOR>(inputs[row * pitch_inputs + idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        static __global__ void computeElementWise_(const T* inputs, uint pitch_inputs, const U* array, uint pitch_array,
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

namespace noa::cuda::math::details {
    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeSingleValue_<ARITH><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, value, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, const U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeSingleValue_<ARITH><<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, values, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, const U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeElementWise_<ARITH><<<dim3(blocks, batches), contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, array, outputs, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, size_t input_pitch, U value,
                      T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape2d);
        padded_::computeSingleValue_<ARITH><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, value, output, output_pitch, shape2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, size_t inputs_pitch, const U* values,
                      T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape2d);
        padded_::computeSingleValue_<ARITH><<<dim3(blocks, batches), padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, values, outputs, outputs_pitch, shape2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, size_t inputs_pitch,
                      const U* array, size_t array_pitch,
                      T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream) {
        uint2_t shape2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape2d);
        padded_::computeElementWise_<ARITH><<<dim3(blocks, batches), padded_::BLOCK_SIZE, 0, stream.get()>>>(
                inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch, shape2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_ARITH_OPERATORS(ARITH, T, U)                                                                        \
    template void details::arithByValue<ARITH, T, U>(const T*, U, T*, size_t, Stream&);                                     \
    template void details::arithByValue<ARITH, T, U>(const T*, const U*, T*, size_t, uint, Stream&);                        \
    template void details::arithByArray<ARITH, T, U>(const T*, const U*, T*, size_t, uint, Stream&);                        \
    template void details::arithByValue<ARITH, T, U>(const T*, size_t, U, T*, size_t, size3_t, Stream&);                    \
    template void details::arithByValue<ARITH, T, U>(const T*, size_t, const U*, T*, size_t, size3_t, uint, Stream&);       \
    template void details::arithByArray<ARITH, T, U>(const T*, size_t, const U*, size_t, T*, size_t, size3_t, uint, Stream&)

    #define INSTANTIATE_ARITH(T, U)                                 \
    INSTANTIATE_ARITH_OPERATORS(details::ARITH_MULTIPLY, T, U);     \
    INSTANTIATE_ARITH_OPERATORS(details::ARITH_DIVIDE, T, U);       \
    INSTANTIATE_ARITH_OPERATORS(details::ARITH_ADD, T, U);          \
    INSTANTIATE_ARITH_OPERATORS(details::ARITH_SUBTRACT, T, U)

    INSTANTIATE_ARITH(float, float);
    INSTANTIATE_ARITH(double, double);
    INSTANTIATE_ARITH(int, int);
    INSTANTIATE_ARITH(uint, uint);
    INSTANTIATE_ARITH(cfloat_t, cfloat_t);
    INSTANTIATE_ARITH(cfloat_t, float);
    INSTANTIATE_ARITH(cdouble_t, cdouble_t);
    INSTANTIATE_ARITH(cdouble_t, double);

    #define INSTANTIATE_DIVIDE_SAFE(T, U)                                                                                                           \
    template void details::arithByArray<details::ARITH_DIVIDE_SAFE, T, U>(const T*, const U*, T*, size_t, uint, Stream&);                           \
    template void details::arithByArray<details::ARITH_DIVIDE_SAFE, T, U>(const T*, size_t, const U*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_DIVIDE_SAFE(float, float);
    INSTANTIATE_DIVIDE_SAFE(double, double);
    INSTANTIATE_DIVIDE_SAFE(cfloat_t, float);
    INSTANTIATE_DIVIDE_SAFE(cdouble_t, double);
}