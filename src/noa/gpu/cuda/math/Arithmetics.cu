#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Arithmetics.h"

namespace {
    using namespace noa;

    template<int OPERATOR, typename T, typename U>
    __forceinline__ __device__ T getArith_(T lhs, U rhs) {
        if constexpr (OPERATOR == cuda::math::details::ARITH_ADD) {
            return lhs + rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_SUBTRACT) {
            return lhs - rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_MULTIPLY) {
            return lhs * rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_DIVIDE) {
            return lhs / rhs;
        } else if constexpr (OPERATOR == cuda::math::details::ARITH_DIVIDE_SAFE) {
            if constexpr (noa::traits::is_float_v<U>)
                return math::abs(rhs) < math::Limits<U>::epsilon() ? static_cast<T>(0) : lhs / rhs;
            else if constexpr (std::is_integral_v<U>)
                return rhs == 0 ? 0 : lhs / rhs;
            else
                    static_assert(noa::traits::always_false_v<T>);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return T(0); // unreachable, https://stackoverflow.com/questions/64523302
    }

    namespace contiguous_ {
        constexpr uint THREADS = 512;

        // One block computes its elements and go to the corresponding
        // elements in next grid, until the end, for each batch.
        uint getBlocks_(uint elements) {
            constexpr uint MAX_BLOCKS = 32768;
            return noa::math::min(noa::math::divideUp(elements, THREADS), MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void computeSingleValue_(const T* input, U value, T* output, uint elements) {
            #pragma unroll 15
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x)
                output[idx] = getArith_<OPERATOR>(input[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void computeSingleValue_(const T* inputs, const U* values, T* outputs, uint elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            U value = values[blockIdx.y];
            #pragma unroll 15
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], value);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void computeElementWise_(const T* inputs, const U* array, T* outputs, uint elements) {
            const T* tmp_in = inputs + blockIdx.y * elements;
            T* tmp_out = outputs + blockIdx.y * elements;
            #pragma unroll 15
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x)
                tmp_out[idx] = getArith_<OPERATOR>(tmp_in[idx], array[idx]);
        }
    }

    namespace padded_ {
        constexpr dim3 THREADS(32, 16);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            // warps per block; every warp processes at least one row.
            return noa::math::min(noa::math::divideUp(shape_2d.y, THREADS.y), MAX_BLOCKS);
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void computeSingleValue_(const T* input, uint pitch_input, U value,
                                 T* output, uint pitch_output, uint2_t shape) {
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    output[row * pitch_output + idx] = getArith_<OPERATOR>(input[row * pitch_input + idx], value);
            }
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void computeSingleValue_(const T* inputs, uint pitch_inputs, const U* values,
                                 T* outputs, uint pitch_outputs, uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            U value = values[blockIdx.y];
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    outputs[row * pitch_outputs + idx] = getArith_<OPERATOR>(inputs[row * pitch_inputs + idx], value);
            }
        }

        template<int OPERATOR, typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void computeElementWise_(const T* inputs, uint pitch_inputs, const U* array, uint pitch_array,
                                 T* outputs, uint pitch_outputs, uint2_t shape) {
            inputs += blockIdx.y * pitch_inputs * shape.y;
            outputs += blockIdx.y * pitch_outputs * shape.y;
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    outputs[row * pitch_outputs + idx] = getArith_<OPERATOR>(inputs[row * pitch_inputs + idx],
                                                                             array[row * pitch_array + idx]);
            }
        }
    }
}

namespace noa::cuda::math::details {
    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeSingleValue_<ARITH><<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                input, value, output, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, const U* values, T* outputs, size_t elements, size_t batches, Stream& stream) {
        dim3 blocks(contiguous_::getBlocks_(elements), batches);
        contiguous_::computeSingleValue_<ARITH><<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                inputs, values, outputs, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, const U* array, T* outputs, size_t elements, size_t batches, Stream& stream) {
        dim3 blocks(contiguous_::getBlocks_(elements), batches);
        contiguous_::computeElementWise_<ARITH><<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                inputs, array, outputs, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, size_t input_pitch, U value,
                      T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        uint2_t shape2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape2d);
        padded_::computeSingleValue_<ARITH><<<blocks, padded_::THREADS, 0, stream.get()>>>(
                input, input_pitch, value, output, output_pitch, shape2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, size_t inputs_pitch, const U* values,
                      T* outputs, size_t outputs_pitch, size3_t shape, size_t batches, Stream& stream) {
        uint2_t shape2d(shape.x, rows(shape));
        dim3 blocks(padded_::getBlocks_(shape2d), batches);
        padded_::computeSingleValue_<ARITH><<<blocks, padded_::THREADS, 0, stream.get()>>>(
                inputs, inputs_pitch, values, outputs, outputs_pitch, shape2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, size_t inputs_pitch,
                      const U* array, size_t array_pitch,
                      T* outputs, size_t outputs_pitch, size3_t shape, size_t batches, Stream& stream) {
        uint2_t shape2d(shape.x, rows(shape));
        dim3 blocks(padded_::getBlocks_(shape2d), batches);
        padded_::computeElementWise_<ARITH><<<blocks, padded_::THREADS, 0, stream.get()>>>(
                inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch, shape2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_ARITH_OPERATORS_(ARITH, T, U)                                                               \
    template void details::arithByValue<ARITH, T, U>(const T*, U, T*, size_t, Stream&);                                 \
    template void details::arithByValue<ARITH, T, U>(const T*, const U*, T*, size_t, size_t, Stream&);                  \
    template void details::arithByArray<ARITH, T, U>(const T*, const U*, T*, size_t, size_t, Stream&);                  \
    template void details::arithByValue<ARITH, T, U>(const T*, size_t, U, T*, size_t, size3_t, Stream&);                \
    template void details::arithByValue<ARITH, T, U>(const T*, size_t, const U*, T*, size_t, size3_t, size_t, Stream&); \
    template void details::arithByArray<ARITH, T, U>(const T*, size_t, const U*, size_t, T*, size_t, size3_t, size_t, Stream&)

    #define NOA_INSTANTIATE_ARITH_(T, U)                                 \
    NOA_INSTANTIATE_ARITH_OPERATORS_(details::ARITH_MULTIPLY, T, U);     \
    NOA_INSTANTIATE_ARITH_OPERATORS_(details::ARITH_DIVIDE, T, U);       \
    NOA_INSTANTIATE_ARITH_OPERATORS_(details::ARITH_ADD, T, U);          \
    NOA_INSTANTIATE_ARITH_OPERATORS_(details::ARITH_SUBTRACT, T, U)

    NOA_INSTANTIATE_ARITH_(float, float);
    NOA_INSTANTIATE_ARITH_(double, double);
    NOA_INSTANTIATE_ARITH_(int, int);
    NOA_INSTANTIATE_ARITH_(long, long);
    NOA_INSTANTIATE_ARITH_(long long, long long);
    NOA_INSTANTIATE_ARITH_(unsigned int, unsigned int);
    NOA_INSTANTIATE_ARITH_(unsigned long, unsigned long);
    NOA_INSTANTIATE_ARITH_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_ARITH_(cfloat_t, cfloat_t);
    NOA_INSTANTIATE_ARITH_(cfloat_t, float);
    NOA_INSTANTIATE_ARITH_(cdouble_t, cdouble_t);
    NOA_INSTANTIATE_ARITH_(cdouble_t, double);

    #define NOA_INSTANTIATE_DIVIDE_SAFE_(T, U)                                                                                  \
    template void details::arithByArray<details::ARITH_DIVIDE_SAFE, T, U>(const T*, const U*, T*, size_t, size_t, Stream&);     \
    template void details::arithByArray<details::ARITH_DIVIDE_SAFE, T, U>(const T*, size_t, const U*, size_t, T*, size_t, size3_t, size_t, Stream&)

    NOA_INSTANTIATE_DIVIDE_SAFE_(float, float);
    NOA_INSTANTIATE_DIVIDE_SAFE_(double, double);
    NOA_INSTANTIATE_DIVIDE_SAFE_(int, int);
    NOA_INSTANTIATE_DIVIDE_SAFE_(long, long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(long long, long long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned int, unsigned int);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned long, unsigned long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_DIVIDE_SAFE_(cfloat_t, float);
    NOA_INSTANTIATE_DIVIDE_SAFE_(cdouble_t, double);
}
