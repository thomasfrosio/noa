#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Generics.h"

namespace {
    using namespace noa;

    template<int GEN, typename T, typename R>
    __forceinline__ __device__ R getValue_(T value) {
        if constexpr (GEN == cuda::math::details::GEN_ONE_MINUS) {
            return T(1) - value;
        } else if constexpr (GEN == cuda::math::details::GEN_INVERSE) {
            return T(1) / value;
        } else if constexpr (GEN == cuda::math::details::GEN_SQUARE) {
            return value * value;
        } else if constexpr (GEN == cuda::math::details::GEN_SQRT) {
            return noa::math::sqrt(value);
        } else if constexpr (GEN == cuda::math::details::GEN_RSQRT) {
            return noa::math::rsqrt(value);
        } else if constexpr (GEN == cuda::math::details::GEN_EXP) {
            return noa::math::exp(value);
        } else if constexpr (GEN == cuda::math::details::GEN_LOG) {
            return noa::math::log(value);
        } else if constexpr (GEN == cuda::math::details::GEN_ABS) {
            return noa::math::abs(value);
        } else if constexpr (GEN == cuda::math::details::GEN_COS) {
            return noa::math::cos(value);
        } else if constexpr (GEN == cuda::math::details::GEN_SIN) {
            return noa::math::sin(value);
        } else if constexpr (GEN == cuda::math::details::GEN_NORMALIZE) {
            return noa::math::normalize(value);
        } else if constexpr (GEN == cuda::math::details::GEN_REAL) {
            return noa::math::real(value);
        } else if constexpr (GEN == cuda::math::details::GEN_IMAG) {
            return noa::math::imag(value);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return R(0); // unreachable
    }

    template<int GEN, typename T, typename R>
    __forceinline__ __device__ R getValue_(T lhs, T rhs) {
        if constexpr (GEN == cuda::math::details::GEN_POW) {
            return noa::math::pow(lhs, rhs);
        } else if constexpr (GEN == cuda::math::details::GEN_MIN) {
            return noa::math::min(lhs, rhs);
        } else if constexpr (GEN == cuda::math::details::GEN_MAX) {
            return noa::math::max(lhs, rhs);
        } else if constexpr (GEN == cuda::math::details::GEN_COMPLEX) {
            return noa::Complex<T>(lhs, rhs);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return R(0); // unreachable
    }

    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 16384;
            uint total_blocks = noa::math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
            return total_blocks;
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, R* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC, T, R>(input[idx]);
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, T value, R* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC, T, R>(input[idx], value);
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, const T* array, R* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC, T, R>(input[idx], array[idx]);
        }

        template<typename T>
        __global__ void clamp_(const T* input, T low, T high, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = noa::math::clamp(input[idx], low, high);
        }

        template<typename T>
        __global__ void realAndImag_(const Complex<T>* input, T* output_real, T* output_imag, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x) {
                output_real[idx] = input[idx].real;
                output_imag[idx] = input[idx].imag;
            }
        }
    }

    namespace padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024;
            constexpr uint WARPS = BLOCK_SIZE.y;
            return noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, uint input_pitch, R* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC, T, R>(input[row * input_pitch + idx]);
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, uint input_pitch, T value,
                                        R* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC, T, R>(input[row * input_pitch + idx], value);
        }

        template<int GENERIC, typename T, typename R>
        __global__ void computeGeneric_(const T* input, uint input_pitch,
                                        const T* array, uint array_pitch,
                                        R* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC, T, R>(input[row * input_pitch + idx],
                                                                                array[row * array_pitch + idx]);
        }

        template<typename T>
        __global__ void clamp_(const T* input, uint input_pitch, T low, T high,
                               T* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = noa::math::clamp(input[row * input_pitch + idx], low, high);
        }

        template<typename T>
        __global__ void realAndImag_(const Complex<T>* input, uint input_pitch,
                                     T* output_real, uint output_real_pitch,
                                     T* output_imag, uint output_imag_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    output_real[row * output_real_pitch + idx] = input[row * input_pitch + idx].real;
                    output_imag[row * output_imag_pitch + idx] = input[row * input_pitch + idx].imag;
                }
            }
        }
    }
}

namespace noa::cuda::math::details {
    template<int GEN, typename T, typename R>
    void generic(const T* input, R* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN, T, R><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T, typename R>
    void genericWithValue(const T* input, T value, R* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN, T, R><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, value, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T, typename R>
    void genericWithArray(const T* input, const T* array, R* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN, T, R><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, array, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T, typename R>
    void generic(const T* input, size_t input_pitch, R* output, size_t output_pitch, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN, T, R><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T, typename R>
    void genericWithValue(const T* input, size_t input_pitch, T value, R* output, size_t output_pitch,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN, T, R><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, value, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T, typename R>
    void genericWithArray(const T* input, size_t input_pitch,
                          const T* array, size_t array_pitch,
                          R* output, size_t output_pitch,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN, T, R><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, array, array_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace noa::cuda::math {
    template<typename T>
    void realAndImag(const Complex<T>* input, T* output_real, T* output_imag, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::realAndImag_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, output_real, output_imag, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void realAndImag(const Complex<T>* input, size_t input_pitch,
                     T* output_real, size_t output_real_pitch,
                     T* output_imag, size_t output_imag_pitch,
                     size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::realAndImag_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, output_real, output_real_pitch, output_imag, output_imag_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void clamp(const T* input, T low, T high, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::clamp_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, low, high, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void clamp(const T* input, size_t input_pitch, T low, T high, T* output, size_t output_pitch,
               size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::clamp_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, low, high, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// INSTANTIATE:
namespace noa::cuda::math::details {
    #define NOA_INSTANTIATE_ONE_MINUS_ABS_(T)                                                   \
    template void generic<GEN_ONE_MINUS, T, T>(const T*, T*, size_t, Stream&);                  \
    template void generic<GEN_ONE_MINUS, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&); \
    template void generic<GEN_ABS, T, T>(const T*, T*, size_t, Stream&);                        \
    template void generic<GEN_ABS, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_ONE_MINUS_ABS_(float);
    NOA_INSTANTIATE_ONE_MINUS_ABS_(double);
    NOA_INSTANTIATE_ONE_MINUS_ABS_(short);
    NOA_INSTANTIATE_ONE_MINUS_ABS_(int);
    NOA_INSTANTIATE_ONE_MINUS_ABS_(long);
    NOA_INSTANTIATE_ONE_MINUS_ABS_(long long);

    #define NOA_INSTANTIATE_SQUARE_(T)                                                      \
    template void generic<GEN_SQUARE, T, T>(const T*, T*, size_t, Stream&);                 \
    template void generic<GEN_SQUARE, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_SQUARE_(float);
    NOA_INSTANTIATE_SQUARE_(double);
    NOA_INSTANTIATE_SQUARE_(short);
    NOA_INSTANTIATE_SQUARE_(int);
    NOA_INSTANTIATE_SQUARE_(long);
    NOA_INSTANTIATE_SQUARE_(long long);
    NOA_INSTANTIATE_SQUARE_(unsigned short);
    NOA_INSTANTIATE_SQUARE_(unsigned int);
    NOA_INSTANTIATE_SQUARE_(unsigned long);
    NOA_INSTANTIATE_SQUARE_(unsigned long long);

    #define NOA_INSTANTIATE_FP_(T)                                                                  \
    template void generic<GEN_INVERSE, T, T>(const T*, T*, size_t, Stream&);                        \
    template void generic<GEN_INVERSE, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);       \
    template void generic<GEN_SQRT, T, T>(const T*, T*, size_t, Stream&);                           \
    template void generic<GEN_SQRT, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);          \
    template void generic<GEN_RSQRT, T, T>(const T*, T*, size_t, Stream&);                          \
    template void generic<GEN_RSQRT, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);         \
    template void generic<GEN_EXP, T, T>(const T*, T*, size_t, Stream&);                            \
    template void generic<GEN_EXP, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);           \
    template void generic<GEN_LOG, T, T>(const T*, T*, size_t, Stream&);                            \
    template void generic<GEN_LOG, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);           \
    template void generic<GEN_COS, T, T>(const T*, T*, size_t, Stream&);                            \
    template void generic<GEN_COS, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);           \
    template void generic<GEN_SIN, T, T>(const T*, T*, size_t, Stream&);                            \
    template void generic<GEN_SIN, T, T>(const T*, size_t, T*, size_t, size3_t, Stream&);           \
    template void genericWithValue<GEN_POW, T, T>(const T*, T, T*, size_t, Stream&);                \
    template void genericWithValue<GEN_POW, T, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_FP_(float);
    NOA_INSTANTIATE_FP_(double);

    #define NOA_INSTANTIATE_MIN_MAX_(T)                                                                                 \
    template void genericWithValue<GEN_MIN, T, T>(const T*, T, T*, size_t, Stream&);                                    \
    template void genericWithValue<GEN_MIN, T, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&);                   \
    template void genericWithArray<GEN_MIN, T, T>(const T*, const T*, T*, size_t, Stream&);                             \
    template void genericWithArray<GEN_MIN, T, T>(const T*, size_t, const T*, size_t, T*, size_t, size3_t, Stream&);    \
    template void genericWithValue<GEN_MAX, T, T>(const T*, T, T*, size_t, Stream&);                                    \
    template void genericWithValue<GEN_MAX, T, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&);                   \
    template void genericWithArray<GEN_MAX, T, T>(const T*, const T*, T*, size_t, Stream&);                             \
    template void genericWithArray<GEN_MAX, T, T>(const T*, size_t, const T*, size_t, T*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_MIN_MAX_(float);
    NOA_INSTANTIATE_MIN_MAX_(double);
    NOA_INSTANTIATE_MIN_MAX_(char);
    NOA_INSTANTIATE_MIN_MAX_(short);
    NOA_INSTANTIATE_MIN_MAX_(int);
    NOA_INSTANTIATE_MIN_MAX_(long);
    NOA_INSTANTIATE_MIN_MAX_(long long);
    NOA_INSTANTIATE_MIN_MAX_(unsigned char);
    NOA_INSTANTIATE_MIN_MAX_(unsigned short);
    NOA_INSTANTIATE_MIN_MAX_(unsigned int);
    NOA_INSTANTIATE_MIN_MAX_(unsigned long);
    NOA_INSTANTIATE_MIN_MAX_(unsigned long long);
}

namespace noa::cuda::math::details {
    #define NOA_INSTANTIATE_COMPLEX_(T)                                                                                                     \
    template void generic<GEN_ONE_MINUS, Complex<T>, Complex<T>>(const Complex<T>*, Complex<T>*, size_t, Stream&);                          \
    template void generic<GEN_ONE_MINUS, Complex<T>, Complex<T>>(const Complex<T>*, size_t, Complex<T>*, size_t, size3_t, Stream&);         \
    template void generic<GEN_INVERSE, Complex<T>, Complex<T>>(const Complex<T>*, Complex<T>*, size_t, Stream&);                        \
    template void generic<GEN_INVERSE, Complex<T>, Complex<T>>(const Complex<T>*, size_t, Complex<T>*, size_t, size3_t, Stream&);       \
    template void generic<GEN_ABS, Complex<T>, T>(const Complex<T>*, T*, size_t, Stream&);                                                  \
    template void generic<GEN_ABS, Complex<T>, T>(const Complex<T>*, size_t, T*, size_t, size3_t, Stream&);                                 \
    template void generic<GEN_SQUARE, Complex<T>, Complex<T>>(const Complex<T>*, Complex<T>*, size_t, Stream&);                             \
    template void generic<GEN_SQUARE, Complex<T>, Complex<T>>(const Complex<T>*, size_t, Complex<T>*, size_t, size3_t, Stream&);            \
    template void generic<GEN_NORMALIZE, Complex<T>, Complex<T>>(const Complex<T>*, Complex<T>*, size_t, Stream&);                          \
    template void generic<GEN_NORMALIZE, Complex<T>, Complex<T>>(const Complex<T>*, size_t, Complex<T>*, size_t, size3_t, Stream&);         \
    template void generic<GEN_REAL, Complex<T>, T>(const Complex<T>*, T*, size_t, Stream&);                                                 \
    template void generic<GEN_REAL, Complex<T>, T>(const Complex<T>*, size_t, T*, size_t, size3_t, Stream&);                                \
    template void generic<GEN_IMAG, Complex<T>, T>(const Complex<T>*, T*, size_t, Stream&);                                                 \
    template void generic<GEN_IMAG, Complex<T>, T>(const Complex<T>*, size_t, T*, size_t, size3_t, Stream&);                                \
    template void genericWithArray<GEN_COMPLEX, T, Complex<T>>(const T*, const T*, Complex<T>*, size_t, Stream&);                           \
    template void genericWithArray<GEN_COMPLEX, T, Complex<T>>(const T*, size_t, const T*, size_t, Complex<T>*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_COMPLEX_(float);
    NOA_INSTANTIATE_COMPLEX_(double);
}

namespace noa::cuda::math {
    template void realAndImag<float>(const cfloat_t*, float*, float*, size_t, Stream&);
    template void realAndImag<double>(const cdouble_t*, double*, double*, size_t, Stream&);
    template void realAndImag<float>(const cfloat_t*, size_t, float*, size_t, float*, size_t, size3_t, Stream&);
    template void realAndImag<double>(const cdouble_t*, size_t, double*, size_t, double*, size_t, size3_t, Stream&);

    #define NOA_INSTANTIATE_CLAMP_(T)                                               \
    template void clamp<T>(const T*, T, T, T*, size_t, Stream&);                    \
    template void clamp<T>(const T*, size_t, T, T, T*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_CLAMP_(float);
    NOA_INSTANTIATE_CLAMP_(double);
    NOA_INSTANTIATE_CLAMP_(char);
    NOA_INSTANTIATE_CLAMP_(short);
    NOA_INSTANTIATE_CLAMP_(int);
    NOA_INSTANTIATE_CLAMP_(long);
    NOA_INSTANTIATE_CLAMP_(long long);
    NOA_INSTANTIATE_CLAMP_(unsigned char);
    NOA_INSTANTIATE_CLAMP_(unsigned short);
    NOA_INSTANTIATE_CLAMP_(unsigned int);
    NOA_INSTANTIATE_CLAMP_(unsigned long);
    NOA_INSTANTIATE_CLAMP_(unsigned long long);
}
