#include "noa/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Generics.h"

namespace {
    using namespace noa;

    template<int GEN, typename T>
    __forceinline__ __device__ T getValue_(T value) {
        T out;
        if constexpr (GEN == cuda::math::details::GEN_ONE_MINUS) {
            out = T(1) - value;
        } else if constexpr (GEN == cuda::math::details::GEN_INVERSE) {
            out = T(1) / value;
        } else if constexpr (GEN == cuda::math::details::GEN_SQUARE) {
            out = value * value;
        } else if constexpr (GEN == cuda::math::details::GEN_SQRT) {
            out = noa::math::sqrt(value);
        } else if constexpr (GEN == cuda::math::details::GEN_RSQRT) {
            out = noa::math::rsqrt(value);
        } else if constexpr (GEN == cuda::math::details::GEN_EXP) {
            out = noa::math::exp(value);
        } else if constexpr (GEN == cuda::math::details::GEN_LOG) {
            out = noa::math::log(value);
        } else if constexpr (GEN == cuda::math::details::GEN_ABS) {
            out = noa::math::abs(value);
        } else if constexpr (GEN == cuda::math::details::GEN_COS) {
            out = noa::math::cos(value);
        } else if constexpr (GEN == cuda::math::details::GEN_SIN) {
            out = noa::math::sin(value);
        } else if constexpr (GEN == cuda::math::details::GEN_NORMALIZE) {
            out = noa::math::normalize(value);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    template<int GEN, typename T>
    __forceinline__ __device__ T getValue_(T lhs, T rhs) {
        T out;
        if constexpr (GEN == cuda::math::details::GEN_POW) {
            out = noa::math::pow(lhs, rhs);
        } else if constexpr (GEN == cuda::math::details::GEN_MIN) {
            out = noa::math::min(lhs, rhs);
        } else if constexpr (GEN == cuda::math::details::GEN_MAX) {
            out = noa::math::max(lhs, rhs);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        return out;
    }

    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 16384;
            uint total_blocks = noa::math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
            return total_blocks;
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx]);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, T value, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx], value);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, const T* array, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx], array[idx]);
        }

        template<typename T>
        __global__ void clamp_(const T* input, T low, T high, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = noa::math::clamp(input[idx], low, high);
        }
    }

    namespace padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024;
            constexpr uint WARPS = BLOCK_SIZE.y;
            return noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, uint input_pitch, T* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC>(input[row * input_pitch + idx]);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, uint input_pitch, T value,
                                        T* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC>(input[row * input_pitch + idx], value);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(const T* input, uint input_pitch,
                                        const T* array, uint array_pitch,
                                        T* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = getValue_<GENERIC>(input[row * input_pitch + idx],
                                                                          array[row * array_pitch + idx]);
        }

        template<typename T>
        __global__ void clamp_(const T* input, uint input_pitch, T low, T high,
                               T* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = noa::math::clamp(input[row * input_pitch + idx], low, high);
        }
    }
}

namespace noa::cuda::math::details {
    template<int GEN, typename T>
    void generic(const T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T>
    void genericWithValue(const T* input, T value, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, value, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T>
    void genericWithArray(const T* input, const T* array, T* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::computeGeneric_<GEN><<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, array, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T>
    void generic(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T>
    void genericWithValue(const T* input, size_t input_pitch, T value, T* output, size_t output_pitch,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, value, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<int GEN, typename T>
    void genericWithArray(const T* input, size_t input_pitch,
                          const T* array, size_t array_pitch,
                          T* output, size_t output_pitch,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::computeGeneric_<GEN><<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, array, array_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace noa::cuda::math {
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
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::clamp_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, low, high, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// INSTANTIATE:
namespace noa::cuda::math {
    #define INSTANTIATE_ONE_MINUS_ABS(T)                                                                        \
    template void details::generic<details::GEN_ONE_MINUS, T>(const T*, T*, size_t, Stream&);                   \
    template void details::generic<details::GEN_ONE_MINUS, T>(const T*, size_t, T*, size_t, size3_t, Stream&);  \
    template void details::generic<details::GEN_ABS, T>(const T*, T*, size_t, Stream&);                         \
    template void details::generic<details::GEN_ABS, T>(const T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_ONE_MINUS_ABS(float);
    INSTANTIATE_ONE_MINUS_ABS(double);
    INSTANTIATE_ONE_MINUS_ABS(cfloat_t);
    INSTANTIATE_ONE_MINUS_ABS(cdouble_t);
    INSTANTIATE_ONE_MINUS_ABS(short);
    INSTANTIATE_ONE_MINUS_ABS(int);
    INSTANTIATE_ONE_MINUS_ABS(long);
    INSTANTIATE_ONE_MINUS_ABS(long long);

    #define INSTANTIATE_SQUARE(T)                                                                           \
    template void details::generic<details::GEN_SQUARE, T>(const T*, T*, size_t, Stream&);                  \
    template void details::generic<details::GEN_SQUARE, T>(const T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_SQUARE(float);
    INSTANTIATE_SQUARE(double);
    INSTANTIATE_SQUARE(cfloat_t);
    INSTANTIATE_SQUARE(cdouble_t);
    INSTANTIATE_SQUARE(short);
    INSTANTIATE_SQUARE(int);
    INSTANTIATE_SQUARE(long);
    INSTANTIATE_SQUARE(long long);
    INSTANTIATE_SQUARE(unsigned short);
    INSTANTIATE_SQUARE(unsigned int);
    INSTANTIATE_SQUARE(unsigned long);
    INSTANTIATE_SQUARE(unsigned long long);

    #define INSTANTIATE_FP(T)                                                                                       \
    template void details::generic<details::GEN_INVERSE, T>(const T*, T*, size_t, Stream&);                         \
    template void details::generic<details::GEN_INVERSE, T>(const T*, size_t, T*, size_t, size3_t, Stream&);        \
    template void details::generic<details::GEN_SQRT, T>(const T*, T*, size_t, Stream&);                            \
    template void details::generic<details::GEN_SQRT, T>(const T*, size_t, T*, size_t, size3_t, Stream&);           \
    template void details::generic<details::GEN_RSQRT, T>(const T*, T*, size_t, Stream&);                           \
    template void details::generic<details::GEN_RSQRT, T>(const T*, size_t, T*, size_t, size3_t, Stream&);          \
    template void details::generic<details::GEN_EXP, T>(const T*, T*, size_t, Stream&);                             \
    template void details::generic<details::GEN_EXP, T>(const T*, size_t, T*, size_t, size3_t, Stream&);            \
    template void details::generic<details::GEN_LOG, T>(const T*, T*, size_t, Stream&);                             \
    template void details::generic<details::GEN_LOG, T>(const T*, size_t, T*, size_t, size3_t, Stream&);            \
    template void details::generic<details::GEN_COS, T>(const T*, T*, size_t, Stream&);                             \
    template void details::generic<details::GEN_COS, T>(const T*, size_t, T*, size_t, size3_t, Stream&);            \
    template void details::generic<details::GEN_SIN, T>(const T*, T*, size_t, Stream&);                             \
    template void details::generic<details::GEN_SIN, T>(const T*, size_t, T*, size_t, size3_t, Stream&);            \
    template void details::genericWithValue<details::GEN_POW, T>(const T*, T, T*, size_t, Stream&);                 \
    template void details::genericWithValue<details::GEN_POW, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&)

    INSTANTIATE_FP(float);
    INSTANTIATE_FP(double);

    template void details::generic<details::GEN_NORMALIZE, cfloat_t>(const cfloat_t*, cfloat_t*, size_t, Stream&);
    template void details::generic<details::GEN_NORMALIZE, cfloat_t>(const cfloat_t*, size_t, cfloat_t*, size_t,
                                                                     size3_t, Stream&);
    template void details::generic<details::GEN_NORMALIZE, cdouble_t>(const cdouble_t*, cdouble_t*, size_t, Stream&);
    template void details::generic<details::GEN_NORMALIZE, cdouble_t>(const cdouble_t*, size_t, cdouble_t*, size_t,
                                                                      size3_t, Stream&);

    #define INSTANTIATE_MIN_MAX(T)                                                                                                  \
    template void details::genericWithValue<details::GEN_MIN, T>(const T*, T, T*, size_t, Stream&);                                 \
    template void details::genericWithValue<details::GEN_MIN, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&);                \
    template void details::genericWithArray<details::GEN_MIN, T>(const T*, const T*, T*, size_t, Stream&);                          \
    template void details::genericWithArray<details::GEN_MIN, T>(const T*, size_t, const T*, size_t, T*, size_t, size3_t, Stream&); \
    template void details::genericWithValue<details::GEN_MAX, T>(const T*, T, T*, size_t, Stream&);                                 \
    template void details::genericWithValue<details::GEN_MAX, T>(const T*, size_t, T, T*, size_t, size3_t, Stream&);                \
    template void details::genericWithArray<details::GEN_MAX, T>(const T*, const T*, T*, size_t, Stream&);                          \
    template void details::genericWithArray<details::GEN_MAX, T>(const T*, size_t, const T*, size_t, T*, size_t, size3_t, Stream&); \
    template void clamp<T>(const T*, T, T, T*, size_t, Stream&);                                                                    \
    template void clamp<T>(const T*, size_t, T, T, T*, size_t, size3_t, Stream&)

    INSTANTIATE_MIN_MAX(float);
    INSTANTIATE_MIN_MAX(double);
    INSTANTIATE_MIN_MAX(char);
    INSTANTIATE_MIN_MAX(short);
    INSTANTIATE_MIN_MAX(int);
    INSTANTIATE_MIN_MAX(long);
    INSTANTIATE_MIN_MAX(long long);
    INSTANTIATE_MIN_MAX(unsigned char);
    INSTANTIATE_MIN_MAX(unsigned short);
    INSTANTIATE_MIN_MAX(unsigned int);
    INSTANTIATE_MIN_MAX(unsigned long);
    INSTANTIATE_MIN_MAX(unsigned long long);
}
