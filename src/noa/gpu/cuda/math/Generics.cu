#include "noa/gpu/cuda/math/Generics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace {
    using namespace Noa;

    template<int GEN, typename T>
    NOA_FD T getValue_(T value) {
        T out;
        if constexpr (GEN == CUDA::Math::Details::GEN_ONE_MINUS) {
            out = T(1) - value;
        } else if constexpr (GEN == CUDA::Math::Details::GEN_INVERSE) {
            out = T(1) / value;
        } else if constexpr (GEN == CUDA::Math::Details::GEN_SQUARE) {
            out = value * value;
        } else if constexpr (GEN == CUDA::Math::Details::GEN_SQRT) {
            out = Noa::Math::sqrt(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_RSQRT) {
            out = Noa::Math::rsqrt(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_EXP) {
            out = Noa::Math::exp(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_LOG) {
            out = Noa::Math::log(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_ABS) {
            out = Noa::Math::abs(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_COS) {
            out = Noa::Math::cos(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_SIN) {
            out = Noa::Math::sin(value);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_NORMALIZE) {
            out = Noa::Math::normalize(value);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out;
    }

    template<int GEN, typename T>
    NOA_FD T getValue_(T lhs, T rhs) {
        T out;
        if constexpr (GEN == CUDA::Math::Details::GEN_POW) {
            out = Noa::Math::pow(lhs, rhs);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_MIN) {
            out = Noa::Math::min(lhs, rhs);
        } else if constexpr (GEN == CUDA::Math::Details::GEN_MAX) {
            out = Noa::Math::max(lhs, rhs);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out;
    }

    namespace Contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 16384;
            uint total_blocks = Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
            return total_blocks;
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx]);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, T value, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx], value);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, T* array, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = getValue_<GENERIC>(input[idx], array[idx]);
        }

        template<typename T>
        __global__ void clamp_(T* input, T low, T high, T* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = Noa::Math::clamp(input[idx], low, high);
        }
    }

    namespace Padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024;
            constexpr uint WARPS = BLOCK_SIZE.y;
            return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, uint pitch_input, T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = getValue_<GENERIC>(input[row * pitch_input + idx]);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, uint pitch_input, T value,
                                        T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = getValue_<GENERIC>(input[row * pitch_input + idx], value);
        }

        template<int GENERIC, typename T>
        __global__ void computeGeneric_(T* input, uint pitch_input, T* array, uint pitch_array,
                                        T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = getValue_<GENERIC>(input[row * pitch_input + idx],
                                                                          array[row * pitch_array + idx]);
        }

        template<typename T>
        __global__ void clamp_(T* input, uint pitch_input, T low, T high,
                               T* output, uint pitch_output, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * pitch_output + idx] = Noa::Math::clamp(input[row * pitch_input + idx], low, high);
        }
    }
}

namespace Noa::CUDA::Math::Details {
    template<int GEN, typename T>
    void generic(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(blocks, Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeGeneric_<GEN>,
                        input, output, elements);
    }

    template<int GEN, typename T>
    void genericWithValue(T* input, T value, T* output, size_t elements, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(blocks, Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeGeneric_<GEN>,
                        input, value, output, elements);
    }

    template<int GEN, typename T>
    void genericWithArray(T* input, T* array, T* output, size_t elements, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(blocks, Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::computeGeneric_<GEN>,
                        input, array, output, elements);
    }

    template<int GEN, typename T>
    void generic(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeGeneric_<GEN>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<int GEN, typename T>
    void genericWithValue(T* input, size_t pitch_input, T value, T* output, size_t pitch_output,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeGeneric_<GEN>,
                        input, pitch_input, value, output, pitch_output, shape_2d);
    }

    template<int GEN, typename T>
    void genericWithArray(T* input, size_t pitch_input, T* array, size_t pitch_array, T* output, size_t pitch_output,
                          size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::computeGeneric_<GEN>,
                        input, pitch_input, array, pitch_array, output, pitch_output, shape_2d);
    }
}

namespace Noa::CUDA::Math {
    template<typename T>
    void clamp(T* input, T low, T high, T* output, size_t elements, Stream& stream) {
        uint blocks = Contiguous_::getBlocks_(elements);
        NOA_CUDA_LAUNCH(blocks, Contiguous_::BLOCK_SIZE, 0, stream.get(),
                        Contiguous_::clamp_,
                        input, low, high, output, elements);
    }

    template<typename T>
    void clamp(T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output,
               size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Padded_::getBlocks_(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Padded_::BLOCK_SIZE, 0, stream.get(),
                        Padded_::clamp_,
                        input, pitch_input, low, high, output, pitch_output, shape_2d);
    }
}

// INSTANTIATE:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_ONE_MINUS_ABS(T)                                                                    \
    template void Details::generic<Details::GEN_ONE_MINUS, T>(T*, T*, size_t, Stream&);                     \
    template void Details::generic<Details::GEN_ONE_MINUS, T>(T*, size_t, T*, size_t, size3_t, Stream&);    \
    template void Details::generic<Details::GEN_ABS, T>(T*, T*, size_t, Stream&);                           \
    template void Details::generic<Details::GEN_ABS, T>(T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_ONE_MINUS_ABS(float);
    INSTANTIATE_ONE_MINUS_ABS(double);
    INSTANTIATE_ONE_MINUS_ABS(cfloat_t);
    INSTANTIATE_ONE_MINUS_ABS(cdouble_t);
    INSTANTIATE_ONE_MINUS_ABS(short);
    INSTANTIATE_ONE_MINUS_ABS(int);
    INSTANTIATE_ONE_MINUS_ABS(long);
    INSTANTIATE_ONE_MINUS_ABS(long long);

    #define INSTANTIATE_SQUARE(T)                                                                       \
    template void Details::generic<Details::GEN_SQUARE, T>(T*, T*, size_t, Stream&);                    \
    template void Details::generic<Details::GEN_SQUARE, T>(T*, size_t, T*, size_t, size3_t, Stream&)

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

    #define INSTANTIATE_FP(T)                                                                                   \
    template void Details::generic<Details::GEN_INVERSE, T>(T*, T*, size_t, Stream&);                           \
    template void Details::generic<Details::GEN_INVERSE, T>(T*, size_t, T*, size_t, size3_t, Stream&);          \
    template void Details::generic<Details::GEN_SQRT, T>(T*, T*, size_t, Stream&);                              \
    template void Details::generic<Details::GEN_SQRT, T>(T*, size_t, T*, size_t, size3_t, Stream&);             \
    template void Details::generic<Details::GEN_RSQRT, T>(T*, T*, size_t, Stream&);                             \
    template void Details::generic<Details::GEN_RSQRT, T>(T*, size_t, T*, size_t, size3_t, Stream&);            \
    template void Details::generic<Details::GEN_EXP, T>(T*, T*, size_t, Stream&);                               \
    template void Details::generic<Details::GEN_EXP, T>(T*, size_t, T*, size_t, size3_t, Stream&);              \
    template void Details::generic<Details::GEN_LOG, T>(T*, T*, size_t, Stream&);                               \
    template void Details::generic<Details::GEN_LOG, T>(T*, size_t, T*, size_t, size3_t, Stream&);              \
    template void Details::generic<Details::GEN_COS, T>(T*, T*, size_t, Stream&);                               \
    template void Details::generic<Details::GEN_COS, T>(T*, size_t, T*, size_t, size3_t, Stream&);              \
    template void Details::generic<Details::GEN_SIN, T>(T*, T*, size_t, Stream&);                               \
    template void Details::generic<Details::GEN_SIN, T>(T*, size_t, T*, size_t, size3_t, Stream&);              \
    template void Details::genericWithValue<Details::GEN_POW, T>(T*, T, T*, size_t, Stream&);                   \
    template void Details::genericWithValue<Details::GEN_POW, T>(T*, size_t, T, T*, size_t, size3_t, Stream&)

    INSTANTIATE_FP(float);
    INSTANTIATE_FP(double);

    template void Details::generic<Details::GEN_NORMALIZE, cfloat_t>(cfloat_t*, cfloat_t*, size_t, Stream&);
    template void Details::generic<Details::GEN_NORMALIZE, cfloat_t>(cfloat_t*, size_t, cfloat_t*, size_t,
                                                                     size3_t, Stream&);
    template void Details::generic<Details::GEN_NORMALIZE, cdouble_t>(cdouble_t*, cdouble_t*, size_t, Stream&);
    template void Details::generic<Details::GEN_NORMALIZE, cdouble_t>(cdouble_t*, size_t, cdouble_t*, size_t,
                                                                      size3_t, Stream&);

    #define INSTANTIATE_MIN_MAX(T)                                                                                      \
    template void Details::genericWithValue<Details::GEN_MIN, T>(T*, T, T*, size_t, Stream&);                           \
    template void Details::genericWithValue<Details::GEN_MIN, T>(T*, size_t, T, T*, size_t, size3_t, Stream&);          \
    template void Details::genericWithArray<Details::GEN_MIN, T>(T*, T*, T*, size_t, Stream&);                          \
    template void Details::genericWithArray<Details::GEN_MIN, T>(T*, size_t, T*, size_t, T*, size_t, size3_t, Stream&); \
    template void Details::genericWithValue<Details::GEN_MAX, T>(T*, T, T*, size_t, Stream&);                           \
    template void Details::genericWithValue<Details::GEN_MAX, T>(T*, size_t, T, T*, size_t, size3_t, Stream&);          \
    template void Details::genericWithArray<Details::GEN_MAX, T>(T*, T*, T*, size_t, Stream&);                          \
    template void Details::genericWithArray<Details::GEN_MAX, T>(T*, size_t, T*, size_t, T*, size_t, size3_t, Stream&); \
    template void clamp<T>(T*, T, T, T*, size_t, Stream&);                                                              \
    template void clamp<T>(T*, size_t, T, T, T*, size_t, size3_t, Stream&)

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
