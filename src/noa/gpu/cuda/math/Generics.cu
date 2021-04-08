#include "noa/gpu/cuda/math/Generics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace Noa::CUDA::Math::Details {
    enum : int {
        GEN_ONE_MINUS, GEN_INVERSE, GEN_SQUARE, GEN_SQRT, GEN_RSQRT, GEN_EXP, GEN_LOG,
        GEN_ABS, GEN_COS, GEN_SIN, GEN_NORMALIZE, GEN_POW, GEN_MIN, GEN_MAX
    };

    template<int OPERATION, typename T>
    NOA_FD T compute(T value) {
        T out;
        if constexpr (OPERATION == GEN_ONE_MINUS) {
            out = T(1) - value;
        } else if constexpr (OPERATION == GEN_INVERSE) {
            out = T(1) / value;
        } else if constexpr (OPERATION == GEN_SQUARE) {
            out = value * value;
        } else if constexpr (OPERATION == GEN_SQRT) {
            out = Noa::Math::sqrt(value);
        } else if constexpr (OPERATION == GEN_RSQRT) {
            out = Noa::Math::rsqrt(value);
        } else if constexpr (OPERATION == GEN_EXP) {
            out = Noa::Math::exp(value);
        } else if constexpr (OPERATION == GEN_LOG) {
            out = Noa::Math::log(value);
        } else if constexpr (OPERATION == GEN_ABS) {
            out = Noa::Math::abs(value);
        } else if constexpr (OPERATION == GEN_COS) {
            out = Noa::Math::cos(value);
        } else if constexpr (OPERATION == GEN_SIN) {
            out = Noa::Math::sin(value);
        } else if constexpr (OPERATION == GEN_NORMALIZE) {
            out = Noa::Math::normalize(value);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out;
    }

    template<int OPERATION, typename T>
    NOA_FD T compute(T lhs, T rhs) {
        T out;
        if constexpr (OPERATION == GEN_POW) {
            out = Noa::Math::pow(lhs, rhs);
        } else if constexpr (OPERATION == GEN_MIN) {
            out = Noa::Math::min(lhs, rhs);
        } else if constexpr (OPERATION == GEN_MAX) {
            out = Noa::Math::max(lhs, rhs);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
        return out;
    }
}

namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 256;

    uint getBlocks(uint elements) {
        constexpr uint MAX_GRIDS = 16384;
        uint total_blocks = Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
        return total_blocks;
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx]);
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, T value, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx], value);
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, T* array, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx], array[idx]);
    }

    template<typename T>
    __global__ void clamp(T* input, T low, T high, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = Noa::Math::clamp(input[idx], low, high);
    }
}

namespace Noa::CUDA::Math::Details::Padded {
    static constexpr dim3 BLOCK_SIZE(32, 8);

    uint getBlocks(uint2_t shape_2d) {
        constexpr uint MAX_BLOCKS = 1024;
        constexpr uint WARPS = BLOCK_SIZE.y;
        return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, uint pitch_input, T* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx]);
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, uint pitch_input, T value, T* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx], value);
    }

    template<int GENERIC, typename T>
    __global__ void computeGeneric(T* input, uint pitch_input,
                                   T* array, uint pitch_array,
                                   T* output, uint pitch_output,
                                   uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx],
                                                                    array[row * pitch_array + idx]);
    }

    template<typename T>
    __global__ void clamp(T* input, uint pitch_input, T low, T high, T* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = Noa::Math::clamp(input[row * pitch_input + idx], low, high);
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T>
    void oneMinus(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_ONE_MINUS>,
                        input, output, elements);
    }

    template<typename T>
    void oneMinus(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_ONE_MINUS>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void inverse(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_INVERSE>,
                        input, output, elements);
    }

    template<typename T>
    void inverse(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_INVERSE>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void square(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_SQUARE>,
                        input, output, elements);
    }

    template<typename T>
    void square(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_SQUARE>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void sqrt(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_SQRT>,
                        input, output, elements);
    }

    template<typename T>
    void sqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_SQRT>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void rsqrt(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_RSQRT>,
                        input, output, elements);
    }

    template<typename T>
    void rsqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_RSQRT>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void pow(T* input, T exponent, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_POW>,
                        input, exponent, output, elements);
    }

    template<typename T>
    void pow(T* input, size_t pitch_input, T exponent, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_POW>,
                        input, pitch_input, exponent, output, pitch_output, shape_2d);
    }

    template<typename T>
    void exp(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_EXP>,
                        input, output, elements);
    }

    template<typename T>
    void exp(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_EXP>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void log(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_LOG>,
                        input, output, elements);
    }

    template<typename T>
    void log(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_LOG>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void abs(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_ABS>,
                        input, output, elements);
    }

    template<typename T>
    void abs(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_ABS>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void cos(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_COS>,
                        input, output, elements);
    }

    template<typename T>
    void cos(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_COS>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void sin(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_SIN>,
                        input, output, elements);
    }

    template<typename T>
    void sin(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_SIN>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void normalize(T* input, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_NORMALIZE>,
                        input, output, elements);
    }

    template<typename T>
    void normalize(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_NORMALIZE>,
                        input, pitch_input, output, pitch_output, shape_2d);
    }

    template<typename T>
    void min(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_MIN>,
                        input, threshold, output, elements);
    }

    template<typename T>
    void min(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_MIN>,
                        input, pitch_input, threshold, output, pitch_output, shape_2d);
    }

    template<typename T>
    void min(T* input, T* array, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_MIN>,
                        input, array, output, elements);
    }

    template<typename T>
    void min(T* input, size_t pitch_input, T* array, size_t pitch_array, T* output, size_t pitch_output,
             size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_MIN>,
                        input, pitch_input, array, pitch_array, output, pitch_output, shape_2d);
    }

    template<typename T>
    void max(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_MAX>,
                        input, threshold, output, elements);
    }

    template<typename T>
    void max(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_MAX>,
                        input, pitch_input, threshold, output, pitch_output, shape_2d);
    }

    template<typename T>
    void max(T* input, T* array, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::computeGeneric<Details::GEN_MAX>,
                        input, array, output, elements);
    }

    template<typename T>
    void max(T* input, size_t pitch_input, T* array, size_t pitch_array, T* output, size_t pitch_output,
             size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::computeGeneric<Details::GEN_MAX>,
                        input, pitch_input, array, pitch_array, output, pitch_output, shape_2d);
    }

    template<typename T>
    void clamp(T* input, T low, T high, T* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::clamp,
                        input, low, high, output, elements);
    }

    template<typename T>
    void clamp(T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output, size3_t shape,
               Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::clamp,
                        input, pitch_input, low, high, output, pitch_output, shape_2d);
    }
}

// INSTANTIATE:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_ONE_MINUS_ABS(T)                                        \
    template void oneMinus<T>(T*, T*, size_t, Stream&);                         \
    template void oneMinus<T>(T*, size_t, T*, size_t, size3_t, Stream&);        \
    template void abs<T>(T*, T*, size_t, Stream&);                              \
    template void abs<T>(T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_ONE_MINUS_ABS(float);
    INSTANTIATE_ONE_MINUS_ABS(double);
    INSTANTIATE_ONE_MINUS_ABS(cfloat_t);
    INSTANTIATE_ONE_MINUS_ABS(cdouble_t);
    INSTANTIATE_ONE_MINUS_ABS(int16_t);
    INSTANTIATE_ONE_MINUS_ABS(int32_t);

    #define INSTANTIATE_SQUARE(T)                                       \
    template void square<T>(T*, T*, size_t, Stream&);                   \
    template void square<T>(T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_SQUARE(float);
    INSTANTIATE_SQUARE(double);
    INSTANTIATE_SQUARE(cfloat_t);
    INSTANTIATE_SQUARE(cdouble_t);
    INSTANTIATE_SQUARE(int16_t);
    INSTANTIATE_SQUARE(uint16_t);
    INSTANTIATE_SQUARE(int32_t);
    INSTANTIATE_SQUARE(uint32_t);

    #define INSTANTIATE_FP(T)                                           \
    template void inverse<T>(T*, T*, size_t, Stream&);                  \
    template void inverse<T>(T*, size_t, T*, size_t, size3_t, Stream&); \
    template void sqrt<T>(T*, T*, size_t, Stream&);                     \
    template void sqrt<T>(T*, size_t, T*, size_t, size3_t, Stream&);    \
    template void rsqrt<T>(T*, T*, size_t, Stream&);                    \
    template void rsqrt<T>(T*, size_t, T*, size_t, size3_t, Stream&);   \
    template void pow<T>(T*, T, T*, size_t, Stream&);                   \
    template void pow<T>(T*, size_t, T, T*, size_t, size3_t, Stream&);  \
    template void exp<T>(T*, T*, size_t, Stream&);                      \
    template void exp<T>(T*, size_t, T*, size_t, size3_t, Stream&);     \
    template void log<T>(T*, T*, size_t, Stream&);                      \
    template void log<T>(T*, size_t, T*, size_t, size3_t, Stream&);     \
    template void cos<T>(T*, T*, size_t, Stream&);                      \
    template void cos<T>(T*, size_t, T*, size_t, size3_t, Stream&);     \
    template void sin<T>(T*, T*, size_t, Stream&);                      \
    template void sin<T>(T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_FP(float);
    INSTANTIATE_FP(double);

    template void normalize<cfloat_t>(cfloat_t*, cfloat_t*, size_t, Stream&);
    template void normalize<cfloat_t>(cfloat_t*, size_t, cfloat_t*, size_t, size3_t, Stream&);
    template void normalize<cdouble_t>(cdouble_t*, cdouble_t*, size_t, Stream&);
    template void normalize<cdouble_t>(cdouble_t*, size_t, cdouble_t*, size_t, size3_t, Stream&);

    #define INSTANTIATE_MIN_MAX(T)                                              \
    template void min<T>(T*, T, T*, size_t, Stream&);                           \
    template void min<T>(T*, size_t, T, T*, size_t, size3_t, Stream&);          \
    template void min<T>(T*, T*, T*, size_t, Stream&);                          \
    template void min<T>(T*, size_t, T*, size_t, T*, size_t, size3_t, Stream&); \
    template void max<T>(T*, T, T*, size_t, Stream&);                           \
    template void max<T>(T*, size_t, T, T*, size_t, size3_t, Stream&);          \
    template void max<T>(T*, T*, T*, size_t, Stream&);                          \
    template void max<T>(T*, size_t, T*, size_t, T*, size_t, size3_t, Stream&); \
    template void clamp<T>(T*, T, T, T*, size_t, Stream&);                      \
    template void clamp<T>(T*, size_t, T, T, T*, size_t, size3_t, Stream&)

    INSTANTIATE_MIN_MAX(float);
    INSTANTIATE_MIN_MAX(double);
    INSTANTIATE_MIN_MAX(int16_t);
    INSTANTIATE_MIN_MAX(uint16_t);
    INSTANTIATE_MIN_MAX(int32_t);
    INSTANTIATE_MIN_MAX(uint32_t);
    INSTANTIATE_MIN_MAX(char);
    INSTANTIATE_MIN_MAX(unsigned char);
}
