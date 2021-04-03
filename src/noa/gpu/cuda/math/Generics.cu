#include "noa/gpu/cuda/math/Generics.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

#define NOA_GENERICS_ONE_MINUS 1
#define NOA_GENERICS_INVERSE 2
#define NOA_GENERICS_SQUARE 3
#define NOA_GENERICS_SQRT 4
#define NOA_GENERICS_RSQRT 5
#define NOA_GENERICS_EXP 6
#define NOA_GENERICS_LOG 7
#define NOA_GENERICS_ABS 8
#define NOA_GENERICS_COS 9
#define NOA_GENERICS_SIN 10
#define NOA_GENERICS_NORMALIZE 11
#define NOA_GENERICS_POW 12
#define NOA_GENERICS_MIN 13
#define NOA_GENERICS_MAX 14

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

template<int OPERATION, typename T>
static NOA_FD T compute(T value) {
    T out;
    if constexpr (OPERATION == NOA_GENERICS_ONE_MINUS) {
        out = T(1) - value;
    } else if constexpr (OPERATION == NOA_GENERICS_INVERSE) {
        out = T(1) / value;
    } else if constexpr (OPERATION == NOA_GENERICS_SQUARE) {
        out = value * value;
    } else if constexpr (OPERATION == NOA_GENERICS_SQRT) {
        out = Math::sqrt(value);
    } else if constexpr (OPERATION == NOA_GENERICS_RSQRT) {
        out = Math::rsqrt(value);
    } else if constexpr (OPERATION == NOA_GENERICS_EXP) {
        out = Math::exp(value);
    } else if constexpr (OPERATION == NOA_GENERICS_LOG) {
        out = Math::log(value);
    } else if constexpr (OPERATION == NOA_GENERICS_ABS) {
        out = Math::abs(value);
    } else if constexpr (OPERATION == NOA_GENERICS_COS) {
        out = Math::cos(value);
    } else if constexpr (OPERATION == NOA_GENERICS_SIN) {
        out = Math::sin(value);
    } else if constexpr (OPERATION == NOA_GENERICS_NORMALIZE) {
        out = Math::normalize(value);
    } else {
        static_assert(Noa::Traits::always_false_v<T>);
    }
    return out;
}

template<int OPERATION, typename T>
static NOA_FD T compute(T lhs, T rhs) {
    T out;
    if constexpr (OPERATION == NOA_GENERICS_POW) {
        out = Math::pow(lhs, rhs);
    } else if constexpr (OPERATION == NOA_GENERICS_MIN) {
        out = Math::min(lhs, rhs);
    } else if constexpr (OPERATION == NOA_GENERICS_MAX) {
        out = Math::max(lhs, rhs);
    } else {
        static_assert(Noa::Traits::always_false_v<T>);
    }
    return out;
}

// KERNELS:
namespace Noa::CUDA::Math::Kernels {
    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx]);
    }

    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, uint pitch_input, T* output, uint pitch_output,
                                          uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx]);
    }

    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, T value, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx], value);
    }

    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, uint pitch_input, T value, T* output, uint pitch_output,
                                          uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx], value);
    }

    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, T* array, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = compute<GENERIC>(input[idx], array[idx]);
    }

    template<int GENERIC, typename T>
    static __global__ void computeGeneric(T* input, uint pitch_input,
                                          T* array, uint pitch_array,
                                          T* output, uint pitch_output,
                                          uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = compute<GENERIC>(input[row * pitch_input + idx],
                                                                    array[row * pitch_array + idx]);
    }

    template<typename T>
    static __global__ void clamp(T* input, T low, T high, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = Noa::Math::clamp(input[idx], low, high);
    }

    template<typename T>
    static __global__ void clamp(T* input, uint pitch_input, T low, T high, T* output, uint pitch_output,
                                 uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = Noa::Math::clamp(input[row * pitch_input + idx], low, high);
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T>
    void oneMinus(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_ONE_MINUS>,
                        input, output, elements);
    }

    template<typename T>
    void oneMinus(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_ONE_MINUS>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void inverse(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_INVERSE>,
                        input, output, elements);
    }

    template<typename T>
    void inverse(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_INVERSE>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void square(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SQUARE>,
                        input, output, elements);
    }

    template<typename T>
    void square(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SQUARE>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void sqrt(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SQRT>,
                        input, output, elements);
    }

    template<typename T>
    void sqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SQRT>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void rsqrt(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_RSQRT>,
                        input, output, elements);
    }

    template<typename T>
    void rsqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_RSQRT>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void pow(T* input, T exponent, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_POW>,
                        input, exponent, output, elements);
    }

    template<typename T>
    void pow(T* input, size_t pitch_input, T exponent, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_POW>,
                        input, pitch_input, exponent, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void exp(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_EXP>,
                        input, output, elements);
    }

    template<typename T>
    void exp(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_EXP>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void log(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_LOG>,
                        input, output, elements);
    }

    template<typename T>
    void log(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_LOG>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void abs(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_ABS>,
                        input, output, elements);
    }

    template<typename T>
    void abs(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_ABS>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void cos(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_COS>,
                        input, output, elements);
    }

    template<typename T>
    void cos(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_COS>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void sin(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SIN>,
                        input, output, elements);
    }

    template<typename T>
    void sin(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_SIN>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void normalize(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_NORMALIZE>,
                        input, output, elements);
    }

    template<typename T>
    void normalize(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_NORMALIZE>,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void min(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MIN>,
                        input, threshold, output, elements);
    }

    template<typename T>
    void min(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MIN>,
                        input, pitch_input, threshold, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void min(T* input, T* array, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MIN>,
                        input, array, output, elements);
    }

    template<typename T>
    void min(T* input, size_t pitch_input, T* array, size_t pitch_array, T* output, size_t pitch_output,
             size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MIN>,
                        input, pitch_input, array, pitch_array, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void max(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MAX>,
                        input, threshold, output, elements);
    }

    template<typename T>
    void max(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MAX>,
                        input, pitch_input, threshold, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void max(T* input, T* array, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MAX>,
                        input, array, output, elements);
    }

    template<typename T>
    void max(T* input, size_t pitch_input, T* array, size_t pitch_array, T* output, size_t pitch_output,
             size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::computeGeneric<NOA_GENERICS_MAX>,
                        input, pitch_input, array, pitch_array, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void clamp(T* input, T low, T high, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::clamp,
                        input, low, high, output, elements);
    }

    template<typename T>
    void clamp(T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::clamp,
                        input, pitch_input, low, high, output, pitch_output, shape.x, getRows(shape));
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
