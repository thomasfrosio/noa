#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Booleans.h"

namespace {
    using namespace noa;

    namespace contiguous_ {
        constexpr uint THREADS = 512;

        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 16384;
            return noa::math::min(noa::math::divideUp(elements, THREADS), MAX_GRIDS);
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void isLess_(const T* input, T threshold, U* output, uint elements) {
            #pragma unroll 10
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x)
                output[idx] = static_cast<U>(input[idx] < threshold);
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void isGreater_(const T* input, T threshold, U* output, uint elements) {
            #pragma unroll 10
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x)
                output[idx] = static_cast<U>(threshold < input[idx]);
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void isWithin_(const T* input, T low, T high, U* output, uint elements) {
            #pragma unroll 10
            for (uint idx = blockIdx.x * THREADS + threadIdx.x; idx < elements; idx += THREADS * gridDim.x) {
                T tmp = input[idx];
                output[idx] = static_cast<U>(low < tmp && tmp < high);
            }
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS)
        void logicNOT_(const T* input, U* output, uint elements) {
            #pragma unroll 10
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = static_cast<U>(!input[idx]);
        }
    }

    namespace padded_ {
        constexpr dim3 THREADS(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            // every warp processes at least one row.
            return noa::math::min(noa::math::divideUp(shape_2d.y, THREADS.y), MAX_BLOCKS);
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void isLess_(const T* input, uint input_pitch, T threshold,
                     U* output, uint output_pitch, uint2_t shape) {
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    output[row * output_pitch + idx] = static_cast<U>(input[row * input_pitch + idx] < threshold);
            }
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void isGreater_(const T* input, uint input_pitch, T threshold,
                        U* output, uint output_pitch, uint2_t shape) {
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    output[row * output_pitch + idx] = static_cast<U>(threshold < input[row * input_pitch + idx]);
            }
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void isWithin_(const T* input, uint input_pitch, T low, T high,
                       U* output, uint output_pitch, uint2_t shape) {
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x) {
                    T tmp = input[row * input_pitch + idx];
                    output[row * output_pitch + idx] = static_cast<U>(low < tmp && tmp < high);
                }
            }
        }

        template<typename T, typename U>
        __global__ __launch_bounds__(THREADS.x * THREADS.y)
        void logicNOT_(const T* input, uint input_pitch, U* output, uint output_pitch, uint2_t shape) {
            for (uint row = THREADS.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * THREADS.y) {
                #pragma unroll 16
                for (uint idx = threadIdx.x; idx < shape.x; idx += THREADS.x)
                    output[row * output_pitch + idx] = static_cast<U>(!input[row * input_pitch + idx]);
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T, typename U>
    void isLess(const T* input, T threshold, U* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isLess_<<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                input, threshold, output, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void isLess(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isLess_<<<blocks, padded_::THREADS, 0, stream.get()>>>(
                input, input_pitch, threshold, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void isGreater(const T* input, T threshold, U* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isGreater_<<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                input, threshold, output, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void isGreater(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                   size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isGreater_<<<blocks, padded_::THREADS, 0, stream.get()>>>(
                input, input_pitch, threshold, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void isWithin(const T* input, T low, T high, U* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isWithin_<<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                input, low, high, output, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void isWithin(const T* input, size_t input_pitch, T low, T high, U* output, size_t output_pitch,
                  size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isWithin_<<<blocks, padded_::THREADS, 0, stream.get()>>>(
                input, input_pitch, low, high, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void logicNOT(const T* input, U* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::logicNOT_<<<blocks, contiguous_::THREADS, 0, stream.get()>>>(
                input, output, elements);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T, typename U>
    void logicNOT(const T* input, size_t input_pitch, U* output, size_t output_pitch,
                  size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint2_t shape_2d(shape.x, rows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::logicNOT_<<<blocks, padded_::THREADS, 0, stream.get()>>>(
                input, input_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_BOOLEANS_(T, U)                                             \
    template void isLess<T, U>(const T*, T, U*, size_t, Stream&);                       \
    template void isLess<T, U>(const T*, size_t, T, U*, size_t, size3_t, Stream&);      \
    template void isGreater<T, U>(const T*, T, U*, size_t, Stream&);                    \
    template void isGreater<T, U>(const T*, size_t, T, U*, size_t, size3_t, Stream&);   \
    template void isWithin<T, U>(const T*, T, T, U*, size_t, Stream&);                  \
    template void isWithin<T, U>(const T*, size_t, T, T, U*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_BOOLEANS_(float, float);
    NOA_INSTANTIATE_BOOLEANS_(double, double);
    NOA_INSTANTIATE_BOOLEANS_(char, char);
    NOA_INSTANTIATE_BOOLEANS_(short, short);
    NOA_INSTANTIATE_BOOLEANS_(int, int);
    NOA_INSTANTIATE_BOOLEANS_(long, long);
    NOA_INSTANTIATE_BOOLEANS_(long long, long long);
    NOA_INSTANTIATE_BOOLEANS_(unsigned char, unsigned char);
    NOA_INSTANTIATE_BOOLEANS_(unsigned short, unsigned short);
    NOA_INSTANTIATE_BOOLEANS_(unsigned int, unsigned int);
    NOA_INSTANTIATE_BOOLEANS_(unsigned long, unsigned long);
    NOA_INSTANTIATE_BOOLEANS_(unsigned long long, unsigned long long);

    NOA_INSTANTIATE_BOOLEANS_(float, bool);
    NOA_INSTANTIATE_BOOLEANS_(double, bool);
    NOA_INSTANTIATE_BOOLEANS_(char, bool);
    NOA_INSTANTIATE_BOOLEANS_(short, bool);
    NOA_INSTANTIATE_BOOLEANS_(int, bool);
    NOA_INSTANTIATE_BOOLEANS_(long, bool);
    NOA_INSTANTIATE_BOOLEANS_(long long, bool);
    NOA_INSTANTIATE_BOOLEANS_(unsigned char, bool);
    NOA_INSTANTIATE_BOOLEANS_(unsigned short, bool);
    NOA_INSTANTIATE_BOOLEANS_(unsigned int, bool);
    NOA_INSTANTIATE_BOOLEANS_(unsigned long, bool);
    NOA_INSTANTIATE_BOOLEANS_(unsigned long long, bool);

    NOA_INSTANTIATE_BOOLEANS_(float, char);
    NOA_INSTANTIATE_BOOLEANS_(float, short);
    NOA_INSTANTIATE_BOOLEANS_(float, int);
    NOA_INSTANTIATE_BOOLEANS_(float, long);
    NOA_INSTANTIATE_BOOLEANS_(float, long long);
    NOA_INSTANTIATE_BOOLEANS_(float, unsigned char);
    NOA_INSTANTIATE_BOOLEANS_(float, unsigned short);
    NOA_INSTANTIATE_BOOLEANS_(float, unsigned int);
    NOA_INSTANTIATE_BOOLEANS_(float, unsigned long);
    NOA_INSTANTIATE_BOOLEANS_(float, unsigned long long);

    NOA_INSTANTIATE_BOOLEANS_(double, char);
    NOA_INSTANTIATE_BOOLEANS_(double, short);
    NOA_INSTANTIATE_BOOLEANS_(double, int);
    NOA_INSTANTIATE_BOOLEANS_(double, long);
    NOA_INSTANTIATE_BOOLEANS_(double, long long);
    NOA_INSTANTIATE_BOOLEANS_(double, unsigned char);
    NOA_INSTANTIATE_BOOLEANS_(double, unsigned short);
    NOA_INSTANTIATE_BOOLEANS_(double, unsigned int);
    NOA_INSTANTIATE_BOOLEANS_(double, unsigned long);
    NOA_INSTANTIATE_BOOLEANS_(double, unsigned long long);

    #define NOA_INSTANTIATE_LOGIC_NOT_(T, U)                        \
    template void logicNOT<T, U>(const T*, U*, size_t, Stream&);    \
    template void logicNOT<T, U>(const T*, size_t, U*, size_t, size3_t, Stream&)

    NOA_INSTANTIATE_LOGIC_NOT_(char, char);
    NOA_INSTANTIATE_LOGIC_NOT_(short, short);
    NOA_INSTANTIATE_LOGIC_NOT_(int, int);
    NOA_INSTANTIATE_LOGIC_NOT_(long, long);
    NOA_INSTANTIATE_LOGIC_NOT_(long long, long long);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned char, unsigned char);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned short, unsigned short);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned int, unsigned int);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned long, unsigned long);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned long long, unsigned long long);
    NOA_INSTANTIATE_LOGIC_NOT_(bool, bool);

    NOA_INSTANTIATE_LOGIC_NOT_(char, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(short, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(int, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(long, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(long long, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned char, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned short, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned int, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned long, bool);
    NOA_INSTANTIATE_LOGIC_NOT_(unsigned long long, bool);
}
