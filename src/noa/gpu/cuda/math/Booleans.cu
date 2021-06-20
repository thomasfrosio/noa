#include "noa/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/math/Booleans.h"

namespace {
    using namespace noa;

    namespace contiguous_ {
        constexpr uint BLOCK_SIZE = 256;

        uint getBlocks_(uint elements) {
            constexpr uint MAX_GRIDS = 16384;
            uint total_blocks = noa::math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
            return total_blocks;
        }

        template<typename T, typename U>
        __global__ void isLess_(const T* input, T threshold, U* output, uint elements) {
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                output[idx] = static_cast<U>(input[idx] < threshold);
        }

        template<typename T, typename U>
        __global__ void isGreater_(const T* input, T threshold, U* output, uint elements) {
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
                output[idx] = static_cast<U>(threshold < input[idx]);
        }

        template<typename T, typename U>
        __global__ void isWithin_(const T* input, T low, T high, U* output, uint elements) {
            for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
                T tmp = input[idx];
                output[idx] = static_cast<U>(low < tmp && tmp < high);
            }
        }

        template<typename T, typename U>
        __global__ void logicNOT_(const T* input, U* output, uint elements) {
            for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
                output[idx] = static_cast<U>(!input[idx]);
        }
    }

    namespace padded_ {
        constexpr dim3 BLOCK_SIZE(32, 8);

        uint getBlocks_(uint2_t shape_2d) {
            constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
            constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
            return noa::math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
        }

        template<typename T, typename U>
        __global__ void isLess_(const T* input, uint input_pitch, T threshold,
                                U* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = static_cast<U>(input[row * input_pitch + idx] < threshold);
        }

        template<typename T, typename U>
        __global__ void isGreater_(const T* input, uint input_pitch, T threshold,
                                   U* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = static_cast<U>(threshold < input[row * input_pitch + idx]);
        }

        template<typename T, typename U>
        __global__ void isWithin_(const T* input, uint input_pitch, T low, T high,
                                  U* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                    T tmp = input[row * input_pitch + idx];
                    output[row * output_pitch + idx] = static_cast<U>(low < tmp && tmp < high);
                }
            }
        }

        template<typename T, typename U>
        __global__ void logicNOT_(const T* input, uint input_pitch, U* output, uint output_pitch, uint2_t shape) {
            for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
                for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                    output[row * output_pitch + idx] = static_cast<U>(!input[row * input_pitch + idx]);
        }
    }
}

namespace noa::cuda::math {
    template<typename T, typename U>
    void isLess(const T* input, T threshold, U* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isLess_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, threshold, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void isLess(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isLess_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, threshold, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void isGreater(const T* input, T threshold, U* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isGreater_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, threshold, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void isGreater(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                   size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isGreater_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, threshold, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void isWithin(const T* input, T low, T high, U* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::isWithin_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, low, high, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void isWithin(const T* input, size_t input_pitch, T low, T high, U* output, size_t output_pitch,
                  size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::isWithin_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, low, high, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void logicNOT(const T* input, U* output, size_t elements, Stream& stream) {
        uint blocks = contiguous_::getBlocks_(elements);
        contiguous_::logicNOT_<<<blocks, contiguous_::BLOCK_SIZE, 0, stream.get()>>>(
                input, output, elements);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T, typename U>
    void logicNOT(const T* input, size_t input_pitch, U* output, size_t output_pitch,
                  size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = padded_::getBlocks_(shape_2d);
        padded_::logicNOT_<<<blocks, padded_::BLOCK_SIZE, 0, stream.get()>>>(
                input, input_pitch, output, output_pitch, shape_2d);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_BOOLEANS(T, U)                                                  \
    template void isLess<T, U>(const T*, T, U*, size_t, Stream&);                       \
    template void isLess<T, U>(const T*, size_t, T, U*, size_t, size3_t, Stream&);      \
    template void isGreater<T, U>(const T*, T, U*, size_t, Stream&);                    \
    template void isGreater<T, U>(const T*, size_t, T, U*, size_t, size3_t, Stream&);   \
    template void isWithin<T, U>(const T*, T, T, U*, size_t, Stream&);                  \
    template void isWithin<T, U>(const T*, size_t, T, T, U*, size_t, size3_t, Stream&)

    INSTANTIATE_BOOLEANS(float, float);
    INSTANTIATE_BOOLEANS(double, double);
    INSTANTIATE_BOOLEANS(char, char);
    INSTANTIATE_BOOLEANS(short, short);
    INSTANTIATE_BOOLEANS(int, int);
    INSTANTIATE_BOOLEANS(long, long);
    INSTANTIATE_BOOLEANS(long long, long long);
    INSTANTIATE_BOOLEANS(unsigned char, unsigned char);
    INSTANTIATE_BOOLEANS(unsigned short, unsigned short);
    INSTANTIATE_BOOLEANS(unsigned int, unsigned int);
    INSTANTIATE_BOOLEANS(unsigned long, unsigned long);
    INSTANTIATE_BOOLEANS(unsigned long long, unsigned long long);

    INSTANTIATE_BOOLEANS(float, bool);
    INSTANTIATE_BOOLEANS(double, bool);
    INSTANTIATE_BOOLEANS(char, bool);
    INSTANTIATE_BOOLEANS(short, bool);
    INSTANTIATE_BOOLEANS(int, bool);
    INSTANTIATE_BOOLEANS(long, bool);
    INSTANTIATE_BOOLEANS(long long, bool);
    INSTANTIATE_BOOLEANS(unsigned char, bool);
    INSTANTIATE_BOOLEANS(unsigned short, bool);
    INSTANTIATE_BOOLEANS(unsigned int, bool);
    INSTANTIATE_BOOLEANS(unsigned long, bool);
    INSTANTIATE_BOOLEANS(unsigned long long, bool);

    INSTANTIATE_BOOLEANS(float, char);
    INSTANTIATE_BOOLEANS(float, short);
    INSTANTIATE_BOOLEANS(float, int);
    INSTANTIATE_BOOLEANS(float, long);
    INSTANTIATE_BOOLEANS(float, long long);
    INSTANTIATE_BOOLEANS(float, unsigned char);
    INSTANTIATE_BOOLEANS(float, unsigned short);
    INSTANTIATE_BOOLEANS(float, unsigned int);
    INSTANTIATE_BOOLEANS(float, unsigned long);
    INSTANTIATE_BOOLEANS(float, unsigned long long);

    INSTANTIATE_BOOLEANS(double, char);
    INSTANTIATE_BOOLEANS(double, short);
    INSTANTIATE_BOOLEANS(double, int);
    INSTANTIATE_BOOLEANS(double, long);
    INSTANTIATE_BOOLEANS(double, long long);
    INSTANTIATE_BOOLEANS(double, unsigned char);
    INSTANTIATE_BOOLEANS(double, unsigned short);
    INSTANTIATE_BOOLEANS(double, unsigned int);
    INSTANTIATE_BOOLEANS(double, unsigned long);
    INSTANTIATE_BOOLEANS(double, unsigned long long);

    #define INSTANTIATE_LOGIC_NOT(T, U)                                    \
    template void logicNOT<T, U>(const T*, U*, size_t, Stream&);                 \
    template void logicNOT<T, U>(const T*, size_t, U*, size_t, size3_t, Stream&)

    INSTANTIATE_LOGIC_NOT(char, char);
    INSTANTIATE_LOGIC_NOT(short, short);
    INSTANTIATE_LOGIC_NOT(int, int);
    INSTANTIATE_LOGIC_NOT(long, long);
    INSTANTIATE_LOGIC_NOT(long long, long long);
    INSTANTIATE_LOGIC_NOT(unsigned char, unsigned char);
    INSTANTIATE_LOGIC_NOT(unsigned short, unsigned short);
    INSTANTIATE_LOGIC_NOT(unsigned int, unsigned int);
    INSTANTIATE_LOGIC_NOT(unsigned long, unsigned long);
    INSTANTIATE_LOGIC_NOT(unsigned long long, unsigned long long);
    INSTANTIATE_LOGIC_NOT(bool, bool);

    INSTANTIATE_LOGIC_NOT(char, bool);
    INSTANTIATE_LOGIC_NOT(short, bool);
    INSTANTIATE_LOGIC_NOT(int, bool);
    INSTANTIATE_LOGIC_NOT(long, bool);
    INSTANTIATE_LOGIC_NOT(long long, bool);
    INSTANTIATE_LOGIC_NOT(unsigned char, bool);
    INSTANTIATE_LOGIC_NOT(unsigned short, bool);
    INSTANTIATE_LOGIC_NOT(unsigned int, bool);
    INSTANTIATE_LOGIC_NOT(unsigned long, bool);
    INSTANTIATE_LOGIC_NOT(unsigned long long, bool);
}
