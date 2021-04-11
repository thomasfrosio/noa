#include "noa/gpu/cuda/math/Booleans.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

namespace Noa::CUDA::Math::Details::Contiguous {
    static constexpr uint BLOCK_SIZE = 256;

    static uint getBlocks(uint elements) {
        constexpr uint MAX_GRIDS = 16384;
        uint total_blocks = Noa::Math::min((elements + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRIDS);
        return total_blocks;
    }

    template<typename T, typename U>
    static __global__ void isLess(T* input, T threshold, U* output, uint elements) {
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
            output[idx] = static_cast<U>(input[idx] < threshold);
    }

    template<typename T, typename U>
    static __global__ void isGreater(T* input, T threshold, U* output, uint elements) {
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x)
            output[idx] = static_cast<U>(threshold < input[idx]);
    }

    template<typename T, typename U>
    static __global__ void isWithin(T* input, T low, T high, U* output, uint elements) {
        for (uint idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < elements; idx += BLOCK_SIZE * gridDim.x) {
            T tmp = input[idx];
            output[idx] = static_cast<U>(low < tmp && tmp < high);
        }
    }

    template<typename T, typename U>
    static __global__ void logicNOT(T* input, U* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = static_cast<U>(!input[idx]);
    }
}

namespace Noa::CUDA::Math::Details::Padded {
    static constexpr dim3 BLOCK_SIZE(32, 8);

    static uint getBlocks(uint2_t shape_2d) {
        constexpr uint MAX_BLOCKS = 1024; // the smaller, the more work per warp.
        constexpr uint WARPS = BLOCK_SIZE.y; // warps per block; every warp processes at least one row.
        return Noa::Math::min((shape_2d.y + (WARPS - 1)) / WARPS, MAX_BLOCKS);
    }

    template<typename T, typename U>
    static __global__ void isLess(T* input, uint pitch_input, T threshold,
                                  U* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = static_cast<U>(input[row * pitch_input + idx] < threshold);
    }

    template<typename T, typename U>
    static __global__ void isGreater(T* input, uint pitch_input, T threshold,
                                     U* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = static_cast<U>(threshold < input[row * pitch_input + idx]);
    }

    template<typename T, typename U>
    static __global__ void isWithin(T* input, uint pitch_input, T low, T high,
                                    U* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y) {
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x) {
                T tmp = input[row * pitch_input + idx];
                output[row * pitch_output + idx] = static_cast<U>(low < tmp && tmp < high);
            }
        }
    }

    template<typename T, typename U>
    static __global__ void logicNOT(T* input, uint pitch_input, U* output, uint pitch_output, uint2_t shape) {
        for (uint row = BLOCK_SIZE.y * blockIdx.x + threadIdx.y; row < shape.y; row += gridDim.x * BLOCK_SIZE.y)
            for (uint idx = threadIdx.x; idx < shape.x; idx += BLOCK_SIZE.x)
                output[row * pitch_output + idx] = static_cast<U>(!input[row * pitch_input + idx]);
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T, typename U>
    void isLess(T* input, T threshold, U* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::isLess,
                        input, threshold, output, elements);
    }

    template<typename T, typename U>
    void isLess(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::isLess,
                        input, pitch_input, threshold, output, pitch_output, shape_2d);
    }

    template<typename T, typename U>
    void isGreater(T* input, T threshold, U* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::isGreater,
                        input, threshold, output, elements);
    }

    template<typename T, typename U>
    void isGreater(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                   size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::isGreater,
                        input, pitch_input, threshold, output, pitch_output, shape_2d);
    }

    template<typename T, typename U>
    void isWithin(T* input, T low, T high, U* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::isWithin,
                        input, low, high, output, elements);
    }

    template<typename T, typename U>
    void isWithin(T* input, size_t pitch_input, T low, T high, U* output, size_t pitch_output,
                  size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::isWithin,
                        input, pitch_input, low, high, output, pitch_output, shape_2d);
    }

    template<typename T, typename U>
    void logicNOT(T* input, U* output, size_t elements, Stream& stream) {
        uint blocks = Details::Contiguous::getBlocks(elements);
        NOA_CUDA_LAUNCH(blocks, Details::Contiguous::BLOCK_SIZE, 0, stream.get(),
                        Details::Contiguous::logicNOT,
                        input, output, elements);
    }

    template<typename T, typename U>
    void logicNOT(T* input, size_t pitch_input, U* output, size_t pitch_output,
                  size3_t shape, Stream& stream) {
        uint2_t shape_2d(shape.x, getRows(shape));
        uint blocks = Details::Padded::getBlocks(shape_2d);
        NOA_CUDA_LAUNCH(blocks, Details::Padded::BLOCK_SIZE, 0, stream.get(),
                        Details::Padded::logicNOT,
                        input, pitch_input, output, pitch_output, shape_2d);
    }
}

// INSTANTIATIONS:
namespace Noa::CUDA::Math {
    #define INSTANTIATE_BOOLEANS(T, U)                                              \
    template void isLess<T, U>(T*, T, U*, size_t, Stream&);                         \
    template void isLess<T, U>(T*, size_t, T, U*, size_t, size3_t, Stream&);        \
    template void isGreater<T, U>(T*, T, U*, size_t, Stream&);                      \
    template void isGreater<T, U>(T*, size_t, T, U*, size_t, size3_t, Stream&);     \
    template void isWithin<T, U>(T*, T, T, U*, size_t, Stream&);                    \
    template void isWithin<T, U>(T*, size_t, T, T, U*, size_t, size3_t, Stream&)

    INSTANTIATE_BOOLEANS(float, float);
    INSTANTIATE_BOOLEANS(double, double);
    INSTANTIATE_BOOLEANS(int16_t, int16_t);
    INSTANTIATE_BOOLEANS(uint16_t, uint16_t);
    INSTANTIATE_BOOLEANS(int32_t, int32_t);
    INSTANTIATE_BOOLEANS(uint32_t, uint32_t);
    INSTANTIATE_BOOLEANS(char, char);
    INSTANTIATE_BOOLEANS(unsigned char, unsigned char);

    INSTANTIATE_BOOLEANS(float, bool);
    INSTANTIATE_BOOLEANS(double, bool);
    INSTANTIATE_BOOLEANS(int16_t, bool);
    INSTANTIATE_BOOLEANS(uint16_t, bool);
    INSTANTIATE_BOOLEANS(int32_t, bool);
    INSTANTIATE_BOOLEANS(uint32_t, bool);
    INSTANTIATE_BOOLEANS(char, bool);
    INSTANTIATE_BOOLEANS(unsigned char, bool);

    INSTANTIATE_BOOLEANS(float, int16_t);
    INSTANTIATE_BOOLEANS(float, uint16_t);
    INSTANTIATE_BOOLEANS(float, int32_t);
    INSTANTIATE_BOOLEANS(float, uint32_t);
    INSTANTIATE_BOOLEANS(float, char);
    INSTANTIATE_BOOLEANS(float, unsigned char);

    INSTANTIATE_BOOLEANS(double, int16_t);
    INSTANTIATE_BOOLEANS(double, uint16_t);
    INSTANTIATE_BOOLEANS(double, int32_t);
    INSTANTIATE_BOOLEANS(double, uint32_t);
    INSTANTIATE_BOOLEANS(double, char);
    INSTANTIATE_BOOLEANS(double, unsigned char);

    #define INSTANTIATE_LOGIC_NOT(T, U)                                    \
    template void logicNOT<T, U>(T*, U*, size_t, Stream&);                 \
    template void logicNOT<T, U>(T*, size_t, U*, size_t, size3_t, Stream&)

    INSTANTIATE_LOGIC_NOT(int16_t, int16_t);
    INSTANTIATE_LOGIC_NOT(uint16_t, uint16_t);
    INSTANTIATE_LOGIC_NOT(int32_t, int32_t);
    INSTANTIATE_LOGIC_NOT(uint32_t, uint32_t);
    INSTANTIATE_LOGIC_NOT(char, char);
    INSTANTIATE_LOGIC_NOT(unsigned char, unsigned char);
    INSTANTIATE_LOGIC_NOT(bool, bool);

    INSTANTIATE_LOGIC_NOT(int16_t, bool);
    INSTANTIATE_LOGIC_NOT(uint16_t, bool);
    INSTANTIATE_LOGIC_NOT(int32_t, bool);
    INSTANTIATE_LOGIC_NOT(uint32_t, bool);
    INSTANTIATE_LOGIC_NOT(char, bool);
    INSTANTIATE_LOGIC_NOT(unsigned char, bool);
}
