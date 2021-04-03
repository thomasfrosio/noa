#include "noa/gpu/cuda/math/Booleans.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/Math.h"

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

// KERNELS:
namespace Noa::CUDA::Math::Kernels {
    template<typename T, typename U>
    static __global__ void isLess(T* input, T threshold, U* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = input[idx] < threshold;
    }

    template<typename T, typename U>
    static __global__ void isLess(T* input, uint pitch_input, T threshold,
                                  U* output, uint pitch_output,
                                  uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = input[row * pitch_input + idx] < threshold;
    }

    template<typename T, typename U>
    static __global__ void isGreater(T* input, T threshold, U* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = threshold < input[idx];
    }

    template<typename T, typename U>
    static __global__ void isGreater(T* input, uint pitch_input, T threshold,
                                     U* output, uint pitch_output,
                                     uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = threshold < input[row * pitch_input + idx];
    }

    template<typename T, typename U>
    static __global__ void isWithin(T* input, T low, T high, U* output, uint elements) {
        T tmp;
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x) {
            tmp = input[idx];
            output[idx] = low < tmp && tmp < high;
        }
    }

    template<typename T, typename U>
    static __global__ void isWithin(T* input, uint pitch_input, T low, T high,
                                    U* output, uint pitch_output,
                                    uint elements_in_row, uint rows) {
        T tmp;
        for (uint row = blockIdx.x; row < rows; row += gridDim.x) {
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x) {
                tmp = input[row * pitch_input + idx];
                output[row * pitch_output + idx] = low < tmp && tmp < high;
            }
        }
    }

    template<typename T>
    static __global__ void logicNOT(T* input, T* output, uint elements) {
        for (uint idx = blockIdx.x * blockDim.x + threadIdx.x; idx < elements; idx += blockDim.x * gridDim.x)
            output[idx] = !input[idx];
    }

    template<typename T>
    static __global__ void logicNOT(T* input, uint pitch_input,
                                    T* output, uint pitch_output,
                                    uint elements_in_row, uint rows) {
        for (uint row = blockIdx.x; row < rows; row += gridDim.x)
            for (uint idx = threadIdx.x; idx < elements_in_row; idx += blockDim.x)
                output[row * pitch_output + idx] = !input[row * pitch_input + idx];
    }
}

// DEFINITIONS:
namespace Noa::CUDA::Math {
    template<typename T, typename U>
    void isLess(T* input, T threshold, U* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isLess,
                        input, threshold, output, elements);
    }

    template<typename T, typename U>
    void isLess(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isLess,
                        input, pitch_input, threshold, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void isGreater(T* input, T threshold, U* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isGreater,
                        input, threshold, output, elements);
    }

    template<typename T, typename U>
    void isGreater(T* input, size_t pitch_input, T threshold, U* output, size_t pitch_output,
                   size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isGreater,
                        input, pitch_input, threshold, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T, typename U>
    void isWithin(T* input, T low, T high, U* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isWithin,
                        input, low, high, output, elements);
    }

    template<typename T, typename U>
    void isWithin(T* input, size_t pitch_input, T low, T high, U* output, size_t pitch_output,
                  size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::isWithin,
                        input, pitch_input, low, high, output, pitch_output, shape.x, getRows(shape));
    }

    template<typename T>
    void logicNOT(T* input, T* output, size_t elements, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(elements);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::logicNOT,
                        input, output, elements);
    }

    template<typename T>
    void logicNOT(T* input, size_t pitch_input, T* output, size_t pitch_output,
                  size3_t shape, Stream& stream) {
        auto[total_blocks, threads_per_block] = getLaunchConfig(shape);
        NOA_CUDA_LAUNCH(total_blocks, threads_per_block, 0, stream.get(),
                        Kernels::logicNOT,
                        input, pitch_input, output, pitch_output, shape.x, getRows(shape));
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

    #define INSTANTIATE_LOGIC_NOT(T)                                    \
    template void logicNOT<T>(T*, T*, size_t, Stream&);                 \
    template void logicNOT<T>(T*, size_t, T*, size_t, size3_t, Stream&)

    INSTANTIATE_LOGIC_NOT(int16_t);
    INSTANTIATE_LOGIC_NOT(uint16_t);
    INSTANTIATE_LOGIC_NOT(int32_t);
    INSTANTIATE_LOGIC_NOT(uint32_t);
    INSTANTIATE_LOGIC_NOT(char);
    INSTANTIATE_LOGIC_NOT(unsigned char);
    INSTANTIATE_LOGIC_NOT(bool);
}
