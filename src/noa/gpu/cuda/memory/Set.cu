#include "noa/gpu/cuda/memory/Set.h"

namespace {
    using namespace Noa;

    template<typename T>
    __global__ void set_(T* array, size_t elements, T value) {
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < elements;
             idx += blockDim.x * gridDim.x)
            array[idx] = value;
    }
}

namespace Noa::CUDA::Memory {
    template<typename T>
    void set(T* array, size_t elements, T value, Stream& stream) {
        if (value == 0) {
            NOA_THROW_IF(cudaMemsetAsync(array, 0, elements * sizeof(T), stream.id()));
        } else {
            uint threads = 256U;
            uint blocks = Math::min((elements + threads - 1) / threads, static_cast<size_t>(8192U));
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(), set_, array, elements, value);
        }
    }
    #define INSTANTIATE_SET(T) \
    template void set<T>(T*, size_t, T, Stream&)

    INSTANTIATE_SET(char);
    INSTANTIATE_SET(short);
    INSTANTIATE_SET(int);
    INSTANTIATE_SET(long);
    INSTANTIATE_SET(long long);
    INSTANTIATE_SET(unsigned char);
    INSTANTIATE_SET(unsigned short);
    INSTANTIATE_SET(unsigned int);
    INSTANTIATE_SET(unsigned long);
    INSTANTIATE_SET(unsigned long long);
    INSTANTIATE_SET(float);
    INSTANTIATE_SET(double);
}
