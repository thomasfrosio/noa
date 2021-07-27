#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Set.h"

namespace {
    using namespace noa;

    template<typename T>
    __global__ void set_(T* array, size_t elements, T value) {
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < elements;
             idx += blockDim.x * gridDim.x)
            array[idx] = value;
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void set(T* array, size_t elements, T value, Stream& stream) {
        if (value == 0) {
            NOA_THROW_IF(cudaMemsetAsync(array, 0, elements * sizeof(T), stream.id()));
        } else {
            uint threads = 512U;
            uint blocks = math::min((elements + threads - 1) / threads, static_cast<size_t>(8192U));
            set_<<<blocks, threads, 0, stream.id()>>>(array, elements, value);
            NOA_THROW_IF(cudaPeekAtLastError());
        }
    }
    #define NOA_INSTANTIATE_SET_(T) \
    template void set<T>(T*, size_t, T, Stream&)

    NOA_INSTANTIATE_SET_(char);
    NOA_INSTANTIATE_SET_(short);
    NOA_INSTANTIATE_SET_(int);
    NOA_INSTANTIATE_SET_(long);
    NOA_INSTANTIATE_SET_(long long);
    NOA_INSTANTIATE_SET_(unsigned char);
    NOA_INSTANTIATE_SET_(unsigned short);
    NOA_INSTANTIATE_SET_(unsigned int);
    NOA_INSTANTIATE_SET_(unsigned long);
    NOA_INSTANTIATE_SET_(unsigned long long);
    NOA_INSTANTIATE_SET_(float);
    NOA_INSTANTIATE_SET_(double);
}
