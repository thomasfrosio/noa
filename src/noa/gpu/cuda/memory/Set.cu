#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Set.h"

namespace {
    using namespace noa;

    template<typename T>
    __global__ void set_(T* array, size_t elements, T value) {
        #pragma unroll 10
        for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
             idx < elements;
             idx += blockDim.x * gridDim.x)
            array[idx] = value;
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void set(T* array, size_t elements, T value, Stream& stream) {
        uint threads = 512U;
        uint blocks = math::min(noa::math::divideUp(static_cast<uint>(elements), threads), 8192U);
        set_<<<blocks, threads, 0, stream.id()>>>(array, elements, value);
        NOA_THROW_IF(cudaGetLastError());
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
    NOA_INSTANTIATE_SET_(cfloat_t);
    NOA_INSTANTIATE_SET_(cdouble_t);
}
