#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Set.h"

namespace {
    using namespace noa;

    constexpr uint MAX_THREADS = 512;
    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void set1D_(T* src, uint stride, uint shape, T value) {
        const uint gid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
        src += gid * stride;

        for (int i = 0; i < 4; ++i) {
            const uint x = gid + blockDim.x * i;
            if (x < shape)
                src[x * stride] = value;
        }
    }

    constexpr dim3 THREADS(32, MAX_THREADS / 32);
    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void setND_(T* src, uint4_t stride, uint3_t shape, T value, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] > shape[1])
            return;
        src += at(batch, gid[0], gid[1], stride);

        for (int i = 0; i < 2; ++i) {
            const uint x = gid[2] + THREADS.x * i;
            if (x < shape[2])
                src[x * stride[3]] = value;
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void set(T* src, size_t elements, T value, Stream& stream) {
        const auto size = static_cast<uint>(elements);
        const uint threads = math::min(math::nextMultipleOf(size, 32U), MAX_THREADS);
        const dim3 blocks(math::divideUp(size, threads * 4));
        set1D_<<<blocks, threads, 0, stream.id()>>>(src, 1, size, value);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void set(T* src, size4_t stride, size4_t shape, T value, Stream& stream) {
        uint3_t u_shape(shape.get() + 1);
        if (shape.ndim() == 1) {
            const uint threads = math::min(math::nextMultipleOf(u_shape[3], 32U), MAX_THREADS);
            const dim3 blocks(math::divideUp(u_shape[3], threads * 4));
            set1D_<<<blocks, threads, 0, stream.id()>>>(src, stride[3], shape[3], value);
        } else {
            const uint blocks_x = math::divideUp(u_shape[2], THREADS.x * 2);
            const uint blocks_y = math::divideUp(u_shape[1], THREADS.y);
            const dim3 blocks(blocks_x * blocks_y, u_shape[0], shape[0]);
            setND_<<<blocks, THREADS, 0, stream.id()>>>(src, uint4_t{stride}, u_shape, value, blocks_x);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_SET_(T)                 \
    template void set<T>(T*, size_t, T, Stream&);   \
    template void set<T>(T*, size4_t, size4_t, T, Stream&);

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
    NOA_INSTANTIATE_SET_(half_t);
    NOA_INSTANTIATE_SET_(float);
    NOA_INSTANTIATE_SET_(double);
    NOA_INSTANTIATE_SET_(chalf_t);
    NOA_INSTANTIATE_SET_(cfloat_t);
    NOA_INSTANTIATE_SET_(cdouble_t);
}
