#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace {
    using namespace noa;

    constexpr uint MAX_THREADS = 512;
    constexpr dim3 THREADS(32, MAX_THREADS / 32);

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void copy1D_(const T* __restrict__ src, uint src_stride,
                 T* __restrict__ dst, uint dst_stride, uint shape) {
        const uint gid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
        src += gid * src_stride;
        dst += gid * dst_stride;

        for (int i = 0; i < 4; ++i) {
            const uint x = gid + blockDim.x * i;
            if (x < shape)
                dst[x * dst_stride] = src[x * src_stride];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void copyND_(const T* __restrict__ src, uint4_t src_stride,
                 T* __restrict__ dst, uint4_t dst_stride, uint3_t shape, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] > shape[1])
            return;
        src += at(batch, gid[0], gid[1], src_stride);
        dst += at(batch, gid[0], gid[1], dst_stride);

        for (int i = 0; i < 2; ++i) {
            const uint x = gid[2] + THREADS.x * i;
            if (x < shape[2])
                dst[x * dst_stride[3]] = src[x * src_stride[3]];
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void copy(const T* src, size4_t src_stride, T* dst, size4_t dst_stride, size4_t shape, Stream& stream) {
        uint3_t u_shape(shape.get() + 1);
        if (shape.ndim() == 1) { // no batch, 1D
            const uint threads = math::min(math::nextMultipleOf(u_shape[3], 32U), MAX_THREADS);
            const dim3 blocks(math::divideUp(u_shape[3], threads * 4));
            stream.enqueue("memory::copy1D_", copy1D_<T>, {blocks, threads},
                           src, src_stride[3], dst, dst_stride[3], shape[3]);
        } else {
            const uint blocks_x = math::divideUp(u_shape[2], THREADS.x * 2);
            const uint blocks_y = math::divideUp(u_shape[1], THREADS.y);
            const dim3 blocks(blocks_x * blocks_y, u_shape[0], shape[0]);
            stream.enqueue("memory::copyND_", copyND_<T>, {blocks, THREADS},
                           src, uint4_t{src_stride}, dst, uint4_t{dst_stride}, u_shape, blocks_x);
        }
    }

    #define NOA_INSTANTIATE_COPY_(T) \
    template void copy<T>(const T*, size4_t, T*, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_COPY_(char);
    NOA_INSTANTIATE_COPY_(short);
    NOA_INSTANTIATE_COPY_(int);
    NOA_INSTANTIATE_COPY_(long);
    NOA_INSTANTIATE_COPY_(long long);
    NOA_INSTANTIATE_COPY_(unsigned char);
    NOA_INSTANTIATE_COPY_(unsigned short);
    NOA_INSTANTIATE_COPY_(unsigned int);
    NOA_INSTANTIATE_COPY_(unsigned long);
    NOA_INSTANTIATE_COPY_(unsigned long long);
    NOA_INSTANTIATE_COPY_(half_t);
    NOA_INSTANTIATE_COPY_(float);
    NOA_INSTANTIATE_COPY_(double);
    NOA_INSTANTIATE_COPY_(chalf_t);
    NOA_INSTANTIATE_COPY_(cfloat_t);
    NOA_INSTANTIATE_COPY_(cdouble_t);
}
