#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = 128;
    constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, 4);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void linspace1D_(T* src, uint stride, uint elements, T start, T stop, T step, bool endpoint) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (endpoint && gid == elements - 1)
                    src[gid * stride] = stop;
                else if (gid < elements)
                    src[gid * stride] = start + static_cast<T>(gid) * step;
            }
        } else {
            NOA_ASSERT(stride == 1);
            (void) stride;
            const uint remaining = elements - base;
            src += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = BLOCK_SIZE * i + threadIdx.x;
                    if (endpoint && gid == remaining - 1)
                        src[gid] = stop;
                    else if (gid < remaining)
                        src[gid] = start + static_cast<T>(base + gid) * step;
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = base + i + ELEMENTS_PER_THREAD * threadIdx.x;
                    values[i] = endpoint && gid == elements - 1 ?
                                stop : start + static_cast<T>(gid) * step;
                }
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, src, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void linspace4D_(T* src, uint4_t stride, uint4_t shape,
                     T start, T stop, T step, bool endpoint, uint blocks_x) {
        const uint4_t logical_stride = shape.stride();
        const uint4_t last = shape - 1;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        const bool is_last_ij = gid[1] == last[1] && gid[0] == last[0];
        src += at(gid[0], gid[1], stride);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                const uint offset = ik * stride[2] + il * stride[3];
                if (endpoint && is_last_ij && ik == last[2] && il == last[3])
                    src[offset] = stop;
                else if (ik < shape[2] && il < shape[3])
                    src[offset] = start + static_cast<T>(at(gid[0], gid[1], ik, il, logical_stride)) * step;
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void linspace(T* src, size_t elements, T start, T stop, bool endpoint, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (elements <= 1)
            return set(src, elements, start, stream);

        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);

        const auto uint_elements = static_cast<uint>(elements);
        const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
        const int vec_size = noa::cuda::util::maxVectorCount(src);
        if (vec_size == 4) {
            stream.enqueue("memory::linspace", linspace1D_<T, 4>,
                           {blocks, BLOCK_SIZE}, src, 1, uint_elements, start, stop, step, endpoint);
        } else if (vec_size == 2) {
            stream.enqueue("memory::linspace", linspace1D_<T, 2>,
                           {blocks, BLOCK_SIZE}, src, 1, uint_elements, start, stop, step, endpoint);
        } else {
            stream.enqueue("memory::linspace", linspace1D_<T, 1>,
                           {blocks, BLOCK_SIZE}, src, 1, uint_elements, start, stop, step, endpoint);
        }
    }

    template<typename T>
    void linspace(T* src, size4_t stride, size4_t shape, T start, T stop, bool endpoint, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t elements = shape.elements();
        if (elements <= 1)
            return set(src, elements, start, stream);

        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);

        const bool4_t is_contiguous = isContiguous(stride, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const auto uint_elements = static_cast<uint>(elements);
            const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
            const uint vec_size = is_contiguous[3] ? noa::cuda::util::maxVectorCount(src) : 1;

            if (vec_size == 4) {
                stream.enqueue("memory::linspace", linspace1D_<T, 4>,
                               {blocks, BLOCK_SIZE}, src, stride[3], uint_elements, start, stop, step, endpoint);
            } else if (vec_size == 2) {
                stream.enqueue("memory::linspace", linspace1D_<T, 2>,
                               {blocks, BLOCK_SIZE}, src, stride[3], uint_elements, start, stop, step, endpoint);
            } else {
                stream.enqueue("memory::linspace", linspace1D_<T, 1>,
                               {blocks, BLOCK_SIZE}, src, stride[3], uint_elements, start, stop, step, endpoint);
            }
        } else {
            const uint4_t uint_shape{shape};
            const uint blocks_x = noa::math::divideUp(uint_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(uint_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, uint_shape[1], uint_shape[0]);
            stream.enqueue("memory::linspace", linspace4D_<T>, {blocks, BLOCK_SIZE_2D},
                           src, uint4_t{stride}, uint_shape, start, stop, step, endpoint, blocks_x);
        }
    }

    #define NOA_INSTANTIATE_LINSPACE_(T)                        \
    template void linspace<T>(T*, size_t, T, T, bool, Stream&); \
    template void linspace<T>(T*, size4_t, size4_t, T, T, bool, Stream&)

    NOA_INSTANTIATE_LINSPACE_(int8_t);
    NOA_INSTANTIATE_LINSPACE_(int16_t);
    NOA_INSTANTIATE_LINSPACE_(int32_t);
    NOA_INSTANTIATE_LINSPACE_(int64_t);
    NOA_INSTANTIATE_LINSPACE_(uint8_t);
    NOA_INSTANTIATE_LINSPACE_(uint16_t);
    NOA_INSTANTIATE_LINSPACE_(uint32_t);
    NOA_INSTANTIATE_LINSPACE_(uint64_t);
    NOA_INSTANTIATE_LINSPACE_(half_t);
    NOA_INSTANTIATE_LINSPACE_(float);
    NOA_INSTANTIATE_LINSPACE_(double);
    NOA_INSTANTIATE_LINSPACE_(chalf_t);
    NOA_INSTANTIATE_LINSPACE_(cfloat_t);
    NOA_INSTANTIATE_LINSPACE_(cdouble_t);
}