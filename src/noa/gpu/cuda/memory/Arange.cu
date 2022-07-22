#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Arange.h"
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
    void arange1D_(T* src, uint stride, uint elements, T start, T step) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
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
                    if (gid < remaining)
                        src[gid] = start + static_cast<T>(base + gid) * step;
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = base + i + ELEMENTS_PER_THREAD * threadIdx.x;
                    values[i] = start + static_cast<T>(gid) * step;
                }
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, src, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void arange4D_(T* src, uint4_t stride, uint4_t shape, T start, T step, uint blocks_x) {
        const uint4_t logical_stride = shape.strides();
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        src += indexing::at(gid[0], gid[1], stride);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[2] && il < shape[3]) {
                    const uint offset = indexing::at(gid[0], gid[1], ik, il, logical_stride);
                    src[ik * stride[2] + il * stride[3]] = start + static_cast<T>(offset) * step;
                }
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    void arange(const shared_t<T[]>& src, size_t elements, T start, T step, Stream& stream) {
        if (!elements)
            return;

        const auto uint_elements = static_cast<uint>(elements);
        const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
        const int vec_size = noa::cuda::util::maxVectorCount(src.get());
        if (vec_size == 4) {
            stream.enqueue("memory::arange", arange1D_<T, 4>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, step);
        } else if (vec_size == 2) {
            stream.enqueue("memory::arange", arange1D_<T, 2>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, step);
        } else {
            stream.enqueue("memory::arange", arange1D_<T, 1>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, step);
        }
        stream.attach(src);
    }

    template<typename T, typename>
    void arange(const shared_t<T[]>& src, size4_t stride, size4_t shape, T start, T step, Stream& stream) {
        if (!shape.elements())
            return;

        const bool4_t is_contiguous = indexing::isContiguous(stride, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const auto uint_elements = static_cast<uint>(shape.elements());
            const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
            const uint vec_size = is_contiguous[3] ? noa::cuda::util::maxVectorCount(src.get()) : 1;

            if (vec_size == 4) {
                stream.enqueue("memory::arange", arange1D_<T, 4>,
                               {blocks, BLOCK_SIZE}, src.get(), stride[3], uint_elements, start, step);
            } else if (vec_size == 2) {
                stream.enqueue("memory::arange", arange1D_<T, 2>,
                               {blocks, BLOCK_SIZE}, src.get(), stride[3], uint_elements, start, step);
            } else {
                stream.enqueue("memory::arange", arange1D_<T, 1>,
                               {blocks, BLOCK_SIZE}, src.get(), stride[3], uint_elements, start, step);
            }
        } else {
            const uint4_t uint_shape{shape};
            const uint blocks_x = noa::math::divideUp(uint_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(uint_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, uint_shape[1], uint_shape[0]);
            stream.enqueue("memory::arange", arange4D_<T>, {blocks, BLOCK_SIZE_2D},
                           src.get(), uint4_t{stride}, uint_shape, start, step, blocks_x);
        }
        stream.attach(src);
    }

    #define NOA_INSTANTIATE_ARANGE_(T)                                          \
    template void arange<T, void>(const shared_t<T[]>&, size_t, T, T, Stream&); \
    template void arange<T, void>(const shared_t<T[]>&, size4_t, size4_t, T, T, Stream&)

    NOA_INSTANTIATE_ARANGE_(int8_t);
    NOA_INSTANTIATE_ARANGE_(int16_t);
    NOA_INSTANTIATE_ARANGE_(int32_t);
    NOA_INSTANTIATE_ARANGE_(int64_t);
    NOA_INSTANTIATE_ARANGE_(uint8_t);
    NOA_INSTANTIATE_ARANGE_(uint16_t);
    NOA_INSTANTIATE_ARANGE_(uint32_t);
    NOA_INSTANTIATE_ARANGE_(uint64_t);
    NOA_INSTANTIATE_ARANGE_(half_t);
    NOA_INSTANTIATE_ARANGE_(float);
    NOA_INSTANTIATE_ARANGE_(double);
    NOA_INSTANTIATE_ARANGE_(chalf_t);
    NOA_INSTANTIATE_ARANGE_(cfloat_t);
    NOA_INSTANTIATE_ARANGE_(cdouble_t);
}
