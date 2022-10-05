#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Linspace.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, 4);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void linspace1D_(T* src, uint32_t stride, uint32_t elements, T start, T stop, T step, bool endpoint) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (endpoint && gid == elements - 1)
                    src[gid * stride] = stop;
                else if (gid < elements)
                    src[gid * stride] = start + static_cast<T>(gid) * step;
            }
        } else {
            NOA_ASSERT(stride == 1);
            (void) stride;
            const uint32_t remaining = elements - base;
            src += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t gid = BLOCK_SIZE * i + threadIdx.x;
                    if (endpoint && gid == remaining - 1)
                        src[gid] = stop;
                    else if (gid < remaining)
                        src[gid] = start + static_cast<T>(base + gid) * step;
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t gid = base + i + ELEMENTS_PER_THREAD * threadIdx.x;
                    values[i] = endpoint && gid == elements - 1 ?
                                stop : start + static_cast<T>(gid) * step;
                }
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, src, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void linspace4D_(Accessor<T, 4, uint32_t> src, uint4_t shape,
                     T start, T stop, T step, bool endpoint, uint32_t blocks_x) {
        const uint4_t logical_strides = shape.strides();
        const uint4_t last = shape - 1;
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        const bool is_last_ij = gid[1] == last[1] && gid[0] == last[0];
        const auto src_ = src[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint32_t ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (endpoint && is_last_ij && ik == last[2] && il == last[3])
                    src_(ik, il) = stop;
                else if (ik < shape[2] && il < shape[3])
                    src_(ik, il) = start + static_cast<T>(indexing::at(gid[0], gid[1], ik, il, logical_strides)) * step;
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    T linspace(const shared_t<T[]>& src, dim_t elements,
               T start, T stop, bool endpoint, Stream& stream) {
        if (elements <= 1) {
            set(src, elements, start, stream);
            return T{0};
        }

        NOA_ASSERT_DEVICE_PTR(src.get(), stream.device());
        auto[count, delta, step] = linspaceStep(elements, start, stop, endpoint);

        const auto uint_elements = static_cast<uint32_t>(elements);
        const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
        const int32_t vec_size = noa::cuda::util::maxVectorCount(src.get());
        if (vec_size == 4) {
            stream.enqueue("memory::linspace", linspace1D_<T, 4>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, stop, step, endpoint);
        } else if (vec_size == 2) {
            stream.enqueue("memory::linspace", linspace1D_<T, 2>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, stop, step, endpoint);
        } else {
            stream.enqueue("memory::linspace", linspace1D_<T, 1>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, start, stop, step, endpoint);
        }
        stream.attach(src);
        return step;
    }

    template<typename T, typename>
    T linspace(const shared_t<T[]>& src, dim4_t strides, dim4_t shape,
               T start, T stop, bool endpoint, Stream& stream) {
        const dim_t elements = shape.elements();
        if (elements <= 1) {
            set(src, elements, start, stream);
            return T{0};
        }

        NOA_ASSERT_DEVICE_PTR(src.get(), stream.device());
        auto[count, delta, step] = linspaceStep(elements, start, stop, endpoint);

        const bool4_t is_contiguous = indexing::isContiguous(strides, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const auto uint_elements = safe_cast<uint>(elements);
            const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
            const uint32_t vec_size = is_contiguous[3] ? noa::cuda::util::maxVectorCount(src.get()) : 1;

            if (vec_size == 4) {
                stream.enqueue("memory::linspace", linspace1D_<T, 4>,
                               {blocks, BLOCK_SIZE}, src.get(), strides[3], uint_elements, start, stop, step, endpoint);
            } else if (vec_size == 2) {
                stream.enqueue("memory::linspace", linspace1D_<T, 2>,
                               {blocks, BLOCK_SIZE}, src.get(), strides[3], uint_elements, start, stop, step, endpoint);
            } else {
                stream.enqueue("memory::linspace", linspace1D_<T, 1>,
                               {blocks, BLOCK_SIZE}, src.get(), strides[3], uint_elements, start, stop, step, endpoint);
            }
        } else {
            const auto uint_shape = safe_cast<uint4_t>(shape);
            const uint32_t blocks_x = noa::math::divideUp(uint_shape[3], BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(uint_shape[2], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, uint_shape[1], uint_shape[0]);
            const Accessor<T, 4, uint32_t> src_accessor(src.get(), safe_cast<uint4_t>(strides));
            stream.enqueue("memory::linspace", linspace4D_<T>, {blocks, BLOCK_SIZE_2D},
                           src_accessor, uint_shape, start, stop, step, endpoint, blocks_x);
        }
        stream.attach(src);
        return step;
    }

    #define NOA_INSTANTIATE_LINSPACE_(T)                                            \
    template T linspace<T, void>(const shared_t<T[]>&, dim_t, T, T, bool, Stream&); \
    template T linspace<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, T, T, bool, Stream&)

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
