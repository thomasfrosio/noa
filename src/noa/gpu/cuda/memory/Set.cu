#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 256;
    constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, 4);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void set1D_(T* src, uint2_t strides, uint32_t elements_per_batch, T value) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        src += blockIdx.y * strides[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements_per_batch)
                    src[gid * strides[1]] = value;
            }
        } else {
            NOA_ASSERT(strides[1] == 1);
            const uint32_t remaining = elements_per_batch - base;
            src += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        src[gid] = value;
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                #pragma unroll
                for (uint32_t i = 0; i < ELEMENTS_PER_THREAD; ++i)
                    values[i] = value;
                cuda::utils::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, src, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void set4D_(Accessor<T, 4, uint32_t> src,
                uint2_t shape, T value, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto src_ = src[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint32_t ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    src_(ik, il) = value;
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T, typename>
    void set(T* src, dim_t elements, T value, Stream& stream) {
        if (!elements)
            return;

        NOA_ASSERT_DEVICE_PTR(src, stream.device());
        const uint2_t strides{0, 1};
        const auto elements_per_batch = safe_cast<uint32_t>(elements);
        const dim3 blocks(noa::math::divideUp(elements_per_batch, BLOCK_WORK_SIZE));
        const int32_t vec_size = noa::cuda::utils::maxVectorCount(src);
        if (vec_size == 4) {
            stream.enqueue("memory::set", set1D_<T, 4>,
                           {blocks, BLOCK_SIZE}, src, strides, elements_per_batch, value);
        } else if (vec_size == 2) {
            stream.enqueue("memory::set", set1D_<T, 2>,
                           {blocks, BLOCK_SIZE}, src, strides, elements_per_batch, value);
        } else {
            stream.enqueue("memory::set", set1D_<T, 1>,
                           {blocks, BLOCK_SIZE}, src, strides, elements_per_batch, value);
        }
    }

    template<typename T, typename>
    void set(const shared_t<T[]>& src, dim4_t strides, dim4_t shape, T value, Stream& stream) {
        if (!shape.elements())
            return;

        NOA_ASSERT_DEVICE_PTR(src.get(), stream.device());
        // Assume rightmost order, since it should have been reordered by now.
        const bool4_t is_contiguous = indexing::isContiguous(strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const auto uint_strides = safe_cast<uint2_t>(dim2_t{strides[0], strides[3]});
            const auto elements_per_batch =
                    safe_cast<uint32_t>(is_contiguous[0] ? shape.elements() : dim3_t(shape.get(1)).elements());
            const dim3 blocks(noa::math::divideUp(elements_per_batch, BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint32_t vec_size = is_contiguous[3] ? noa::cuda::utils::maxVectorCount(src.get()) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = uint_strides[0] % vec_size ? 1 : vec_size;

            if (vec_size == 4) {
                stream.enqueue("memory::set", set1D_<T, 4>,
                               {blocks, BLOCK_SIZE}, src.get(), uint_strides, elements_per_batch, value);
            } else if (vec_size == 2) {
                stream.enqueue("memory::set", set1D_<T, 2>,
                               {blocks, BLOCK_SIZE}, src.get(), uint_strides, elements_per_batch, value);
            } else {
                stream.enqueue("memory::set", set1D_<T, 1>,
                               {blocks, BLOCK_SIZE}, src.get(), uint_strides, elements_per_batch, value);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const Accessor<T, 4, uint32_t> accessor(src.get(), safe_cast<uint4_t>(strides));
            stream.enqueue("memory::set", set4D_<T>, {blocks, BLOCK_SIZE_2D},
                           accessor, i_shape, value, blocks_x);
        }
        stream.attach(src);
    }

    #define NOA_INSTANTIATE_SET_(T)                     \
    template void set<T, void>(T*, dim_t, T, Stream&);  \
    template void set<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, T, Stream&)

    NOA_INSTANTIATE_SET_(bool);
    NOA_INSTANTIATE_SET_(int8_t);
    NOA_INSTANTIATE_SET_(int16_t);
    NOA_INSTANTIATE_SET_(int32_t);
    NOA_INSTANTIATE_SET_(int64_t);
    NOA_INSTANTIATE_SET_(uint8_t);
    NOA_INSTANTIATE_SET_(uint16_t);
    NOA_INSTANTIATE_SET_(uint32_t);
    NOA_INSTANTIATE_SET_(uint64_t);
    NOA_INSTANTIATE_SET_(half_t);
    NOA_INSTANTIATE_SET_(float);
    NOA_INSTANTIATE_SET_(double);
    NOA_INSTANTIATE_SET_(chalf_t);
    NOA_INSTANTIATE_SET_(cfloat_t);
    NOA_INSTANTIATE_SET_(cdouble_t);
}
