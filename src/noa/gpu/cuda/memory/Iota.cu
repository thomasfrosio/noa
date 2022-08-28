#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Arange.h"
#include "noa/gpu/cuda/memory/Iota.h"
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
    void iota1D_(T* src, uint strides, uint elements, uint tile) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    src[gid * strides] = static_cast<T>(gid % tile);
            }
        } else {
            NOA_ASSERT(strides == 1);
            (void) strides;
            const uint remaining = elements - base;
            src += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        src[gid] = static_cast<T>((base + gid) % tile);
                }
            } else {
                T values[ELEMENTS_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = base + i + ELEMENTS_PER_THREAD * threadIdx.x;
                    values[i] = static_cast<T>(gid % tile);
                }
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(values, src, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void iota4D_(T* src, uint4_t strides, uint4_t shape, uint4_t tile, uint blocks_x) {
        const uint4_t tile_strides = tile.strides();
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        src += indexing::at(gid[0], gid[1], strides);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[2] && il < shape[3]) {
                    const uint value = indexing::at(gid[0] % tile[0],
                                                    gid[1] % tile[1],
                                                    ik % tile[2],
                                                    il % tile[3],
                                                    tile_strides);
                    src[ik * strides[2] + il * strides[3]] = static_cast<T>(value);
                }
            }
        }
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    void iota(const shared_t<T[]>& src, size_t elements, size_t tile, Stream& stream) {
        if (tile == elements)
            return arange(src, elements, T{0}, T{1}, stream);

        const auto uint_elements = static_cast<uint>(elements);
        const dim3 blocks(noa::math::divideUp(uint_elements, BLOCK_WORK_SIZE));
        const int vec_size = noa::cuda::util::maxVectorCount(src.get());
        if (vec_size == 4) {
            stream.enqueue("memory::iota", iota1D_<T, 4>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, tile);
        } else if (vec_size == 2) {
            stream.enqueue("memory::iota", iota1D_<T, 2>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, tile);
        } else {
            stream.enqueue("memory::iota", iota1D_<T, 1>,
                           {blocks, BLOCK_SIZE}, src.get(), 1, uint_elements, tile);
        }
        stream.attach(src);
    }

    template<typename T, typename>
    void iota(const shared_t<T[]>& src, size4_t strides, size4_t shape, size4_t tile, Stream& stream) {
        if (all(tile == shape))
            return arange(src, strides, shape, T{0}, T{1}, stream);

        const uint4_t uint_shape(shape);
        const uint blocks_x = noa::math::divideUp(uint_shape[3], BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = noa::math::divideUp(uint_shape[2], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, uint_shape[1], uint_shape[0]);
        stream.enqueue("memory::iota", iota4D_<T>, {blocks, BLOCK_SIZE_2D},
                       src.get(), uint4_t(strides), uint_shape, uint4_t(tile), blocks_x);
        stream.attach(src);
    }

    #define NOA_INSTANTIATE_IOTA_(T)                                            \
    template void iota<T, void>(const shared_t<T[]>&, size_t, size_t, Stream&); \
    template void iota<T, void>(const shared_t<T[]>&, size4_t, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_IOTA_(int8_t);
    NOA_INSTANTIATE_IOTA_(int16_t);
    NOA_INSTANTIATE_IOTA_(int32_t);
    NOA_INSTANTIATE_IOTA_(int64_t);
    NOA_INSTANTIATE_IOTA_(uint8_t);
    NOA_INSTANTIATE_IOTA_(uint16_t);
    NOA_INSTANTIATE_IOTA_(uint32_t);
    NOA_INSTANTIATE_IOTA_(uint64_t);
    NOA_INSTANTIATE_IOTA_(half_t);
    NOA_INSTANTIATE_IOTA_(float);
    NOA_INSTANTIATE_IOTA_(double);
}
