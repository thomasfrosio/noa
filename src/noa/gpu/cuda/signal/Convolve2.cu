#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/signal/Convolve.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr i32 MAX_FILTER_SIZE = 11;
    constexpr i32 MAX_FILTER_BYTES = MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(f64);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_2d_(AccessorRestrict<const T, 4, u32> input,
                      AccessorRestrict<T, 4, u32> output,
                      Shape2<i32> shape, Shape2<i32> filter_shape, u32 blocks_x) {

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};
        const auto input_2d = input[gid[0]][gid[1]];

        const auto OFFSET = static_cast<i32>(BLOCK_SIZE.x); // block is 16x16 square
        const auto PADDING = Vec2<i32>(filter_shape.vec() - 1);
        const auto HALO = Vec2<i32>(PADDING / 2);
        const auto SHARED_LEN = Vec2<i32>(OFFSET + PADDING);
        T* shared = noa::cuda::utils::block_dynamic_shared_resource<T>();

        // Load to shared memory. Loop to take into account padding.
        for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            const i32 i_y = gy - HALO[0];
            const bool is_in_y = i_y >= 0 && i_y < shape[0];
            for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                const i32 i_x = gx - HALO[1];
                const bool is_in_x = i_x >= 0 && i_x < shape[1];
                shared[ly * SHARED_LEN[1] + lx] = is_in_y && is_in_x ? input_2d(i_y, i_x) : T{0};
            }
        }
        noa::cuda::utils::block_synchronize();

        if (gid[2] < shape[0] && gid[3] < shape[1]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result{0};
            for (i32 y = 0; y < filter_shape[0]; ++y)
                for (i32 x = 0; x < filter_shape[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * window[y * filter_shape[1] + x];
            output(gid) = result;
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve_2d(const T* input, const Strides4<i64>& input_strides,
                     T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                     const U* filter, const Shape2<i64>& filter_shape, Stream& stream) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(filter_shape.elements() * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(noa::all(filter_shape % 2 == 1));

        if (noa::all(filter_shape <= 1))
            return noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        NOA_THROW_IF(cudaMemcpyToSymbolAsync(
                cfilter, filter, filter_shape.elements() * sizeof(T),
                0, cudaMemcpyDefault, stream.get()));

        const auto shape_2d = shape.filter(2, 3).as_safe<i32>();
        const u32 blocks_x = noa::math::divide_up(static_cast<u32>(shape_2d[1]), BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(static_cast<u32>(shape_2d[0]), BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const u32 shared_bytes = (BLOCK_SIZE.x + filter_shape[1] - 1) *
                                 (BLOCK_SIZE.y + filter_shape[0] - 1) * sizeof(T);
        const auto config = LaunchConfig{blocks, BLOCK_SIZE, shared_bytes};
        const auto input_accessor = AccessorRestrict<const T, 4, u32> (input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32> (output, output_strides.as_safe<u32>());

        stream.enqueue("convolve_2d", convolve_2d_<T>, config,
                       input_accessor, output_accessor, shape_2d,
                       filter_shape.as_safe<i32>(), blocks_x);
    }

    #define NOA_INSTANTIATE_CONV2_(T) \
    template void convolve_2d<T,T,void>(    \
        const T*, const Strides4<i64>&,     \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&,                 \
        const T*, const Shape2<i64>&, Stream&)

    NOA_INSTANTIATE_CONV2_(f16);
    NOA_INSTANTIATE_CONV2_(f32);
    NOA_INSTANTIATE_CONV2_(f64);
}
