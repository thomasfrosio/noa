#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/signal/Convolve.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr i32 MAX_FILTER_SIZE = 5;
    constexpr i32 MAX_FILTER_BYTES = MAX_FILTER_SIZE * MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(f64);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T, i32 FILTER_LEN>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_3d_square_(AccessorRestrict<const T, 4, u32> input,
                             AccessorRestrict<T, 4, u32> output,
                             Shape3<i32> shape, u32 blocks_x) {
        static_assert(FILTER_LEN % 2); // only support odd windows.
        constexpr i32 PADDING = FILTER_LEN - 1; // assume odd
        constexpr i32 HALO = FILTER_LEN / 2;
        constexpr auto SHARED_SHAPE = Shape3<i32>(FILTER_LEN, BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING);
        constexpr auto SHARED_SIZE = SHARED_SHAPE.elements();

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_3d = input[gid[0]];

        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[SHARED_SIZE];
        T* shared = reinterpret_cast<T*>(buffer);

        // Load shared memory. Loop to take into account padding.
        for (i32 lz = 0, gz = gid[1]; lz < SHARED_SHAPE[0]; ++lz, ++gz) {
            const i32 i_z = gz - HALO;
            const i32 tmp_z = lz * SHARED_SHAPE[1] * SHARED_SHAPE[2];
            const bool is_in_z = i_z >= 0 && i_z < shape[0];
            for (i32 ly = tid[0], gy = gid[2]; ly < SHARED_SHAPE[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                const i32 i_y = gy - HALO;
                const i32 tmp = tmp_z + ly * SHARED_SHAPE[2];
                const bool is_in_y = i_y >= 0 && i_y < shape[1];
                for (i32 lx = tid[1], gx = gid[3]; lx < SHARED_SHAPE[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                    const i32 i_x = gx - HALO;
                    const bool is_in_x = i_x >= 0 && i_x < shape[2];
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ? input_3d(i_z, i_y, i_x) : T{0};
                }
            }
        }
        noa::cuda::utils::block_synchronize();

        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result{0};
            for (i32 z = 0; z < FILTER_LEN; ++z)
                for (i32 y = 0; y < FILTER_LEN; ++y)
                    for (i32 x = 0; x < FILTER_LEN; ++x)
                        result += shared[(z * SHARED_SHAPE[1] + tid[0] + y) * SHARED_SHAPE[2] + tid[1] + x] *
                                  window[(z * FILTER_LEN + y) * FILTER_LEN + x];
            output(gid) = result;
        }
    }

    // Version with filter_length not fixed at compile time.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve_3d_(AccessorRestrict<const T, 4, u32> input,
                      AccessorRestrict<T, 4, u32> output,
                      Shape3<i32> shape, Shape3<i32> filter_length, u32 blocks_x) {
        const auto padding = Vec3<i32>(filter_length.vec() - 1); // assume odd
        const auto halo = padding / 2;
        const auto shared_shape = Vec3<i32>(filter_length[0], BLOCK_SIZE.y + padding[1], BLOCK_SIZE.x + padding[2]);

        const auto index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const auto tid = Vec2<i32>{threadIdx.y, threadIdx.x};
        const auto gid = Vec4<i32>{blockIdx.z,
                                   blockIdx.y,
                                   BLOCK_SIZE.y * index[0] + tid[0],
                                   BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_3d = input[gid[0]];

        // Load shared memory. Loop to take into account padding.
        T* shared = noa::cuda::utils::block_dynamic_shared_resource<T>();
        for (i32 lz = 0, gz = gid[1]; lz < shared_shape[0]; ++lz, ++gz) {
            const i32 i_z = gz - halo[0];
            const i32 tmp_z = lz * shared_shape[1] * shared_shape[2];
            const bool is_in_z = i_z >= 0 && i_z < shape[0];
            for (i32 ly = tid[0], gy = gid[2]; ly < shared_shape[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                const i32 i_y = gy - halo[1];
                const i32 tmp = tmp_z + ly * shared_shape[2];
                const bool is_in_y = i_y >= 0 && i_y < shape[1];
                for (i32 lx = tid[1], gx = gid[3]; lx < shared_shape[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                    const i32 i_x = gx - halo[2];
                    const bool is_in_x = i_x >= 0 && i_x < shape[2];
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ? input_3d(i_z, i_y, i_x) : T{0};
                }
            }
        }
        noa::cuda::utils::block_synchronize();

        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result{0};
            for (i32 z = 0; z < filter_length[0]; ++z)
                for (i32 y = 0; y < filter_length[1]; ++y)
                    for (i32 x = 0; x < filter_length[2]; ++x)
                        result += shared[(z * shared_shape[1] + tid[0] + y) * shared_shape[2] + tid[1] + x] *
                                  window[(z * filter_length[1] + y) * filter_length[2] + x];
            output(gid) = result;
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve_3d(const T* input, const Strides4<i64>& input_strides,
                     T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                     const U* filter, const Shape3<i64>& filter_shape, Stream& stream) {
        NOA_ASSERT(input != output && noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(filter_shape.elements() * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(noa::all(filter_shape % 2 == 1));

        if (noa::all(filter_shape <= 1))
            return noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);

        // Copy to constant memory.
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(
                cfilter, filter, filter_shape.elements() * sizeof(T),
                0, cudaMemcpyDefault, stream.get()));

        const auto shape_3d = shape.pop_front().as_safe<i32>();
        const u32 blocks_x = noa::math::divide_up(static_cast<u32>(shape_3d[2]), BLOCK_SIZE.x);
        const u32 blocks_y = noa::math::divide_up(static_cast<u32>(shape_3d[1]), BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (noa::all(filter_shape == 5)) {
            stream.enqueue("convolve_3d", convolve_3d_square_<T, 5>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, shape_3d, blocks_x);
        } else if (noa::all(filter_shape == 3)) {
            stream.enqueue("convolve_3d", convolve_3d_square_<T, 3>, {blocks, BLOCK_SIZE},
                           input_accessor, output_accessor, shape_3d, blocks_x);
        } else {
            const u32 shared_bytes = (BLOCK_SIZE.x + filter_shape[2] - 1) *
                                     (BLOCK_SIZE.y + filter_shape[1] - 1) *
                                     filter_shape[0] * sizeof(T);
            stream.enqueue("convolve_3d", convolve_3d_<T>, {blocks, BLOCK_SIZE, shared_bytes},
                           input_accessor, output_accessor, shape_3d,
                           filter_shape.as_safe<i32>(), blocks_x);
        }
    }

    #define NOA_INSTANTIATE_CONV3_(T) \
    template void convolve_3d<T,T,void>(    \
        const T*, const Strides4<i64>&,     \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&,                 \
        const T*, const Shape3<i64>&, Stream&)

    NOA_INSTANTIATE_CONV3_(f16);
    NOA_INSTANTIATE_CONV3_(f32);
    NOA_INSTANTIATE_CONV3_(f64);
}
