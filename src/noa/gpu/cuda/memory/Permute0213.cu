#include "noa/core/math/Generic.hpp"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Permute.h"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;

    constexpr u32 TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // Transpose 0213 is a specific case: the innermost dimension is unchanged,
    // which makes everything much simpler. Only the last two dimensions are swapped:
    //  - input_strides[1]->output_strides[2]
    //  - input_strides[2]->output_strides[1]
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0213_(AccessorRestrict<const T, 4, u32> input,
                       AccessorRestrict<T, 4, u32> output_swapped,
                       Shape2<u32> shape_yx, u32 blocks_x) {
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> gid = TILE_DIM * index + tid;
        if (!IS_MULTIPLE_OF_TILE && gid[1] >= shape_yx[1])
            return;

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_ = output_swapped[blockIdx.z][blockIdx.y];

        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gy = gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || gy < shape_yx[0])
                output_(gy, gid[1]) = input_(gy, gid[1]);
        }
    }

    // In-place.
    // This is simply swapping the Y with the X, such as swap(o[z][y][x], o[y][z][x]).
    // Only process one triangle, plus the diagonal. The other blocks are idle...
    // The shared memory simply acts as a per thread buffer.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0213_inplace_(Accessor<T, 4, u32> output, Shape2<u32> shape, u32 blocks_x) {
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[BLOCK_SIZE.y][BLOCK_SIZE.x];
        T(& tile)[BLOCK_SIZE.y][BLOCK_SIZE.x] = *reinterpret_cast<T(*)[BLOCK_SIZE.y][BLOCK_SIZE.x]>(&buffer);

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            TILE_DIM * index[0] + tid[0],
                            TILE_DIM * index[1] + tid[1]};
        if (gid[3] >= shape[1])
            return;

        const auto output_ = output[gid[0]];
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gy = gid[2] + repeat;
            if (gid[1] > gy) // process one triangle + diagonal
                continue;

            if (IS_MULTIPLE_OF_TILE || gy < shape[0]) {
                T& src = output_(gid[1], gy, gid[3]);
                T& dst = output_(gy, gid[1], gid[3]); // permutation 1 <-> 2
                tile[tid[0]][tid[1]] = dst;
                dst = src;
                src = tile[tid[0]][tid[1]];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute_0213(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile = all((shape_2d % TILE_DIM) == 0);

        const u32 blocks_x = noa::math::divide_up(shape_2d[1], TILE_DIM);
        const u32 blocks_y = noa::math::divide_up(shape_2d[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, u_shape[1], u_shape[0]);

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());
        const auto swapped_output = output_accessor.swap_dimensions(1, 2);

        if (are_multiple_tile) {
            stream.enqueue("permute0213", permute_0213_<T, true>, {blocks, BLOCK_SIZE},
                           input_accessor, swapped_output, shape_2d, blocks_x);
        } else {
            stream.enqueue("permute0213", permute_0213_<T, false>, {blocks, BLOCK_SIZE},
                           input_accessor, swapped_output, shape_2d, blocks_x);
        }
    }

    template<typename T>
    void permute_0213_inplace(T* output, const Strides4<i64>& output_strides,
                              const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if (shape[1] != shape[2])
            NOA_THROW("For a \"0213\" in-place permutation, shape[1] should be equal to shape[2]. Got {}", shape);

        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(2, 3);
        const bool are_multiple_tile = (shape_2d[0] % TILE_DIM) == 0;

        const u32 blocks_x = noa::math::divide_up(shape_2d[1], TILE_DIM);
        const u32 blocks_y = noa::math::divide_up(shape_2d[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_y, u_shape[1], u_shape[0]);
        const auto accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());

        if (are_multiple_tile) {
            stream.enqueue("permute_0213_inplace",
                           permute_0213_inplace_<T, true>, {blocks, BLOCK_SIZE},
                           accessor, shape_2d, blocks_x);
        } else {
            stream.enqueue("permute_0213_inplace",
                           permute_0213_inplace_<T, false>, {blocks, BLOCK_SIZE},
                           accessor, shape_2d, blocks_x);
        }
    }

    #define NOA_INSTANTIATE_TRANSPOSE_(T)   \
    template void permute_0213<T>(          \
        const T*, const Strides4<i64>&,     \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&, Stream&);       \
    template void permute_0213_inplace<T>(  \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_TRANSPOSE_(bool);
    NOA_INSTANTIATE_TRANSPOSE_(i8);
    NOA_INSTANTIATE_TRANSPOSE_(i16);
    NOA_INSTANTIATE_TRANSPOSE_(i32);
    NOA_INSTANTIATE_TRANSPOSE_(i64);
    NOA_INSTANTIATE_TRANSPOSE_(u8);
    NOA_INSTANTIATE_TRANSPOSE_(u16);
    NOA_INSTANTIATE_TRANSPOSE_(u32);
    NOA_INSTANTIATE_TRANSPOSE_(u64);
    NOA_INSTANTIATE_TRANSPOSE_(f16);
    NOA_INSTANTIATE_TRANSPOSE_(f32);
    NOA_INSTANTIATE_TRANSPOSE_(f64);
    NOA_INSTANTIATE_TRANSPOSE_(c16);
    NOA_INSTANTIATE_TRANSPOSE_(c32);
    NOA_INSTANTIATE_TRANSPOSE_(c64);
}
