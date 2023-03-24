#include "noa/core/math/Generic.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/memory/Permute.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr u32 TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0321_(AccessorRestrict<const T, 4, u32> input_swapped,
                       AccessorRestrict<T, 4, u32> output_swapped,
                       Shape2<u32> shape_zx, u32 blocks_x) {
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_DIM * index; // ZX

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            u32 gz = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape_zx[1] && gz < shape_zx[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        noa::cuda::utils::block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            u32 gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape_zx[0] && gz < shape_zx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0321_inplace_(Accessor<T, 4, u32> output_swapped, u32 shape, u32 blocks_x) {
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_DIM * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const auto src_gid = offset + tid; // ZX
            const auto dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(sz, src_gid[1]);

                const u32 dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output_swapped_(dz, dst_gid[1]);
            }

            noa::cuda::utils::block_synchronize();

            // Write permuted tiles to global memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    output_swapped_(dz, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const u32 sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    output_swapped_(sz, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const auto gid = offset + tid; // ZX

            // Read tile to shared memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(gz, gid[1]);
            }

            noa::cuda::utils::block_synchronize();

            // Write permuted tile to global memory.
            for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const u32 gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    output_swapped_(gz, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute_0321(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const auto u_shape = shape.as_safe<u32>();
        const auto shape_2d = u_shape.filter(1, 3);
        const bool are_multiple_tile = all((shape_2d % TILE_DIM) == 0);

        const u32 blocks_x = noa::math::divide_up(shape_2d[1], TILE_DIM);
        const u32 blocks_z = noa::math::divide_up(shape_2d[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_z, u_shape[2], u_shape[0]);

        const auto input_accessor = AccessorRestrict<const T, 4, u32>(input, input_strides.as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, 4, u32>(output, output_strides.as_safe<u32>());
        const auto swapped_input = input_accessor.swap_dimensions(1, 2);
        const auto swapped_output = output_accessor.swap_dimensions(1, 2);

        if (are_multiple_tile) {
            stream.enqueue("permute_0321",
                           permute_0321_<T, true>, {blocks, BLOCK_SIZE},
                           swapped_input, swapped_output, shape_2d, blocks_x);
        } else {
            stream.enqueue("permute_0321",
                           permute_0321_<T, false>, {blocks, BLOCK_SIZE},
                           swapped_input, swapped_output, shape_2d, blocks_x);
        }
    }

    template<typename T>
    void permute_0321_inplace(T* output, const Strides4<i64>& output_strides,
                              const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        if (shape[1] != shape[3])
            NOA_THROW("For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got {}", shape);

        const auto u_shape = shape.as_safe<u32>();
        const bool is_multiple_tile = (u_shape[1] % TILE_DIM) == 0;

        const u32 blocks_x = noa::math::divide_up(u_shape[1], TILE_DIM); // blocks_z == blocks_x
        const dim3 blocks(blocks_x * blocks_x, u_shape[2], u_shape[0]);
        const auto output_accessor = Accessor<T, 4, u32>(output, output_strides.as_safe<u32>());
        const auto swapped_output = output_accessor.swap_dimensions(1, 2);

        if (is_multiple_tile) {
            stream.enqueue("permute_0321_inplace",
                           permute_0321_inplace_<T, true>, {blocks, BLOCK_SIZE},
                           swapped_output, u_shape[1], blocks_x);
        } else {
            stream.enqueue("permute_0321_inplace",
                           permute_0321_inplace_<T, false>, {blocks, BLOCK_SIZE},
                           swapped_output, u_shape[1], blocks_x);
        }
    }

    #define NOA_INSTANTIATE_TRANSPOSE_(T)   \
    template void permute_0321<T>(          \
        const T*, const Strides4<i64>&,     \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&, Stream&);       \
    template void permute_0321_inplace<T>(  \
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
