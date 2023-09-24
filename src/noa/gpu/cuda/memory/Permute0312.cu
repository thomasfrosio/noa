#include "noa/core/math/Generic.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/memory/Permute.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;

    // Transpose XY plane (by chunk of 32x32 tiles) for every Z.
    constexpr u32 TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XY tile along Z becomes X'Z' (X'=Y, Z'=X) along Y' (Y'=Z)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute_0312_(
            AccessorRestrict<const T, 4, u32> input,
            AccessorRestrict<T, 4, u32> output_swapped,
            Shape2<u32> shape_yx /* YX */ , u32 blocks_x
    ) {
        using uninit_t = noa::cuda::utils::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        const auto input_ = input[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        const Vec2<u32> tid{threadIdx.y, threadIdx.x};
        const Vec2<u32> index = noa::indexing::offset2index(blockIdx.x, blocks_x);
        const Vec2<u32> offset = TILE_DIM * index;

        // Read tile to shared memory.
        const auto old_gid = offset + tid;
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gy = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape_yx[1] && gy < shape_yx[0]))
                tile[tid[0] + repeat][tid[1]] = input_(gy, old_gid[1]);
        }

        noa::cuda::utils::block_synchronize();

        // Write permuted tile to global memory.
        const auto new_gid = offset.flip() + tid;
        for (u32 repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            const u32 gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape_yx[0] && gz < shape_yx[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // Since all axes are permuted, in-place permute cannot easily be expressed as a 2D transposition
    // along a common plane. https://www.aldapa.eus/res/cuTranspose/Readme.html has an implementation
    // based on a 3D shared memory array, but since it is unlikely to be used anyway, don't bother for now.
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute_0312(
            const T* input, const Strides4<i64>& input_strides,
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream
    ) {
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
            stream.enqueue("memory::permute0312", permute_0312_<T, true>, {blocks, BLOCK_SIZE},
                           input_accessor, swapped_output, shape_2d, blocks_x);
        } else {
            stream.enqueue("memory::permute0312", permute_0312_<T, false>, {blocks, BLOCK_SIZE},
                           input_accessor, swapped_output, shape_2d, blocks_x);
        }
    }

    #define NOA_INSTANTIATE_TRANSPOSE_(T)   \
    template void permute_0312<T>(          \
        const T*, const Strides4<i64>&,     \
        T*, const Strides4<i64>&,           \
        const Shape4<i64>&, Stream&)

//    NOA_INSTANTIATE_TRANSPOSE_(bool);
//    NOA_INSTANTIATE_TRANSPOSE_(i8);
//    NOA_INSTANTIATE_TRANSPOSE_(i16);
    NOA_INSTANTIATE_TRANSPOSE_(i32);
    NOA_INSTANTIATE_TRANSPOSE_(i64);
//    NOA_INSTANTIATE_TRANSPOSE_(u8);
//    NOA_INSTANTIATE_TRANSPOSE_(u16);
    NOA_INSTANTIATE_TRANSPOSE_(u32);
    NOA_INSTANTIATE_TRANSPOSE_(u64);
//    NOA_INSTANTIATE_TRANSPOSE_(f16);
    NOA_INSTANTIATE_TRANSPOSE_(f32);
    NOA_INSTANTIATE_TRANSPOSE_(f64);
//    NOA_INSTANTIATE_TRANSPOSE_(c16);
    NOA_INSTANTIATE_TRANSPOSE_(c32);
    NOA_INSTANTIATE_TRANSPOSE_(c64);
}
