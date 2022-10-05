#include "noa/common/Math.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/memory/Permute.h"

#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    // Transpose XZ plane (by chunk of 32x32 tiles) for every Y.
    constexpr uint32_t TILE_DIM = 32;
    constexpr dim3 BLOCK_SIZE(TILE_DIM, 256 / TILE_DIM);

    // Out-of-place.
    // The XZ tile along Y becomes X'Z' (X'=Z, Z'=X) along Y' (Y'=Y)
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0321_(AccessorRestrict<const T, 4, uint32_t> input_swapped,
                      AccessorRestrict<T, 4, uint32_t> output_swapped,
                      uint2_t shape /* ZX */, uint32_t blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[TILE_DIM][TILE_DIM + 1];
        T(& tile)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer);

        const auto input_swapped_ = input_swapped[blockIdx.z][blockIdx.y];
        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        // Read tile to shared memory.
        const uint2_t old_gid = offset + tid;
        for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            uint32_t gz = old_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (old_gid[1] < shape[1] && gz < shape[0]))
                tile[tid[0] + repeat][tid[1]] = input_swapped_(gz, old_gid[1]);
        }

        util::block::synchronize();

        // Write permuted tile to global memory.
        const uint2_t new_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'
        for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
            uint32_t gz = new_gid[0] + repeat;
            if (IS_MULTIPLE_OF_TILE || (new_gid[1] < shape[0] && gz < shape[1]))
                output_swapped_(gz, new_gid[1]) = tile[tid[1]][tid[0] + repeat];
        }
    }

    // In-place.
    template<typename T, bool IS_MULTIPLE_OF_TILE>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void permute0321_inplace_(Accessor<T, 4, uint32_t> output_swapped, uint32_t shape, uint32_t blocks_x) {
        using uninit_t = cuda::util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer_src[TILE_DIM][TILE_DIM + 1];
        __shared__ uninit_t buffer_dst[TILE_DIM][TILE_DIM + 1];
        T(& tile_src)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_src);
        T(& tile_dst)[TILE_DIM][TILE_DIM + 1] = *reinterpret_cast<T(*)[TILE_DIM][TILE_DIM + 1]>(&buffer_dst);

        const auto output_swapped_ = output_swapped[blockIdx.z][blockIdx.y];

        // Get the current indexes.
        const uint2_t tid{threadIdx.y, threadIdx.x};
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint2_t offset = TILE_DIM * index; // ZX

        if (offset[0] > offset[1]) { // lower t
            const uint2_t src_gid = offset + tid; // ZX
            const uint2_t dst_gid = offset.flip() + tid; // ZX.flip() -> XZ -> Z'X'

            // Read tiles to shared memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(sz, src_gid[1]);

                const uint32_t dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    tile_dst[tid[0] + repeat][tid[1]] = output_swapped_(dz, dst_gid[1]);
            }

            util::block::synchronize();

            // Write permuted tiles to global memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t dz = dst_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (dst_gid[1] < shape && dz < shape))
                    output_swapped_(dz, dst_gid[1]) = tile_src[tid[1]][tid[0] + repeat];

                const uint32_t sz = src_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (src_gid[1] < shape && sz < shape))
                    output_swapped_(sz, src_gid[1]) = tile_dst[tid[1]][tid[0] + repeat];
            }

        } else if (offset[0] == offset[1]) { // diagonal
            const uint2_t gid = offset + tid; // ZX

            // Read tile to shared memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    tile_src[tid[0] + repeat][tid[1]] = output_swapped_(gz, gid[1]);
            }

            util::block::synchronize();

            // Write permuted tile to global memory.
            for (uint32_t repeat = 0; repeat < TILE_DIM; repeat += BLOCK_SIZE.y) {
                const uint32_t gz = gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE || (gid[1] < shape && gz < shape))
                    output_swapped_(gz, gid[1]) = tile_src[tid[1]][tid[0] + repeat];
            }
        }
    }
}

namespace noa::cuda::memory::details {
    template<typename T>
    void permute0321(const shared_t<T[]>& input, dim4_t input_strides,
                     const shared_t<T[]>& output, dim4_t output_strides,
                     dim4_t shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const auto uint_shape = safe_cast<uint2_t>(dim2_t{shape[1], shape[3]});
        const bool are_multiple_tile = all((uint_shape % TILE_DIM) == 0);

        const uint32_t blocks_x = math::divideUp(uint_shape[1], TILE_DIM);
        const uint32_t blocks_z = math::divideUp(uint_shape[0], TILE_DIM);
        const dim3 blocks(blocks_x * blocks_z, shape[2], shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));
        const auto swapped_input = input_accessor.swap(1, 2);
        const auto swapped_output = output_accessor.swap(1, 2);

        if (are_multiple_tile) {
            stream.enqueue("memory::permute0321", permute0321_<T, true>, {blocks, BLOCK_SIZE},
                           swapped_input, swapped_output, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::permute0321", permute0321_<T, false>, {blocks, BLOCK_SIZE},
                           swapped_input, swapped_output, uint_shape, blocks_x);
        }
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void permute0321(const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if (shape[1] != shape[3])
            NOA_THROW("For a \"0321\" in-place permutation, shape[1] should be equal to shape[3]. Got {}", shape);

        const auto uint_shape = safe_cast<uint32_t>(shape[1]);
        const bool is_multiple_tile = (uint_shape % TILE_DIM) == 0;

        const uint32_t blocks_x = math::divideUp(uint_shape, TILE_DIM); // blocks_z == blocks_x
        const dim3 blocks(blocks_x * blocks_x, shape[2], shape[0]);
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));
        const auto swapped_output = output_accessor.swap(1, 2);

        if (is_multiple_tile) {
            stream.enqueue("memory::permute0321_inplace", permute0321_inplace_<T, true>, {blocks, BLOCK_SIZE},
                           swapped_output, uint_shape, blocks_x);
        } else {
            stream.enqueue("memory::permute0321_inplace", permute0321_inplace_<T, false>, {blocks, BLOCK_SIZE},
                           swapped_output, uint_shape, blocks_x);
        }
        stream.attach(output);
    }
}

#define NOA_INSTANTIATE_TRANSPOSE_(T)                                                                                                   \
template void noa::cuda::memory::details::permute0321<T>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
template void noa::cuda::memory::details::inplace::permute0321<T>(const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

NOA_INSTANTIATE_TRANSPOSE_(bool);
NOA_INSTANTIATE_TRANSPOSE_(int8_t);
NOA_INSTANTIATE_TRANSPOSE_(int16_t);
NOA_INSTANTIATE_TRANSPOSE_(int32_t);
NOA_INSTANTIATE_TRANSPOSE_(int64_t);
NOA_INSTANTIATE_TRANSPOSE_(uint8_t);
NOA_INSTANTIATE_TRANSPOSE_(uint16_t);
NOA_INSTANTIATE_TRANSPOSE_(uint32_t);
NOA_INSTANTIATE_TRANSPOSE_(uint64_t);
NOA_INSTANTIATE_TRANSPOSE_(half_t);
NOA_INSTANTIATE_TRANSPOSE_(float);
NOA_INSTANTIATE_TRANSPOSE_(double);
NOA_INSTANTIATE_TRANSPOSE_(chalf_t);
NOA_INSTANTIATE_TRANSPOSE_(cfloat_t);
NOA_INSTANTIATE_TRANSPOSE_(cdouble_t);
