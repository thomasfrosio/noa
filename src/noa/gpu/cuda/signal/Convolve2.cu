#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/signal/Convolve.h"
#include "noa/gpu/cuda/utils/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int32_t MAX_FILTER_SIZE = 11;
    constexpr int32_t MAX_FILTER_BYTES = MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(double);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve2_(AccessorRestrict<const T, 4, uint32_t> input,
                    AccessorRestrict<T, 4, uint32_t> output,
                    uint2_t shape, int2_t filter_size, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};
        const auto input_ = input[gid[0]][gid[1]];

        const auto OFFSET = static_cast<int32_t>(BLOCK_SIZE.x); // block is 16x16 square
        const int2_t PADDING(filter_size - 1);
        const int2_t HALO(PADDING / 2);
        const int2_t SHARED_LEN(OFFSET + PADDING);
        T* shared = utils::block::dynamicSharedResource<T>();

        // Load to shared memory. Loop to take into account padding.
        for (int32_t ly = tid[0], gy = gid[2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            const int32_t i_y = gy - HALO[0];
            const bool is_in_y = i_y >= 0 && i_y < shape[0];
            for (int32_t lx = tid[1], gx = gid[3]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                const int32_t i_x = gx - HALO[1];
                const bool is_in_x = i_x >= 0 && i_x < shape[1];
                shared[ly * SHARED_LEN[1] + lx] = is_in_y && is_in_x ? input_(i_y, i_x) : T{0};
            }
        }
        utils::block::synchronize();

        if (gid[2] < shape[0] && gid[3] < shape[1]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result{0};
            for (int32_t y = 0; y < filter_size[0]; ++y)
                for (int32_t x = 0; x < filter_size[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * window[y * filter_size[1] + x];
            output(gid) = result;
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve2(const shared_t<T[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                   const shared_t<U[]>& filter, dim2_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(filter_shape.elements() * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(all(filter_shape % 2 == 1));

        if (all(filter_shape <= 1))
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter.get(), math::prod(filter_shape) * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const auto uint_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint32_t shared_bytes = (BLOCK_SIZE.x + filter_shape[1] - 1) *
                                  (BLOCK_SIZE.y + filter_shape[0] - 1) * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("filter::convolve2", convolve2_<T>, config,
                       input_accessor, output_accessor, uint_shape, int2_t(filter_shape), blocks_x);
        stream.attach(input, output, filter);
    }

    #define NOA_INSTANTIATE_CONV2_(T) \
    template void convolve2<T,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim2_t, Stream&)

    NOA_INSTANTIATE_CONV2_(half_t);
    NOA_INSTANTIATE_CONV2_(float);
    NOA_INSTANTIATE_CONV2_(double);
}
