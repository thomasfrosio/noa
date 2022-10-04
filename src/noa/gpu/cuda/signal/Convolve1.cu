#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/signal/Convolve.h"
#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int32_t MAX_FILTER_SIZE = 129;
    constexpr int32_t MAX_FILTER_BYTES = MAX_FILTER_SIZE * sizeof(double);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve1_(AccessorRestrict<const T, 4, uint32_t> input,
                    AccessorRestrict<T, 4, uint32_t> output,
                    uint2_t shape, int32_t filter_size, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid{threadIdx.y, threadIdx.x};
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]};

        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        T* shared = util::block::dynamicSharedResource<T>();
        if (gid[2] < shape[0]) {
            const int32_t PADDING = filter_size - 1;
            const int32_t HALO = PADDING / 2;
            const auto SHARED_LEN = static_cast<int32_t>(BLOCK_SIZE.x) + PADDING;
            shared += tid[0] * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (int32_t lx = tid[1], gx = gid[3]; lx < SHARED_LEN; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                const int32_t idx = gx - HALO;
                shared[lx] = (idx >= 0 && idx < shape[1]) ? input_row[idx] : T{0};
            }
            util::block::synchronize();

            if (gid[3] < shape[1]) {
                // Weighted sum.
                const T* window = reinterpret_cast<T*>(cfilter);
                T result{0};
                for (int32_t idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * window[idx];
                output(gid) = result;
            }
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve1(const shared_t<T[]>& input, size4_t input_strides,
                   const shared_t<T[]>& output, size4_t output_strides, size4_t shape,
                   const shared_t<U[]>& filter, dim_t filter_size, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(filter_size % 2);

        if (filter_size <= 1)
            return memory::copy(input, input_strides, output, output_strides, shape, stream);

        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter.get(), filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const auto uint_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint32_t shared_bytes = (BLOCK_SIZE.x + filter_size - 1) * BLOCK_SIZE.y * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("filter::convolve1", convolve1_<T>, config,
                       input_accessor, output_accessor, uint_shape, filter_size, blocks_x);
        stream.attach(input, output, filter);
    }

    #define NOA_INSTANTIATE_CONV1_(T) \
    template void convolve1<T,T,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, dim_t, Stream&)

    NOA_INSTANTIATE_CONV1_(half_t);
    NOA_INSTANTIATE_CONV1_(float);
    NOA_INSTANTIATE_CONV1_(double);
}
