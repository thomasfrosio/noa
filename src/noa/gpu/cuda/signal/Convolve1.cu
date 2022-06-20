#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/signal/Convolve.h"
#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int MAX_FILTER_SIZE = 129;
    constexpr int MAX_FILTER_BYTES = MAX_FILTER_SIZE * sizeof(double);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve1_(const T* __restrict__ input, uint4_t input_stride,
                    T* __restrict__ output, uint4_t output_stride,
                    uint2_t shape, int filter_size, uint blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid(threadIdx.y, threadIdx.x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]);

        input += indexing::at(gid[0], gid[1], gid[2], input_stride);

        T* shared = util::block::dynamicSharedResource<T>();
        if (gid[2] < shape[0]) {
            const int PADDING = filter_size - 1;
            const int HALO = PADDING / 2;
            const int SHARED_LEN = static_cast<int>(BLOCK_SIZE.x) + PADDING;
            shared += tid[0] * SHARED_LEN; // focus on current row

            // Load shared memory. Loop to take into account padding.
            for (int lx = tid[1], gx = gid[3]; lx < SHARED_LEN; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                int idx = gx - HALO;
                shared[lx] = (idx >= 0 && idx < shape[1]) ? input[idx * input_stride[3]] : static_cast<T>(0);
            }
            util::block::synchronize();

            if (gid[3] < shape[1]) {
                // Weighted sum.
                const T* window = reinterpret_cast<T*>(cfilter);
                T result = static_cast<T>(0);
                for (int idx = 0; idx < filter_size; ++idx)
                    result += shared[tid[1] + idx] * window[idx];
                output[indexing::at(gid, output_stride)] = result;
            }
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve1(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                   const shared_t<U[]>& filter, size_t filter_size, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(filter_size * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(filter_size % 2);

        if (filter_size <= 1)
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter.get(), filter_size * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint2_t uint_shape{shape.get(2)};
        const uint blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint shared_bytes = (BLOCK_SIZE.x + filter_size - 1) * BLOCK_SIZE.y * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};
        stream.enqueue("filter::convolve1", convolve1_<T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                       uint_shape, filter_size, blocks_x);
        stream.attach(input, output, filter);
    }

    #define NOA_INSTANTIATE_CONV1_(T) \
    template void convolve1<T,T,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size_t, Stream&)

    NOA_INSTANTIATE_CONV1_(half_t);
    NOA_INSTANTIATE_CONV1_(float);
    NOA_INSTANTIATE_CONV1_(double);
}
