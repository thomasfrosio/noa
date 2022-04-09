#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/filter/Convolve.h"
#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int MAX_FILTER_SIZE = 17;
    __constant__ char cfilter[MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(double)];

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve2_(const T* __restrict__ input, uint4_t input_stride,
                    T* __restrict__ output, uint4_t output_stride,
                    uint2_t shape, int2_t filter_size, uint blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid(threadIdx.y, threadIdx.x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]);
        input += indexing::at(gid[0], gid[1], input_stride);

        const int OFFSET = static_cast<int>(BLOCK_SIZE.x); // block is 16x16 square
        const int2_t PADDING(filter_size - 1);
        const int2_t HALO(PADDING / 2);
        const int2_t SHARED_LEN(OFFSET + PADDING);
        T* shared = util::block::dynamicSharedResource<T>();

        // Load to shared memory. Loop to take into account padding.
        for (int ly = tid[0], gy = gid[2]; ly < SHARED_LEN[0]; ly += OFFSET, gy += OFFSET) {
            int i_y = gy - HALO[0];
            bool is_in_y = i_y >= 0 && i_y < shape[0];
            for (int lx = tid[1], gx = gid[3]; lx < SHARED_LEN[1]; lx += OFFSET, gx += OFFSET) {
                int i_x = gx - HALO[1];
                bool is_in_x = i_x >= 0 && i_x < shape[1];
                shared[ly * SHARED_LEN[1] + lx] = is_in_y && is_in_x ?
                                                  input[i_y * input_stride[2] + i_x * input_stride[3]] :
                                                  static_cast<T>(0);
            }
        }
        util::block::synchronize();

        if (gid[2] < shape[0] && gid[3] < shape[1]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int y = 0; y < filter_size[0]; ++y)
                for (int x = 0; x < filter_size[1]; ++x)
                    result += shared[(tid[0] + y) * SHARED_LEN[1] + tid[1] + x] * window[y * filter_size[1] + x];
            output[indexing::at(gid, output_stride)] = result;
        }
    }
}

namespace noa::cuda::filter {
    template<typename T, typename U>
    void convolve2(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                   const shared_t<U[]>& filter, size2_t filter_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);

        if (all(filter_shape <= 1))
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter.get(), math::prod(filter_shape) * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint2_t uint_shape(shape.get() + 2);
        const uint blocks_x = math::divideUp(uint_shape[1], BLOCK_SIZE.x);
        const uint blocks_y = math::divideUp(uint_shape[0], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const uint shared_bytes = (BLOCK_SIZE.x + filter_shape[1] - 1) *
                                  (BLOCK_SIZE.y + filter_shape[0] - 1) * sizeof(T);
        const LaunchConfig config{blocks, BLOCK_SIZE, shared_bytes};
        stream.enqueue("filter::convolve2", convolve2_<T>, config,
                       input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                       uint_shape, int2_t{filter_shape}, blocks_x);
        stream.attach(input, output, filter);
    }

    #define NOA_INSTANTIATE_CONV2_(T) \
    template void convolve2<T,T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size2_t, Stream&)

    NOA_INSTANTIATE_CONV2_(half_t);
    NOA_INSTANTIATE_CONV2_(float);
    NOA_INSTANTIATE_CONV2_(double);
}
