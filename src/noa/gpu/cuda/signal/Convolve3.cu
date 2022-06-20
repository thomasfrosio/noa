#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/signal/Convolve.h"
#include "noa/gpu/cuda/util/Block.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr dim3 BLOCK_SIZE(16, 16);
    constexpr int MAX_FILTER_SIZE = 5;
    constexpr int MAX_FILTER_BYTES = MAX_FILTER_SIZE * MAX_FILTER_SIZE * MAX_FILTER_SIZE * sizeof(double);
    __constant__ char cfilter[MAX_FILTER_BYTES];

    template<typename T, int FILTER_LEN>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve3Square_(const T* __restrict__ input, uint4_t input_stride,
                          T* __restrict__ output, uint4_t output_stride,
                          uint3_t shape, uint blocks_x) {
        static_assert(FILTER_LEN % 2); // only support odd windows.
        constexpr int PADDING = FILTER_LEN - 1; // assume odd
        constexpr int HALO = FILTER_LEN / 2;
        constexpr int3_t SHARED_LEN(FILTER_LEN, BLOCK_SIZE.y + PADDING, BLOCK_SIZE.x + PADDING);

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid(threadIdx.y, threadIdx.x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]);

        input += gid[0] * input_stride[0];

        using uninit_t = util::traits::uninitialized_type_t<T>;
        __shared__ uninit_t buffer[math::prod(SHARED_LEN)];
        T* shared = reinterpret_cast<T*>(buffer);

        // Load shared memory. Loop to take into account padding.
        for (int lz = 0, gz = gid[1]; lz < SHARED_LEN[0]; ++lz, ++gz) {
            int i_z = gz - HALO;
            int tmp_z = lz * SHARED_LEN[1] * SHARED_LEN[2];
            bool is_in_z = i_z >= 0 && i_z < shape[0];
            for (int ly = tid[0], gy = gid[2]; ly < SHARED_LEN[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                int i_y = gy - HALO;
                int tmp = tmp_z + ly * SHARED_LEN[2];
                bool is_in_y = i_y >= 0 && i_y < shape[1];
                for (int lx = tid[1], gx = gid[3]; lx < SHARED_LEN[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                    int i_x = gx - HALO;
                    bool is_in_x = i_x >= 0 && i_x < shape[2];
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ?
                                       input[i_z * input_stride[1] + i_y * input_stride[2] + i_x * input_stride[3]] :
                                       static_cast<T>(0);
                }
            }
        }
        util::block::synchronize();

        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int z = 0; z < FILTER_LEN; ++z)
                for (int y = 0; y < FILTER_LEN; ++y)
                    for (int x = 0; x < FILTER_LEN; ++x)
                        result += shared[(z * SHARED_LEN[1] + tid[0] + y) * SHARED_LEN[2] + tid[1] + x] *
                                  window[(z * FILTER_LEN + y) * FILTER_LEN + x];
            output[indexing::at(gid, output_stride)] = result;
        }
    }

    // Version with filter_length not fixed at compile time.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void convolve3_(const T* __restrict__ input, uint4_t input_stride,
                    T* __restrict__ output, uint4_t output_stride,
                    uint3_t shape, int3_t filter_length, uint blocks_x) {
        const int3_t padding(filter_length - 1); // assume odd
        const int3_t halo = padding / 2;
        const int3_t shared_len(filter_length[0], BLOCK_SIZE.y + padding[1], BLOCK_SIZE.x + padding[2]);

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int2_t tid(threadIdx.y, threadIdx.x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_SIZE.y * index[0] + tid[0],
                         BLOCK_SIZE.x * index[1] + tid[1]);

        input += gid[0] * input_stride[0];

        // Load shared memory. Loop to take into account padding.
        T* shared = util::block::dynamicSharedResource<T>();
        for (int lz = 0, gz = gid[1]; lz < shared_len[0]; ++lz, ++gz) {
            int i_z = gz - halo[0];
            int tmp_z = lz * shared_len[1] * shared_len[2];
            bool is_in_z = i_z >= 0 && i_z < shape[0];
            for (int ly = tid[0], gy = gid[2]; ly < shared_len[1]; ly += BLOCK_SIZE.y, gy += BLOCK_SIZE.y) {
                int i_y = gy - halo[1];
                int tmp = tmp_z + ly * shared_len[2];
                bool is_in_y = i_y >= 0 && i_y < shape[1];
                for (int lx = tid[1], gx = gid[3]; lx < shared_len[2]; lx += BLOCK_SIZE.x, gx += BLOCK_SIZE.x) {
                    int i_x = gx - halo[2];
                    bool is_in_x = i_x >= 0 && i_x < shape[2];
                    shared[tmp + lx] = (is_in_z && is_in_y && is_in_x) ?
                                       input[i_z * input_stride[1] + i_y * input_stride[2] + i_x * input_stride[3]] :
                                       static_cast<T>(0);
                }
            }
        }
        util::block::synchronize();

        if (gid[2] < shape[1] && gid[3] < shape[2]) {
            // Weighted sum.
            const T* window = reinterpret_cast<T*>(cfilter);
            T result = static_cast<T>(0);
            for (int z = 0; z < filter_length[0]; ++z)
                for (int y = 0; y < filter_length[1]; ++y)
                    for (int x = 0; x < filter_length[2]; ++x)
                        result += shared[(z * shared_len[1] + tid[0] + y) * shared_len[2] + tid[1] + x] *
                                  window[(z * filter_length[1] + y) * filter_length[2] + x];
            output[indexing::at(gid, output_stride)] = result;
        }
    }
}

namespace noa::cuda::signal {
    template<typename T, typename U, typename>
    void convolve3(const shared_t<T[]>& input, size4_t input_stride,
                   const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                   const shared_t<U[]>& filter, size3_t filter_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(filter_shape.elements() * sizeof(T) <= MAX_FILTER_BYTES);
        NOA_ASSERT(all(filter_shape % 2 == 1));

        if (all(filter_shape <= 1))
            return memory::copy(input, input_stride, output, output_stride, shape, stream);

        // Copy to constant memory.
        NOA_THROW_IF(cudaMemcpyToSymbolAsync(cfilter, filter.get(), math::prod(filter_shape) * sizeof(T),
                                             0, cudaMemcpyDefault, stream.get()));

        const uint3_t uint_shape(shape.get(1));
        const uint blocks_x = math::divideUp(uint_shape[2], BLOCK_SIZE.x);
        const uint blocks_y = math::divideUp(uint_shape[1], BLOCK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        if (all(filter_shape == 5)) {
            stream.enqueue("filter::convolve3", convolve3Square_<T, 5>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        } else if (all(filter_shape == 3)) {
            stream.enqueue("filter::convolve3", convolve3Square_<T, 3>, {blocks, BLOCK_SIZE},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, blocks_x);
        } else {
            const uint shared_bytes = (BLOCK_SIZE.x + filter_shape[2] - 1) *
                                      (BLOCK_SIZE.y + filter_shape[1] - 1) *
                                      filter_shape[0] * sizeof(T);
            stream.enqueue("filter::convolve3", convolve3_<T>, {blocks, BLOCK_SIZE, shared_bytes},
                           input.get(), uint4_t{input_stride}, output.get(), uint4_t{output_stride},
                           uint_shape, int3_t{filter_shape}, blocks_x);
        }
        stream.attach(input, output, filter);
    }

    #define NOA_INSTANTIATE_CONV3_(T) \
    template void convolve3<T,T,void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size3_t, Stream&)

    NOA_INSTANTIATE_CONV3_(half_t);
    NOA_INSTANTIATE_CONV3_(float);
    NOA_INSTANTIATE_CONV3_(double);
}
