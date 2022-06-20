#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Resize.h"

namespace {
    using namespace noa;
    constexpr uint MAX_THREADS = 512;
    constexpr uint ELEMENTS_PER_THREAD_X = 2;
    constexpr dim3 BLOCK_SIZE(32, MAX_THREADS / 32);
    constexpr dim3 BLOCK_WORK_SIZE(BLOCK_SIZE.x * ELEMENTS_PER_THREAD_X, BLOCK_SIZE.y);

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void cropH2H_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                  T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x);
        if (gid[2] >= output_shape[1])
            return;

        const uint iz = gid[1] < (output_shape[0] + 1) / 2 ? gid[1] : gid[1] + input_shape[0] - output_shape[0];
        const uint iy = gid[2] < (output_shape[1] + 1) / 2 ? gid[2] : gid[2] + input_shape[1] - output_shape[1];
        input += indexing::at(gid[0], iz, iy, input_stride);
        output += indexing::at(gid[0], gid[1], gid[2], output_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint x = gid[3] + BLOCK_SIZE.x * i;
            if (x < output_shape[2] / 2 + 1)
                output[x * output_stride[3]] = input[x * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void cropF2F_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                  T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x);
        if (gid[2] >= output_shape[1])
            return;

        const uint iz = gid[1] < (output_shape[0] + 1) / 2 ? gid[1] : gid[1] + input_shape[0] - output_shape[0];
        const uint iy = gid[2] < (output_shape[1] + 1) / 2 ? gid[2] : gid[2] + input_shape[1] - output_shape[1];
        input += indexing::at(gid[0], iz, iy, input_stride);
        output += indexing::at(gid[0], gid[1], gid[2], output_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint ox = gid[3] + BLOCK_SIZE.x * i;
            const uint ix = ox < (output_shape[2] + 1) / 2 ? ox : ox + input_shape[2] - output_shape[2];
            if (ox < output_shape[2])
                output[ox * output_stride[3]] = input[ix * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void padH2H_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                 T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x);
        if (gid[2] >= input_shape[1])
            return;

        const uint oz = gid[1] < (input_shape[0] + 1) / 2 ? gid[1] : gid[1] + output_shape[0] - input_shape[0];
        const uint oy = gid[2] < (input_shape[1] + 1) / 2 ? gid[2] : gid[2] + output_shape[1] - input_shape[1];
        input += indexing::at(gid[0], gid[1], gid[2], input_stride);
        output += indexing::at(gid[0], oz, oy, output_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint x = gid[3] + BLOCK_SIZE.x * i;
            if (x < input_shape[2] / 2 + 1)
                output[x * output_stride[3]] = input[x * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void padF2F_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                 T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x);
        if (gid[2] >= input_shape[1])
            return;

        const uint oz = gid[1] < (input_shape[0] + 1) / 2 ? gid[1] : gid[1] + output_shape[0] - input_shape[0];
        const uint oy = gid[2] < (input_shape[1] + 1) / 2 ? gid[2] : gid[2] + output_shape[1] - input_shape[1];
        input += indexing::at(gid[0], gid[1], gid[2], input_stride);
        output += indexing::at(gid[0], oz, oy, output_stride);

        for (int i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint ix = gid[3] + BLOCK_SIZE.x * i;
            if (ix < input_shape[2]) {
                const uint ox = ix < (input_shape[2] + 1) / 2 ? ix : ix + output_shape[2] - input_shape[2];
                output[ox * output_stride[3]] = input[ix * input_stride[3]];
            }
        }
    }
}

namespace noa::cuda::fft::details {
    template<typename T>
    void cropH2H(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape.fft(), stream);

        const uint3_t old_shape(input_shape.get(1));
        const uint3_t new_shape(output_shape.get(1));
        const uint blocks_x = math::divideUp(new_shape[2] / 2 + 1, BLOCK_WORK_SIZE.x);
        const uint blocks_y = math::divideUp(new_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);
        stream.enqueue("cropH2H_", cropH2H_<T>, {blocks, BLOCK_SIZE},
                       input.get(), uint4_t{input_stride}, old_shape,
                       output.get(), uint4_t{output_stride}, new_shape, blocks_x);
        stream.attach(input, output);
    }

    template<typename T>
    void cropF2F(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape, stream);

        const uint3_t old_shape(input_shape.get(1));
        const uint3_t new_shape(output_shape.get(1));
        const uint blocks_x = math::divideUp(new_shape[2], BLOCK_WORK_SIZE.x);
        const uint blocks_y = math::divideUp(new_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);
        stream.enqueue("cropF2F_", cropF2F_<T>, {blocks, BLOCK_SIZE},
                       input.get(), uint4_t{input_stride}, old_shape,
                       output.get(), uint4_t{output_stride}, new_shape, blocks_x);
        stream.attach(input, output);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padH2H(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape.fft(), stream);

        memory::set(output, output_stride, output_shape.fft(), T{0}, stream);
        const uint3_t old_shape(input_shape.get(1));
        const uint3_t new_shape(output_shape.get(1));
        const uint blocks_x = math::divideUp(old_shape[2] / 2 + 1, BLOCK_WORK_SIZE.x);
        const uint blocks_y = math::divideUp(old_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);
        stream.enqueue("padH2H_", padH2H_<T>, {blocks, BLOCK_SIZE},
                       input.get(), uint4_t{input_stride}, old_shape,
                       output.get(), uint4_t{output_stride}, new_shape, blocks_x);
        stream.attach(input, output);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padF2F(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape, stream);

        memory::set(output, output_stride, output_shape, T{0}, stream);
        const uint3_t old_shape(input_shape.get(1));
        const uint3_t new_shape(output_shape.get(1));
        const uint blocks_x = math::divideUp(old_shape[2], BLOCK_WORK_SIZE.x);
        const uint blocks_y = math::divideUp(old_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);
        stream.enqueue("padF2F_", padF2F_<T>, {blocks, BLOCK_SIZE},
                       input.get(), uint4_t{input_stride}, old_shape,
                       output.get(), uint4_t{output_stride}, new_shape, blocks_x);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_CROP_(T)                                                                                    \
    template void cropH2H<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void cropF2F<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);  \
    template void padH2H<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
    template void padF2F<T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_CROP_(half_t);
    NOA_INSTANTIATE_CROP_(float);
    NOA_INSTANTIATE_CROP_(double);
    NOA_INSTANTIATE_CROP_(chalf_t);
    NOA_INSTANTIATE_CROP_(cfloat_t);
    NOA_INSTANTIATE_CROP_(cdouble_t);
}
