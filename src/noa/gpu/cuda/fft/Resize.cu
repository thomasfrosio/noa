#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Resize.h"

namespace {
    using namespace noa;
    constexpr uint MAX_THREADS = 512;
    constexpr dim3 THREADS(32, MAX_THREADS / 32);

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cropH2H_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                  T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] >= output_shape[1])
            return;

        const uint iz = gid[0] < (output_shape[0] + 1) / 2 ? gid[0] : gid[0] + input_shape[0] - output_shape[0];
        const uint iy = gid[1] < (output_shape[1] + 1) / 2 ? gid[1] : gid[1] + input_shape[1] - output_shape[1];
        input += at(batch, iz, iy, input_stride);
        output += at(batch, gid[0], gid[1], output_stride);

        for (int i = 0; i < 2; ++i) {
            const uint x = gid[2] + THREADS.x * i;
            if (x < output_shape[2] / 2 + 1)
                output[x * output_stride[3]] = input[x * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cropF2F_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                  T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] >= output_shape[1])
            return;

        const uint iz = gid[0] < (output_shape[0] + 1) / 2 ? gid[0] : gid[0] + input_shape[0] - output_shape[0];
        const uint iy = gid[1] < (output_shape[1] + 1) / 2 ? gid[1] : gid[1] + input_shape[1] - output_shape[1];
        input += at(batch, iz, iy, input_stride);
        output += at(batch, gid[0], gid[1], output_stride);

        for (int i = 0; i < 2; ++i) {
            const uint ox = gid[2] + THREADS.x * i;
            const uint ix = ox < (output_shape[2] + 1) / 2 ? ox : ox + input_shape[2] - output_shape[2];
            if (ox < output_shape[2])
                output[ox * output_stride[3]] = input[ix * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void padH2H_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                 T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] >= output_shape[1])
            return;

        const uint oz = gid[0] < (input_shape[0] + 1) / 2 ? gid[0] : gid[0] + output_shape[0] - input_shape[0];
        const uint oy = gid[1] < (input_shape[1] + 1) / 2 ? gid[1] : gid[1] + output_shape[1] - input_shape[1];
        input += at(batch, gid[0], gid[1], input_stride);
        output += at(batch, oz, oy, output_stride);

        for (int i = 0; i < 2; ++i) {
            const uint x = gid[2] + THREADS.x * i;
            if (x < input_shape[2] / 2 + 1)
                output[x * output_stride[3]] = input[x * input_stride[3]];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void padF2F_(const T* __restrict__ input, uint4_t input_stride, uint3_t input_shape,
                 T* __restrict__ output, uint4_t output_stride, uint3_t output_shape, uint blocks_x) {
        const uint batch = blockIdx.z;
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint3_t gid(blockIdx.y,
                          THREADS.y * index[0] + threadIdx.y,
                          THREADS.x * index[1] * 2 + threadIdx.x);
        if (gid[1] >= output_shape[1])
            return;

        const uint oz = gid[0] < (input_shape[0] + 1) / 2 ? gid[0] : gid[0] + output_shape[0] - input_shape[0];
        const uint oy = gid[1] < (input_shape[1] + 1) / 2 ? gid[1] : gid[1] + output_shape[1] - input_shape[1];
        input += at(batch, gid[0], gid[1], input_stride);
        output += at(batch, oz, oy, output_stride);

        for (int i = 0; i < 2; ++i) {
            const uint ix = gid[2] + THREADS.x * i;
            if (ix < input_shape[2]) {
                const uint ox = ix < (input_shape[2] + 1) / 2 ? ix : ix + output_shape[2] - input_shape[2];
                output[ox * output_stride[3]] = input[ix * input_stride[3]];
            }
        }
    }
}

namespace noa::cuda::fft::details {
    template<typename T>
    void cropH2H(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape.fft(), stream);

        const uint3_t old_shape(input_shape.get() + 1);
        const uint3_t new_shape(output_shape.get() + 1);
        const uint blocks_x = math::divideUp(new_shape[2] / 2U + 1, THREADS.x * 2);
        const uint blocks_y = math::divideUp(new_shape[1], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);
        stream.enqueue("cropH2H_", cropH2H_<T>, {blocks, THREADS},
                       input, uint4_t{input_stride}, old_shape, output, uint4_t{output_stride}, new_shape, blocks_x);
    }

    template<typename T>
    void cropF2F(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape >= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape, stream);

        const uint3_t old_shape(input_shape.get() + 1);
        const uint3_t new_shape(output_shape.get() + 1);
        const uint blocks_x = math::divideUp(new_shape[2], THREADS.x * 2);
        const uint blocks_y = math::divideUp(new_shape[1], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);
        stream.enqueue("cropF2F_", cropF2F_<T>, {blocks, THREADS},
                       input, uint4_t{input_stride}, old_shape, output, uint4_t{output_stride}, new_shape, blocks_x);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padH2H(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape.fft(), stream);

        memory::set(output, output_stride, output_shape.fft(), T{0}, stream);

        const uint3_t old_shape(input_shape.get() + 1);
        const uint3_t new_shape(output_shape.get() + 1);
        const uint blocks_x = math::divideUp(old_shape[2] / 2 + 1, THREADS.x * 2);
        const uint blocks_y = math::divideUp(old_shape[1], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);
        stream.enqueue("padH2H_", padH2H_<T>, {blocks, THREADS},
                       input, uint4_t{input_stride}, old_shape, output, uint4_t{output_stride}, new_shape, blocks_x);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padF2F(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        NOA_ASSERT(all(input_shape <= output_shape));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape == output_shape))
            return memory::copy(input, input_stride, output, output_stride, input_shape, stream);

        memory::set(output, output_stride, output_shape, T{0}, stream);
        const uint3_t old_shape(input_shape.get() + 1);
        const uint3_t new_shape(output_shape.get() + 1);
        const uint blocks_x = math::divideUp(old_shape[2], THREADS.x * 2);
        const uint blocks_y = math::divideUp(old_shape[1], THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);
        stream.enqueue("padF2F_", padF2F_<T>, {blocks, THREADS},
                       input, uint4_t{input_stride}, old_shape, output, uint4_t{output_stride}, new_shape, blocks_x);
    }

    #define NOA_INSTANTIATE_CROP_(T)                                                        \
    template void cropH2H<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, Stream&);    \
    template void cropF2F<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, Stream&);    \
    template void padH2H<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, Stream&);     \
    template void padF2F<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_CROP_(half_t);
    NOA_INSTANTIATE_CROP_(float);
    NOA_INSTANTIATE_CROP_(double);
    NOA_INSTANTIATE_CROP_(chalf_t);
    NOA_INSTANTIATE_CROP_(cfloat_t);
    NOA_INSTANTIATE_CROP_(cdouble_t);
}
