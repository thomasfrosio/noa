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
    void crop_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
               T* __restrict__ out, uint3_t shape_out, uint out_pitch, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.y >= shape_out.y)
            return;

        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;
        const uint in_y = gid.y < (shape_out.y + 1) / 2 ? gid.y : gid.y + shape_in.y - shape_out.y;
        const uint in_z = gid.z < (shape_out.z + 1) / 2 ? gid.z : gid.z + shape_in.z - shape_out.z;
        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (gid.z * shape_out.y + gid.y) * out_pitch;

        for (int i = 0; i < 2; ++i) {
            const uint x = gid.x + THREADS.x * i;
            if (x < shape_out.x / 2 + 1)
                out[x] = in[x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void cropFull_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
                   T* __restrict__ out, uint3_t shape_out, uint out_pitch, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.y >= shape_out.y)
            return;

        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;
        const uint in_y = gid.y < (shape_out.y + 1) / 2 ? gid.y : gid.y + shape_in.y - shape_out.y;
        const uint in_z = gid.z < (shape_out.z + 1) / 2 ? gid.z : gid.z + shape_in.z - shape_out.z;
        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (gid.z * shape_out.y + gid.y) * out_pitch;

        for (int i = 0; i < 2; ++i) {
            const uint out_x = gid.x + THREADS.x * i;
            const uint in_x = out_x < (shape_out.x + 1) / 2 ? out_x : out_x + shape_in.x - shape_out.x;
            if (out_x < shape_out.x)
                out[out_x] = in[in_x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void pad_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
              T* __restrict__ out, uint3_t shape_out, uint out_pitch, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.y >= shape_in.y)
            return;

        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;
        const uint out_y = gid.y < (shape_in.y + 1) / 2 ? gid.y : gid.y + shape_out.y - shape_in.y;
        const uint out_z = gid.z < (shape_in.z + 1) / 2 ? gid.z : gid.z + shape_out.z - shape_in.z;
        in += (gid.z * shape_in.y + gid.y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        for (int i = 0; i < 2; ++i) {
            const uint x = gid.x + THREADS.x * i;
            if (x < shape_in.x / 2 + 1)
                out[x] = in[x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void padFull_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
                  T* __restrict__ out, uint3_t shape_out, uint out_pitch, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.y >= shape_in.y)
            return;

        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;
        const uint out_y = gid.y < (shape_in.y + 1) / 2 ? gid.y : gid.y + shape_out.y - shape_in.y;
        const uint out_z = gid.z < (shape_in.z + 1) / 2 ? gid.z : gid.z + shape_out.z - shape_in.z;
        in += (gid.z * shape_in.y + gid.y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        for (int i = 0; i < 2; ++i) {
            const uint in_x = gid.x + THREADS.x * i;
            if (in_x < shape_in.x) {
                const uint out_x = in_x < (shape_in.x + 1) / 2 ? in_x : in_x + shape_out.x - shape_in.x;
                out[out_x] = in[in_x];
            }
        }
    }
}

namespace noa::cuda::fft {
    template<typename T>
    void crop(const T* inputs, size_t input_pitch, size3_t input_shape,
              T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), batches, stream);

        const uint3_t old_shape(input_shape);
        const uint3_t new_shape(output_shape);
        const uint blocks_x = math::divideUp(new_shape.x / 2U + 1, THREADS.x * 2);
        const uint blocks_y = math::divideUp(new_shape.y, THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape.z, batches);
        crop_<<<blocks, THREADS, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void cropFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                  T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches, stream);

        const uint3_t old_shape(input_shape);
        const uint3_t new_shape(output_shape);
        const uint blocks_x = math::divideUp(new_shape.x, THREADS.x * 2);
        const uint blocks_y = math::divideUp(new_shape.y, THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape.z, batches);
        cropFull_<<<blocks, THREADS, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void pad(const T* inputs, size_t input_pitch, size3_t input_shape,
             T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), batches, stream);

        memory::set(outputs, output_pitch * rows(output_shape) * batches, T{0}, stream);

        const uint3_t old_shape(input_shape);
        const uint3_t new_shape(output_shape);
        const uint blocks_x = math::divideUp(old_shape.x / 2 + 1, THREADS.x * 2);
        const uint blocks_y = math::divideUp(old_shape.y, THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape.z, batches);
        pad_<<<blocks, THREADS, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                 T* outputs, size_t output_pitch, size3_t output_shape,
                 size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches, stream);

        memory::set(outputs, output_pitch * rows(output_shape) * batches, T{0}, stream);
        const uint3_t old_shape(input_shape);
        const uint3_t new_shape(output_shape);
        const uint blocks_x = math::divideUp(old_shape.x, THREADS.x * 2);
        const uint blocks_y = math::divideUp(old_shape.y, THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape.z, batches);
        padFull_<<<blocks, THREADS, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_CROP_(T)                                                            \
    template void crop<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size_t, Stream&);     \
    template void cropFull<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size_t, Stream&); \
    template void pad<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size_t, Stream&);      \
    template void padFull<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size_t, Stream&)

    NOA_INSTANTIATE_CROP_(cfloat_t);
    NOA_INSTANTIATE_CROP_(float);
    NOA_INSTANTIATE_CROP_(cdouble_t);
    NOA_INSTANTIATE_CROP_(double);
}
