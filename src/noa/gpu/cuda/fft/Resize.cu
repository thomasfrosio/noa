#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Resize.h"

namespace {
    using namespace noa;
    const uint MAX_THREADS = 256;

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void crop_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
               T* __restrict__ out, uint3_t shape_out, uint out_pitch) {
        // Rebase to the current batch.
        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint out_y = blockIdx.x, out_z = blockIdx.y;
        uint in_y = out_y < (shape_out.y + 1) / 2 ? out_y : out_y + shape_in.y - shape_out.y;
        uint in_z = out_z < (shape_out.z + 1) / 2 ? out_z : out_z + shape_in.z - shape_out.z;

        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_out.x / 2 + 1; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void cropFull_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
                   T* __restrict__ out, uint3_t shape_out, uint out_pitch) {
        // Rebase to the current batch.
        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint out_y = blockIdx.x, out_z = blockIdx.y;
        uint in_y = out_y < (shape_out.y + 1) / 2 ? out_y : out_y + shape_in.y - shape_out.y;
        uint in_z = out_z < (shape_out.z + 1) / 2 ? out_z : out_z + shape_in.z - shape_out.z;
        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        // Similarly to the other dimension, if half in new x is passed, add offset to skip cropped elements.
        for (uint out_x = threadIdx.x; out_x < shape_out.x; out_x += blockDim.x) {
            uint in_x = out_x < (shape_out.x + 1) / 2 ? out_x : out_x + shape_in.x - shape_out.x;
            out[out_x] = in[in_x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void pad_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
              T* __restrict__ out, uint3_t shape_out, uint out_pitch) {
        // Rebase to the current batch.
        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint in_y = blockIdx.x, in_z = blockIdx.y;
        uint out_y = in_y < (shape_in.y + 1) / 2 ? in_y : in_y + shape_out.y - shape_in.y;
        uint out_z = in_z < (shape_in.z + 1) / 2 ? in_z : in_z + shape_out.z - shape_in.z;
        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_in.x / 2 + 1; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void padFull_(const T* __restrict__ in, uint3_t shape_in, uint in_pitch,
                  T* __restrict__ out, uint3_t shape_out, uint out_pitch) {
        // Rebase to the current batch.
        in += in_pitch * shape_in.y * shape_in.z * blockIdx.z;
        out += out_pitch * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint in_y = blockIdx.x, in_z = blockIdx.y;
        uint out_y = in_y < (shape_in.y + 1) / 2 ? in_y : in_y + shape_out.y - shape_in.y;
        uint out_z = in_z < (shape_in.z + 1) / 2 ? in_z : in_z + shape_out.z - shape_in.z;
        in += (in_z * shape_in.y + in_y) * in_pitch;
        out += (out_z * shape_out.y + out_y) * out_pitch;

        // Similarly to the other dimension, if half in new x is passed, add offset to skip padded elements.
        for (uint in_x = threadIdx.x; in_x < shape_in.x; in_x += blockDim.x) {
            uint out_x = in_x < (shape_in.x + 1) / 2 ? in_x : in_x + shape_out.x - shape_in.x;
            out[out_x] = in[in_x];
        }
    }
}

namespace noa::cuda::fft {
    template<typename T>
    void crop(const T* inputs, size_t input_pitch, size3_t input_shape,
              T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), stream);

        uint3_t old_shape(input_shape), new_shape(output_shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(new_shape.x / 2U + 1, Limits::WARP_SIZE));
        dim3 blocks{new_shape.y, new_shape.z, static_cast<uint>(batches)};
        crop_<<<blocks, threads, 0, stream.get()>>>(inputs, old_shape, input_pitch, outputs, new_shape, output_pitch);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void cropFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                  T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, stream);

        uint3_t old_shape(input_shape), new_shape(output_shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(new_shape.x, Limits::WARP_SIZE));
        dim3 blocks{new_shape.y, new_shape.z, static_cast<uint>(batches)};
        cropFull_<<<blocks, threads, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void pad(const T* inputs, size_t input_pitch, size3_t input_shape,
             T* outputs, size_t output_pitch, size3_t output_shape, size_t batches, Stream& stream) {
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(input_shape), stream);

        memory::set(outputs, output_pitch * rows(output_shape), T{0}, stream);
        uint3_t old_shape(input_shape), new_shape(output_shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(old_shape.x / 2U + 1U, Limits::WARP_SIZE));
        dim3 blocks{old_shape.y, old_shape.z, static_cast<uint>(batches)};
        pad_<<<blocks, threads, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    // TODO(TF) Replace memset with kernel that loops through padded regions as well.
    template<typename T>
    void padFull(const T* inputs, size_t input_pitch, size3_t input_shape,
                 T* outputs, size_t output_pitch, size3_t output_shape,
                 size_t batches, Stream& stream) {
        if (all(input_shape == output_shape))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, input_shape, stream);

        memory::set(outputs, output_pitch * rows(output_shape), T{0}, stream);
        uint3_t old_shape(input_shape), new_shape(output_shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(old_shape.x, Limits::WARP_SIZE));
        dim3 blocks{old_shape.y, old_shape.z, static_cast<uint>(batches)};
        padFull_<<<blocks, threads, 0, stream.get()>>>(
                inputs, old_shape, input_pitch, outputs, new_shape, output_pitch);
        NOA_THROW_IF(cudaPeekAtLastError());
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
