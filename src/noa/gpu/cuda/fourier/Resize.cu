#include "noa/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/fourier/Resize.h"

namespace {
    using namespace Noa;

    template<class T>
    __global__ void crop_(const T* in, uint3_t shape_in, uint pitch_in, T* out, uint3_t shape_out, uint pitch_out) {
        // Rebase to the current batch.
        in += pitch_in * shape_in.y * shape_in.z * blockIdx.z;
        out += pitch_out * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint out_y = blockIdx.x, out_z = blockIdx.y;
        uint in_y = out_y < (shape_out.y + 1) / 2 ? out_y : out_y + shape_in.y - shape_out.y;
        uint in_z = out_z < (shape_out.z + 1) / 2 ? out_z : out_z + shape_in.z - shape_out.z;

        in += (in_z * shape_in.y + in_y) * pitch_in;
        out += (out_z * shape_out.y + out_y) * pitch_out;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_out.x / 2 + 1; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void cropFull_(const T* in, uint3_t shape_in, uint pitch_in, T* out, uint3_t shape_out, uint pitch_out) {
        // Rebase to the current batch.
        in += pitch_in * shape_in.y * shape_in.z * blockIdx.z;
        out += pitch_out * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint out_y = blockIdx.x, out_z = blockIdx.y;
        uint in_y = out_y < (shape_out.y + 1) / 2 ? out_y : out_y + shape_in.y - shape_out.y;
        uint in_z = out_z < (shape_out.z + 1) / 2 ? out_z : out_z + shape_in.z - shape_out.z;
        in += (in_z * shape_in.y + in_y) * pitch_in;
        out += (out_z * shape_out.y + out_y) * pitch_out;

        // Similarly to the other dimension, if half in new x is passed, add offset to skip cropped elements.
        for (uint out_x = threadIdx.x; out_x < shape_out.x; out_x += blockDim.x) {
            uint in_x = out_x < (shape_out.x + 1) / 2 ? out_x : out_x + shape_in.x - shape_out.x;
            out[out_x] = in[in_x];
        }
    }

    template<class T>
    __global__ void pad_(const T* in, uint3_t shape_in, uint pitch_in, T* out, uint3_t shape_out, uint pitch_out) {
        // Rebase to the current batch.
        in += pitch_in * shape_in.y * shape_in.z * blockIdx.z;
        out += pitch_out * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint in_y = blockIdx.x, in_z = blockIdx.y;
        uint out_y = in_y < (shape_in.y + 1) / 2 ? in_y : in_y + shape_out.y - shape_in.y;
        uint out_z = in_z < (shape_in.z + 1) / 2 ? in_z : in_z + shape_out.z - shape_in.z;
        in += (in_z * shape_in.y + in_y) * pitch_in;
        out += (out_z * shape_out.y + out_y) * pitch_out;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_in.x / 2 + 1; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void padFull_(const T* in, uint3_t shape_in, uint pitch_in, T* out, uint3_t shape_out, uint pitch_out) {
        // Rebase to the current batch.
        in += pitch_in * shape_in.y * shape_in.z * blockIdx.z;
        out += pitch_out * shape_out.y * shape_out.z * blockIdx.z;

        // Rebase to the current row.
        uint in_y = blockIdx.x, in_z = blockIdx.y;
        uint out_y = in_y < (shape_in.y + 1) / 2 ? in_y : in_y + shape_out.y - shape_in.y;
        uint out_z = in_z < (shape_in.z + 1) / 2 ? in_z : in_z + shape_out.z - shape_in.z;
        in += (in_z * shape_in.y + in_y) * pitch_in;
        out += (out_z * shape_out.y + out_y) * pitch_out;

        // Similarly to the other dimension, if half in new x is passed, add offset to skip padded elements.
        for (uint in_x = threadIdx.x; in_x < shape_in.x; in_x += blockDim.x) {
            uint out_x = in_x < (shape_in.x + 1) / 2 ? in_x : in_x + shape_out.x - shape_in.x;
            out[out_x] = in[in_x];
        }
    }
}

namespace Noa::CUDA::Fourier {
    template<typename T>
    void crop(const T* in, size3_t shape_in, size_t pitch_in, T* out, size3_t shape_out, size_t pitch_out,
              uint batches, Stream& stream) {
        if (shape_in == shape_out) {
            Memory::copy(in, pitch_in, out, pitch_out, getShapeFFT(shape_in), stream);
            return;
        }
        uint3_t old_shape(shape_in), new_shape(shape_out);
        uint threads = Math::min(256U, Math::nextMultipleOf(new_shape.x / 2U + 1, Limits::WARP_SIZE));
        dim3 blocks{new_shape.y, new_shape.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        crop_,
                        in, old_shape, pitch_in, out, new_shape, pitch_out);
    }

    template<typename T>
    void cropFull(const T* in, size3_t shape_in, size_t pitch_in, T* out, size3_t shape_out, size_t pitch_out,
                  uint batches, Stream& stream) {
        if (shape_in == shape_out) {
            Memory::copy(in, pitch_in, out, pitch_out, shape_in, stream);
            return;
        }
        uint3_t old_shape(shape_in), new_shape(shape_out);
        uint threads = Math::min(256U, Math::nextMultipleOf(new_shape.x, Limits::WARP_SIZE));
        dim3 blocks{new_shape.y, new_shape.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        cropFull_,
                        in, old_shape, pitch_in, out, new_shape, pitch_out);
    }

    // TODO: not a priority, but maybe replace memset with a single kernel that loops through output.
    template<typename T>
    void pad(const T* in, size3_t shape_in, size_t pitch_in, T* out, size3_t shape_out, size_t pitch_out,
             uint batches, Stream& stream) {
        if (shape_in == shape_out) {
            Memory::copy(in, pitch_in, out, pitch_out, getShapeFFT(shape_in), stream);
            return;
        }
        NOA_THROW_IF(cudaMemsetAsync(out, 0, pitch_out * shape_out.y * shape_out.z * sizeof(T), stream.get()));

        uint3_t old_shape(shape_in), new_shape(shape_out);
        uint threads = Math::min(256U, Math::nextMultipleOf(old_shape.x / 2U + 1U, Limits::WARP_SIZE));
        dim3 blocks{old_shape.y, old_shape.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        pad_,
                        in, old_shape, pitch_in, out, new_shape, pitch_out);
    }

    // TODO: not a priority, but maybe replace memset with kernel that loops through output.
    template<typename T>
    void padFull(const T* in, size3_t shape_in, size_t pitch_in, T* out, size3_t shape_out, size_t pitch_out,
                 uint batches, Stream& stream) {
        if (shape_in == shape_out) {
            Memory::copy(in, pitch_in, out, pitch_out, shape_in, stream);
            return;
        }
        NOA_THROW_IF(cudaMemsetAsync(out, 0, pitch_out * shape_out.y * shape_out.z * sizeof(T), stream.get()));

        uint3_t old_shape(shape_in), new_shape(shape_out);
        uint threads = Math::min(256U, Math::nextMultipleOf(old_shape.x, Limits::WARP_SIZE));
        dim3 blocks{old_shape.y, old_shape.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        padFull_,
                        in, old_shape, pitch_in, out, new_shape, pitch_out);
    }

    #define INSTANTIATE_CROP(T) \
    template void crop<T>(const T*, size3_t, size_t, T*, size3_t, size_t, uint, Stream&);       \
    template void cropFull<T>(const T*, size3_t, size_t, T*, size3_t, size_t, uint, Stream&);   \
    template void pad<T>(const T*, size3_t, size_t, T*, size3_t, size_t, uint, Stream&);        \
    template void padFull<T>(const T*, size3_t, size_t, T*, size3_t, size_t, uint, Stream&)

    INSTANTIATE_CROP(cfloat_t);
    INSTANTIATE_CROP(float);
}
