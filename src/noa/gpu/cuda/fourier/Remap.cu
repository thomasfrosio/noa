#include "noa/common/Math.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/fourier/Remap.h"

namespace {
    using namespace noa;

    template<class T>
    __global__ void hc2h_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_half) * blockIdx.z;
        out += out_pitch * getRows(shape_half) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_half.y + out_y) * out_pitch;

        // Select corresponding row in the centered array.
        uint in_y = math::FFTShift(out_y, shape_half.y), in_z = math::FFTShift(out_z, shape_half.z);
        in += (in_z * shape_half.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void h2hc_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_half) * blockIdx.z;
        out += out_pitch * getRows(shape_half) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_half.y + out_y) * out_pitch;

        // Select corresponding row in the non-centered array.
        uint in_y = math::iFFTShift(out_y, shape_half.y), in_z = math::iFFTShift(out_z, shape_half.z);
        in += (in_z * shape_half.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void f2fc_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_full) * blockIdx.z;
        out += out_pitch * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * out_pitch;

        // Select corresponding row in the non-centered array.
        uint in_y = math::iFFTShift(out_y, shape_full.y), in_z = math::iFFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x; x += blockDim.x)
            out[x] = in[math::iFFTShift(x, shape_full.x)];
    }

    template<class T>
    __global__ void fc2f_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_full) * blockIdx.z;
        out += out_pitch * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * out_pitch;

        // Select corresponding row in the non-centered array.
        uint in_y = math::FFTShift(out_y, shape_full.y), in_z = math::FFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x; x += blockDim.x)
            out[x] = in[math::FFTShift(x, shape_full.x)];
    }

    template<class T>
    __global__ void f2h_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_half) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_half) * blockIdx.z;
        out += out_pitch * getRows(shape_half) * blockIdx.z;

        // Rebase to the current row.
        out += (idx_z * shape_half.y + idx_y) * out_pitch;
        in += (idx_z * shape_half.y + idx_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void h2f_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_full) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;
        uint half_x = shape_full.x / 2 + 1;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_full) * blockIdx.z;
        out += out_pitch * getRows(shape_full) * blockIdx.z;

        // Rebase to the current row.
        out += (idx_z * shape_full.y + idx_y) * out_pitch;

        // Copy the first half of the row.
        for (uint x = threadIdx.x; x < half_x; x += blockDim.x)
            out[x] = in[(idx_z * shape_full.y + idx_y) * in_pitch + x];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        if (idx_y) idx_y = shape_full.y - idx_y;
        if (idx_z) idx_z = shape_full.z - idx_z;
        in += (idx_z * shape_full.y + idx_y) * in_pitch;

        // Flip (and conjugate if complex) and copy to generate the redundant elements.
        for (uint x = half_x + threadIdx.x; x < shape_full.x; x += blockDim.x) {
            if constexpr (noa::traits::is_complex_v<T>)
                out[x] = math::conj(in[shape_full.x - x]);
            else
                out[x] = in[shape_full.x - x];
        }
    }

    template<class T>
    __global__ void fc2h_(const T* in, uint in_pitch, T* out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * getRows(shape_full) * blockIdx.z;
        out += out_pitch * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * out_pitch;

        // Select corresponding row in the non-centered array.
        uint in_y = math::FFTShift(out_y, shape_full.y), in_z = math::FFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x / 2 + 1; x += blockDim.x)
            out[x] = in[math::FFTShift(x, shape_full.x)];
    }
}

namespace noa::cuda::fourier {
    template<typename T>
    void hc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = math::min(256U, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        hc2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void h2hc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = math::min(256U, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        h2hc_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void f2fc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = math::min(256U, math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        f2fc_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void fc2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = math::min(256U, math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        fc2f_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    // TODO: not a priority, but check if a memcpy is faster.
    template<typename T>
    void f2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
             size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = math::min(256U, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        f2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void h2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
             size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = math::min(256U, math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        h2f_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void fc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = math::min(256U, math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        fc2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// Instantiate supported types.
namespace noa::cuda::fourier {
    #define INSTANTIATE_REMAPS(T)                                                   \
    template void hc2h<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void h2hc<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void f2fc<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void fc2f<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void f2h<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);     \
    template void h2f<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);     \
    template void fc2h<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_REMAPS(cfloat_t);
    INSTANTIATE_REMAPS(float);
}
