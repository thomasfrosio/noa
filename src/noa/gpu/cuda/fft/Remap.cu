#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Remap.h"
#include "noa/gpu/cuda/util/ExternShared.h"

namespace {
    using namespace noa;
    constexpr uint MAX_THREADS = 256;

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2h_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_half) * blockIdx.z;
        out += out_pitch * rows(shape_half) * blockIdx.z;

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
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hc_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_half) * blockIdx.z;
        out += out_pitch * rows(shape_half) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_half.y + out_y) * out_pitch;

        // Select corresponding row in the non-centered array.
        uint in_y = math::iFFTShift(out_y, shape_half.y);
        uint in_z = math::iFFTShift(out_z, shape_half.z);
        in += (in_z * shape_half.y + in_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    // In-place, Y and Z dimensions have both an even number of elements.
    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hcInPlace_(T* out, uint out_pitch, uint3_t shape_half) {
        uint out_y = blockIdx.x;
        uint out_z = blockIdx.y;

        // Rebase to the current batch.
        out += out_pitch * rows(shape_half) * blockIdx.z;

        // Select current row and corresponding row in the non-centered array.
        uint in_y = math::iFFTShift(out_y, shape_half.y);
        uint in_z = math::iFFTShift(out_z, shape_half.z);
        uint in_offset = (in_z * shape_half.y + in_y) * out_pitch;
        uint out_offset = (out_z * shape_half.y + out_y) * out_pitch;

        // Copy the row.
        T* shared = cuda::ExternShared<T>::getBlockResource();
        int count = 0;
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x, ++count) {
            shared[x - count * blockDim.x] = out[out_offset + x];
            out[out_offset + x] = out[in_offset + x];
            out[in_offset + x] = shared[x - count * blockDim.x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void f2fc_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_full) * blockIdx.z;
        out += out_pitch * rows(shape_full) * blockIdx.z;

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
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2f_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_full) * blockIdx.z;
        out += out_pitch * rows(shape_full) * blockIdx.z;

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
    __global__ __launch_bounds__(MAX_THREADS)
    void f2h_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_half) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_half) * blockIdx.z;
        out += out_pitch * rows(shape_half) * blockIdx.z;

        // Rebase to the current row.
        out += (idx_z * shape_half.y + idx_y) * out_pitch;
        in += (idx_z * shape_half.y + idx_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2f_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_full) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;
        uint half_x = shape_full.x / 2 + 1;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_full) * blockIdx.z;
        out += out_pitch * rows(shape_full) * blockIdx.z;

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
    __global__ __launch_bounds__(MAX_THREADS)
    void f2hc_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_half) {
        uint o_y = blockIdx.x, o_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_half) * blockIdx.z;
        out += out_pitch * rows(shape_half) * blockIdx.z;

        // Rebase to the current row.
        uint i_z = math::iFFTShift(o_z, shape_half.z);
        uint i_y = math::iFFTShift(o_y, shape_half.y);
        out += (o_z * shape_half.y + o_y) * out_pitch;
        in += (i_z * shape_half.y + i_y) * in_pitch;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void hc2f_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_full) {
        uint o_y = blockIdx.x, o_z = blockIdx.y;
        uint half_x = shape_full.x / 2 + 1;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_full) * blockIdx.z;
        out += out_pitch * rows(shape_full) * blockIdx.z;

        // Rebase to the current row.
        out += (o_z * shape_full.y + o_y) * out_pitch;

        // Copy the first half of the row.
        uint i_z = math::FFTShift(o_z, shape_full.z);
        uint i_y = math::FFTShift(o_y, shape_full.y);
        for (uint x = threadIdx.x; x < half_x; x += blockDim.x)
            out[x] = in[(i_z * shape_full.y + i_y) * in_pitch + x];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        i_z = math::FFTShift(o_z ? shape_full.z - o_z : o_z, shape_full.z);
        i_y = math::FFTShift(o_y ? shape_full.y - o_y : o_y, shape_full.y);
        in += (i_z * shape_full.y + i_y) * in_pitch;

        // Flip (and conjugate if complex) and copy to generate the redundant elements.
        for (uint x = half_x + threadIdx.x; x < shape_full.x; x += blockDim.x) {
            if constexpr (noa::traits::is_complex_v<T>)
                out[x] = math::conj(in[shape_full.x - x]);
            else
                out[x] = in[shape_full.x - x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(MAX_THREADS)
    void fc2h_(const T* __restrict__ in, uint in_pitch, T* __restrict__ out, uint out_pitch, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += in_pitch * rows(shape_full) * blockIdx.z;
        out += out_pitch * rows(shape_full) * blockIdx.z;

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

namespace noa::cuda::fft::details {
    template<typename T>
    void hc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_half(shapeFFT(shape));
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, static_cast<uint>(batches)};
        hc2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void h2hc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_half(shapeFFT(shape));
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));

        if (inputs == outputs) {
            if ((shape.y != 1 && shape.y % 2) || (shape.z != 1 && shape.z % 2))
                NOA_THROW("In-place roll is only available when y and z have an even number of elements");
            dim3 blocks{noa::math::max(shape_half.y / 2, 1U), shape_half.z, static_cast<uint>(batches)};
            h2hcInPlace_<<<blocks, threads, threads * sizeof(T), stream.get()>>>(outputs, outputs_pitch, shape_half);
        } else {
            dim3 blocks{shape_half.y, shape_half.z, static_cast<uint>(batches)};
            h2hc_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void f2fc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_full(shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, static_cast<uint>(batches)};
        f2fc_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void fc2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_full(shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, static_cast<uint>(batches)};
        fc2f_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void f2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
             size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_half(shapeFFT(shape));
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, static_cast<uint>(batches)};
        f2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void h2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
             size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_full(shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, static_cast<uint>(batches)};
        h2f_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void f2hc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_half(shapeFFT(shape));
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, static_cast<uint>(batches)};
        f2hc_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_half);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void hc2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_full(shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, static_cast<uint>(batches)};
        hc2f_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<typename T>
    void fc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
              size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(inputs != outputs);
        uint3_t shape_full(shape);
        uint threads = math::min(MAX_THREADS, math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, static_cast<uint>(batches)};
        fc2h_<<<blocks, threads, 0, stream.get()>>>(inputs, inputs_pitch, outputs, outputs_pitch, shape_full);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_REMAPS_(T)                                              \
    template void hc2h<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void h2hc<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void f2fc<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void fc2f<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void f2h<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);   \
    template void h2f<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);   \
    template void f2hc<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void hc2f<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&);  \
    template void fc2h<T>(const T*, size_t, T*, size_t, size3_t, size_t, Stream&)

    NOA_INSTANTIATE_REMAPS_(cfloat_t);
    NOA_INSTANTIATE_REMAPS_(float);
}
