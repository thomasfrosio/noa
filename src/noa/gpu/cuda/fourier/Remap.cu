#include "noa/Math.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/fourier/Remap.h"

namespace {
    using namespace Noa;

    template<class T>
    __global__ void HC2H_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_half) * blockIdx.z;
        out += pitch_out * getRows(shape_half) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_half.y + out_y) * pitch_out;

        // Select corresponding row in the centered array.
        uint in_y = Math::FFTShift(out_y, shape_half.y), in_z = Math::FFTShift(out_z, shape_half.z);
        in += (in_z * shape_half.y + in_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void H2HC_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_half) * blockIdx.z;
        out += pitch_out * getRows(shape_half) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_half.y + out_y) * pitch_out;

        // Select corresponding row in the non-centered array.
        uint in_y = Math::iFFTShift(out_y, shape_half.y), in_z = Math::iFFTShift(out_z, shape_half.z);
        in += (in_z * shape_half.y + in_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void F2FC_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_full) * blockIdx.z;
        out += pitch_out * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * pitch_out;

        // Select corresponding row in the non-centered array.
        uint in_y = Math::iFFTShift(out_y, shape_full.y), in_z = Math::iFFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x; x += blockDim.x)
            out[x] = in[Math::iFFTShift(x, shape_full.x)];
    }

    template<class T>
    __global__ void FC2F_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_full) * blockIdx.z;
        out += pitch_out * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * pitch_out;

        // Select corresponding row in the non-centered array.
        uint in_y = Math::FFTShift(out_y, shape_full.y), in_z = Math::FFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x; x += blockDim.x)
            out[x] = in[Math::FFTShift(x, shape_full.x)];
    }

    template<class T>
    __global__ void F2H_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_half) * blockIdx.z;
        out += pitch_out * getRows(shape_half) * blockIdx.z;

        // Rebase to the current row.
        out += (idx_z * shape_half.y + idx_y) * pitch_out;
        in += (idx_z * shape_half.y + idx_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_half.x; x += blockDim.x)
            out[x] = in[x];
    }

    template<class T>
    __global__ void H2F_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
        uint idx_y = blockIdx.x, idx_z = blockIdx.y;
        uint half_x = shape_full.x / 2 + 1;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_full) * blockIdx.z;
        out += pitch_out * getRows(shape_full) * blockIdx.z;

        // Rebase to the current row.
        out += (idx_z * shape_full.y + idx_y) * pitch_out;

        // Copy the first half of the row.
        for (uint x = threadIdx.x; x < half_x; x += blockDim.x)
            out[x] = in[(idx_z * shape_full.y + idx_y) * pitch_in + x];

        // Rebase to the symmetric row in the non-redundant array corresponding to the redundant elements.
        if (idx_y) idx_y = shape_full.y - idx_y;
        if (idx_z) idx_z = shape_full.z - idx_z;
        in += (idx_z * shape_full.y + idx_y) * pitch_in;

        // Flip (and conjugate if complex) and copy to generate the redundant elements.
        for (uint x = half_x + threadIdx.x; x < shape_full.x; x += blockDim.x) {
            if constexpr (Noa::Traits::is_complex_v<T>)
                out[x] = Math::conj(in[shape_full.x - x]);
            else
                out[x] = in[shape_full.x - x];
        }
    }

    template<class T>
    __global__ void FC2H_(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
        uint out_y = blockIdx.x, out_z = blockIdx.y;

        // Rebase to the current batch.
        in += pitch_in * getRows(shape_full) * blockIdx.z;
        out += pitch_out * getRows(shape_full) * blockIdx.z;

        // Select current row.
        out += (out_z * shape_full.y + out_y) * pitch_out;

        // Select corresponding row in the non-centered array.
        uint in_y = Math::FFTShift(out_y, shape_full.y), in_z = Math::FFTShift(out_z, shape_full.z);
        in += (in_z * shape_full.y + in_y) * pitch_in;

        // Copy the row.
        for (uint x = threadIdx.x; x < shape_full.x / 2 + 1; x += blockDim.x)
            out[x] = in[Math::FFTShift(x, shape_full.x)];
    }
}

namespace Noa::CUDA::Fourier {
    template<typename T>
    void HC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        HC2H_,
                        in, pitch_in, out, pitch_out, shape_half);
    }

    template<typename T>
    void H2HC(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        H2HC_,
                        in, pitch_in, out, pitch_out, shape_half);
    }

    template<typename T>
    void F2FC(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        F2FC_,
                        in, pitch_in, out, pitch_out, shape_full);
    }

    template<typename T>
    void FC2F(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_full.x, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        FC2F_,
                        in, pitch_in, out, pitch_out, shape_full);
    }

    // TODO: not a priority, but check if a memcpy is faster.
    template<typename T>
    void F2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_half.x, Limits::WARP_SIZE));
        dim3 blocks{shape_half.y, shape_half.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        F2H_,
                        in, pitch_in, out, pitch_out, shape_half);
    }

    template<typename T>
    void H2F(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        H2F_,
                        in, pitch_in, out, pitch_out, shape_full);
    }

    template<typename T>
    void FC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batches, Stream& stream) {
        uint3_t shape_full(shape);
        uint threads = Math::min(256U, Math::nextMultipleOf(shape_full.x / 2 + 1, Limits::WARP_SIZE));
        dim3 blocks{shape_full.y, shape_full.z, batches};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.get(),
                        FC2H_,
                        in, pitch_in, out, pitch_out, shape_full);
    }
}

// Instantiate supported types.
namespace Noa::CUDA::Fourier {
    #define INSTANTIATE_REMAPS(T)                                                   \
    template void HC2H<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void H2HC<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void F2FC<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void FC2F<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);    \
    template void F2H<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);     \
    template void H2F<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&);     \
    template void FC2H<T>(const T*, size_t, T*, size_t, size3_t, uint, Stream&)

    INSTANTIATE_REMAPS(cfloat_t);
    INSTANTIATE_REMAPS(float);
}
