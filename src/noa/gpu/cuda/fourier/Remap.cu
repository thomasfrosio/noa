#include "noa/gpu/cuda/fourier/Remap.h"
#include "noa/util/Math.h"

// Forward declarations
namespace Noa::CUDA::Fourier::Kernels {
    template<class T>
    __global__ void HC2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half);

    template<class T>
    __global__ void H2HC(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half);

    template<class T>
    __global__ void F2FC(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full);

    template<class T>
    __global__ void FC2F(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full);

    template<class T>
    __global__ void F2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half);

    template<class T>
    __global__ void H2F(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full);

    template<class T>
    __global__ void FC2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full);
}

namespace Noa::CUDA::Fourier {
    template<typename T>
    void HC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_half.x, Limits::warp_size));
        dim3 rows_to_process{shape_half.y, shape_half.z, batch};
        Kernels::HC2H<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void H2HC(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_half.x, Limits::warp_size));
        dim3 rows_to_process{shape_half.y, shape_half.z, batch};
        Kernels::H2HC<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void F2FC(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_full(shape);
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_full.x, Limits::warp_size));
        dim3 rows_to_process{shape_full.y, shape_full.z, batch};
        Kernels::F2FC<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void FC2F(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_full(shape);
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_full.x, Limits::warp_size));
        dim3 rows_to_process{shape_full.y, shape_full.z, batch};
        Kernels::FC2F<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    // TODO: not a priority, but check if a memcpy is faster.
    template<typename T>
    void F2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_half(getShapeFFT(shape));
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_half.x, Limits::warp_size));
        dim3 rows_to_process{shape_half.y, shape_half.z, batch};
        Kernels::F2H<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_half);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void H2F(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_full(shape);
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_full.x / 2 + 1, Limits::warp_size));
        dim3 rows_to_process{shape_full.y, shape_full.z, batch};
        Kernels::H2F<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<typename T>
    void FC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, uint batch, Stream& stream) {
        uint3_t shape_full(shape);
        uint workers_per_row = Math::min(256U, getNextMultipleOf(shape_full.x / 2 + 1, Limits::warp_size));
        dim3 rows_to_process{shape_full.y, shape_full.z, batch};
        Kernels::FC2H<<<rows_to_process, workers_per_row, 0, stream.get()>>>(
                in, static_cast<uint>(pitch_in), out, static_cast<uint>(pitch_out), shape_full);
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

namespace Noa::CUDA::Fourier::Kernels {
    template<class T>
    __global__ void HC2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
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
    __global__ void H2HC(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
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
    __global__ void F2FC(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
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
    __global__ void FC2F(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
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
    __global__ void F2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_half) {
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
    __global__ void H2F(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
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
    __global__ void FC2H(const T* in, uint pitch_in, T* out, uint pitch_out, uint3_t shape_full) {
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

// Instantiate supported types.
namespace Noa::CUDA::Fourier {
    #define INSTANTIATE_HC2H(T) \
    template void HC2H<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_HC2H(cfloat_t);
    INSTANTIATE_HC2H(float);

    #define INSTANTIATE_H2HC(T) \
    template void H2HC<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_H2HC(cfloat_t);
    INSTANTIATE_H2HC(float);

    #define INSTANTIATE_F2FC(T) \
    template void F2FC<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_F2FC(cfloat_t);
    INSTANTIATE_F2FC(float);

    #define INSTANTIATE_FC2F(T) \
    template void FC2F<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_FC2F(cfloat_t);
    INSTANTIATE_FC2F(float);

    #define INSTANTIATE_F2H(T) \
    template void F2H<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_F2H(cfloat_t);
    INSTANTIATE_F2H(float);

    #define INSTANTIATE_H2F(T) \
    template void H2F<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_H2F(cfloat_t);
    INSTANTIATE_H2F(float);

    #define INSTANTIATE_FC2H(T) \
    template void FC2H<T>(const T* in, size_t pitch_in, T* out, size_t pitch_out, size3_t shape, \
                          uint batch, Stream& stream)

    INSTANTIATE_FC2H(cfloat_t);
    INSTANTIATE_FC2H(float);
}
