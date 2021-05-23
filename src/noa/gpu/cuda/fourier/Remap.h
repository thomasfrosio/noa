#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Fourier {
    /**
     * CUDA version of Noa::Fourier::HC2H. The same features and restrictions apply to this function.
     * @tparam T            Real or complex.
     * @param[in] in        The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param pitch_in      Pitch of @a in, in number of @a T elements.
     * @param[out] out      The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param pitch_out     Pitch of @a out, in number of @a T elements.
     * @param shape         Logical {fast, medium, slow} shape of @a in and @a out.
     * @param batches       Number of contiguous batches to process.
     * @param[in] stream    Stream on which to enqueue this function.
     *                      Both @a in and @a out should be on the device associated with this stream.
     */
    template<typename T>
    NOA_HOST void HC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::HC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void HC2H(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        HC2H(in, shape.x / 2 + 1, out, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2HC(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2HC(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        H2HC(in, shape.x / 2 + 1, out, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void FC2F(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void FC2F(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        FC2F(in, shape.x, out, shape.x, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void F2FC(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2FC(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        F2FC(in, shape.x, out, shape.x, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void F2H(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                    size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2H(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        F2H(in, shape.x, out, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2F(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                      size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2F(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        H2F(in, shape.x / 2 + 1, out, shape.x, shape, batches, stream);
    }

    /// CUDA version of Noa::Fourier::FC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void FC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of Noa::Fourier::FC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void FC2H(const T* in, T* out, size3_t shape, uint batches, Stream& stream) {
        FC2H(in, shape.x, out, shape.x / 2 + 1, shape, batches, stream);
    }
}
