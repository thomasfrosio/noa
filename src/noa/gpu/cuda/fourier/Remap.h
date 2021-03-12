#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Fourier {
    /**
     * CUDA version of Noa::Fourier::HC2H. The same features and restrictions apply to this function.
     * @param pitch_in      Pitch of @a in, in number of @a T elements.
     * @param pitch_out     Pitch of @a out, in number of @a T elements.
     * @param batch         Number of contiguous batches to process.
     * @param stream        Stream on which to enqueue this function.
     */
    template<typename T>
    NOA_HOST void HC2H(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::HC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void HC2H(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        HC2H(in, shape.x / 2 + 1, out, shape.x / 2 + 1, shape, batch, stream);
    }

    /// CUDA version of Noa::Fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2HC(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2HC(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        H2HC(in, shape.x / 2 + 1, out, shape.x / 2 + 1, shape, batch, stream);
    }

    /// CUDA version of Noa::Fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void FC2F(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void FC2F(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        FC2F(in, shape.x, out, shape.x, shape, batch, stream);
    }

    /// CUDA version of Noa::Fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void F2FC(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                       size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2FC(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        F2FC(in, shape.x, out, shape.x, shape, batch, stream);
    }

    /// CUDA version of Noa::Fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2H(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                    size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2H(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        F2H(in, shape.x, out, shape.x / 2 + 1, shape, batch, stream);
    }

    /// CUDA version of Noa::Fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2F(const T* in, size_t pitch_in, T* out, size_t pitch_out,
                      size3_t shape, uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2F(const T* in, T* out, size3_t shape, uint batch, Stream& stream) {
        H2F(in, shape.x / 2 + 1, out, shape.x, shape, batch, stream);
    }
}
