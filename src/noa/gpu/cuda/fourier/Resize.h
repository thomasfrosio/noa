#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Fourier {
    /**
     * CUDA version of Noa::Fourier::crop. The same features and restrictions apply to this function.
     * @tparam T        float or cfloat_t
     * @param pitch_in  Pitch of @a in, in number of @a T elements.
     * @param pitch_out Pitch of @a out, in number of @a T elements.
     * @param batch     Number of contiguous batches to process.
     * @param stream    Stream on which to enqueue this function.
     * @see Noa::Fourier::crop for more details about the other input arguments.
     * @warning This function runs asynchronously with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void crop(const T* in, size3_t shape_in, size_t pitch_in,
                       T* out, size3_t shape_out, size_t pitch_out,
                       uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::crop. See overload above.
    template<typename T>
    NOA_IH void crop(const T* in, size3_t shape_in, T* out, size3_t shape_out, uint batch, Stream& stream) {
        crop(in, shape_in, shape_in.x / 2 + 1, out, shape_out, shape_out.x / 2 + 1, batch, stream);
    }

    /// CUDA version of Noa::Fourier::cropFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void cropFull(const T* in, size3_t shape_in, size_t pitch_in,
                           T* out, size3_t shape_out, size_t pitch_out,
                           uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::cropFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void cropFull(const T* in, size3_t shape_in, T* out, size3_t shape_out, uint batch, Stream& stream) {
        cropFull(in, shape_in, shape_in.x, out, shape_out, shape_out.x, batch, stream);
    }

    /// CUDA version of Noa::Fourier::pad. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void pad(const T* in, size3_t shape_in, size_t pitch_in,
                      T* out, size3_t shape_out, size_t pitch_out,
                      uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::pad. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void pad(const T* in, size3_t shape_in, T* out, size3_t shape_out, uint batch, Stream& stream) {
        pad(in, shape_in, shape_in.x / 2 + 1, out, shape_out, shape_out.x / 2 + 1, batch, stream);
    }

    /// CUDA version of Noa::Fourier::padFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void padFull(const T* in, size3_t shape_in, size_t pitch_in,
                          T* out, size3_t shape_out, size_t pitch_out,
                          uint batch, Stream& stream);

    /// CUDA version of Noa::Fourier::padFull. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void padFull(const T* in, size3_t shape_in, T* out, size3_t shape_out, uint batch, Stream& stream) {
        padFull(in, shape_in, shape_in.x, out, shape_out, shape_out.x, batch, stream);
    }
}
