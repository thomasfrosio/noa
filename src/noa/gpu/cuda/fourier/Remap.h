/// \file noa/gpu/cuda/fourier/Remap.h
/// \brief Remap functions (e.g. fftshift).
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fourier {
    /// CUDA version of noa::fourier::HC2H. The same features and restrictions apply to this function.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
    /// \param inputs_pitch     Pitch of \a inputs, in number of \a T elements.
    /// \param[out] outputs     The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
    /// \param outputs_pitch    Pitch of \a outputs, in number of \a T elements.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         Both \a inputs and \a outputs should be on the device associated with this stream.
    template<typename T>
    NOA_HOST void HC2H(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::HC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void HC2H(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        HC2H(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2HC(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::H2HC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2HC(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        H2HC(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void FC2F(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::FC2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void FC2F(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        FC2F(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void F2FC(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::F2FC. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2FC(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        F2FC(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void F2H(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                    size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::F2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void F2H(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        F2H(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void H2F(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::H2F. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void H2F(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        H2F(inputs, shape.x / 2 + 1, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::FC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void FC2H(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::FC2H. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void FC2H(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        FC2H(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }
}
