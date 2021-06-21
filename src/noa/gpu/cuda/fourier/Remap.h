/// \file noa/gpu/cuda/fourier/Remap.h
/// \brief Remap functions (e.g. fftshift).
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fourier {
    /// CUDA version of noa::fourier::hc2h. The same features and restrictions apply to this function.
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
    NOA_HOST void hc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::hc2h. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void hc2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        hc2h(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::h2hc. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void h2hc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::h2hc. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void h2hc(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        h2hc(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::fc2f. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void fc2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::fc2f. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void fc2f(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        fc2f(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::f2fc. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void f2fc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::f2fc. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void f2fc(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        f2fc(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::f2h. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void f2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                    size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::f2h. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void f2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        f2h(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::h2f. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void h2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::h2f. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void h2f(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        h2f(inputs, shape.x / 2 + 1, outputs, shape.x, shape, batches, stream);
    }

    /// CUDA version of noa::fourier::fc2h. The same features and restrictions apply to this function.
    template<typename T>
    NOA_HOST void fc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// CUDA version of noa::fourier::fc2h. The same features and restrictions apply to this function.
    template<typename T>
    NOA_IH void fc2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        fc2h(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }
}
