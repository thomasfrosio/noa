/// \file noa/gpu/cuda/fourier/Remap.h
/// \brief Remap functions (e.g. fftshift).
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::fourier {
    /// Remaps "half centered to half", i.e. file format to FFT format.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant, centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant, non-centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void hc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// Remaps "half centered to half", i.e. file format to FFT format. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void hc2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        hc2h(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// Remaps "half to half centered", i.e. FFT format to file format.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant, non-centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant, centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void h2hc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// Remaps "half to half centered", i.e. FFT format to file format. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void h2hc(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        h2hc(inputs, shape.x / 2 + 1, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// Remaps "full centered to full", i.e. iFFTShift.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant, centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Redundant, non-centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void fc2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    ///Remaps "full centered to full", i.e. iFFTShift. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void fc2f(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        fc2f(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// Remaps "full to full centered", i.e. FFTShift.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant, non-centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Redundant, centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void f2fc(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// Remaps "full to full centered", i.e. FFTShift. Version for contiguous layouts.
    template<typename T>
    NOA_IH void f2fc(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        f2fc(inputs, shape.x, outputs, shape.x, shape, batches, stream);
    }

    /// Remaps "full to half".
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant, non-centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant, centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void f2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                    size3_t shape, uint batches, Stream& stream);

    /// Remaps "full to half". Version for contiguous layouts.
    template<typename T>
    NOA_IH void f2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        f2h(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }

    /// Remaps "half to full", i.e. applies the hermitian symmetry to generate the redundant transform.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Non-redundant, non-centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Redundant, non-centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void h2f(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);

    /// Remaps "half to full". Version for contiguous layouts.
    template<typename T>
    NOA_IH void h2f(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        h2f(inputs, shape.x / 2 + 1, outputs, shape.x, shape, batches, stream);
    }

    /// Remaps "full centered to half".
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Redundant, centered transform.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant, non-centered transform.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void fc2h(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                       size3_t shape, uint batches, Stream& stream);

    /// Remaps "full centered to half". Version for contiguous layouts.
    template<typename T>
    NOA_IH void fc2h(const T* inputs, T* outputs, size3_t shape, uint batches, Stream& stream) {
        fc2h(inputs, shape.x, outputs, shape.x / 2 + 1, shape, batches, stream);
    }
}
