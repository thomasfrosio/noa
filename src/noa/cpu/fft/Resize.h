/// \file noa/cpu/fft/Resize.h
/// \brief Fourier crop and pad.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::fft {
    /// Crops non-redundant FFTs.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs On the \b host. Cropped non-redundant non-centered FFT.
    /// \param output_shape Logical {fast, medium, slow} shape of \p outputs.
    ///                     All dimensions should be less or equal than \p input_shape.
    /// \param batches      Number of contiguous batches to compute.
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void crop(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches);

    /// Crops redundant FFTs.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Redundant, non-centered FFT to crop.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs On the \b host. Cropped redundant non-centered FFT.
    /// \param output_shape Logical {fast, medium, slow} shape of \p outputs.
    ///                     All dimensions should be less or equal than \p input_shape.
    /// \param batches      Number of contiguous batches to compute.
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void cropFull(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches);

    /// Pads non-redundant FFTs with zeros.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-centered non-redundant FFT.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs On the \b host. Padded non-centered non-redundant FFT.
    /// \param output_shape Logical {fast, medium, slow} shape of \p outputs.
    ///                     All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \param batches      Number of contiguous batches to compute.
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void pad(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches);

    /// Pads redundant FFTs with zeros.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Non-centered redundant FFT.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p inputs.
    /// \param[out] outputs On the \b host. Padded non-centered redundant FFT.
    /// \param output_shape Logical {fast, medium, slow} shape of \p outputs.
    ///                     All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \param batches      Number of contiguous batches to compute.
    /// \note See noa::fft::Layout for more details about FFT layouts.
    /// \note \p inputs and \p outputs should not overlap.
    template<typename T>
    NOA_HOST void padFull(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape, size_t batches);
}
