/// \file noa/cpu/fourier/Resize.h
/// \brief Fourier crop and pad.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// Centering
// =========
//
// The not-centered format is the layout used by FFT routines, with the origin (the DC) at index 0.
// The centered format is the layout often used in files, with the origin in the middle left (N/2).
//
// Redundancy
// ==========
//
// Refers to non-redundant Fourier transforms of real inputs, resulting into transforms with a LOGICAL shape of
// {fast,medium,slow} real elements having a PHYSICAL shape of {fast/2+1,medium,slow} complex elements.
// Note that with even dimensions, the Nyquist frequency is real and the C2R routines will assume the imaginary
// part is zero.

namespace noa::fourier {
    /// Crops a Fourier transform.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       Input array. Should be not-centered, not-redundant and contiguous.
    /// \param inputs_shape     Logical {fast, medium, slow} shape of \a inputs, in complex elements.
    /// \param[out] outputs     Output array. Will be not-centered, not-redundant and contiguous.
    /// \param outputs_shape    Logical {fast, medium, slow} shape of \a outputs.
    ///                         All dimensions should be less or equal than the dimensions of \a inputs_shape.
    /// \note If \a inputs_shape and \a outputs_shape are equal, \a inputs is copied into \a outputs.
    /// \note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
    /// \note \a inputs and \a outputs should not overlap.
    template<typename T>
    NOA_HOST void crop(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape);

    /// Crops a Fourier transform.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       Input array. Should be not-centered, redundant and contiguous.
    /// \param inputs_shape     Logical and physical {fast, medium, slow} shape of \a inputs.
    /// \param[out] outputs     Output array. Will be not-centered, redundant and contiguous.
    /// \param outputs_shape    Logical and physical {fast, medium, slow} shape of \a outputs.
    ///                         All dimensions should be less or equal than the dimensions of \a inputs_shape.
    /// \note If \a inputs_shape and \a outputs_shape are equal, \a inputs is copied into \a outputs.
    /// \note \a inputs and \a outputs should not overlap.
    template<typename T>
    NOA_HOST void cropFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape);

    /// Pads a Fourier transform with zeros.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       Input array. Should be not-centered, not-redundant and contiguous.
    /// \param inputs_shape     Logical {fast, medium, slow} shape of \a inputs.
    /// \param[out] outputs     Output array. Will be not-centered, not-redundant and contiguous.
    /// \param outputs_shape    Logical {fast, medium, slow} shape of \a outputs.
    ///                         All dimensions should be greater or equal than the dimensions of \a inputs_shape.
    /// \note If \a inputs_shape and \a outputs_shape are equal, \a inputs is copied into \a outputs.
    /// \note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
    /// \note \a inputs and @a outputs should not overlap.
    template<typename T>
    NOA_HOST void pad(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape);

    /// Pads a Fourier transform.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       Input array. Should be not-centered, redundant and contiguous.
    /// \param inputs_shape     Logical and physical {fast, medium, slow} shape of \a inputs.
    /// \param[out] outputs     Output array. Will be not-centered, redundant and contiguous.
    /// \param outputs_shape    Logical and physical {fast, medium, slow} shape of \a outputs.
    ///                         All dimensions should be greater or equal than the dimensions of \a inputs_shape.
    /// \note If \a inputs_shape and \a outputs_shape are equal, \a inputs is copied into \a outputs.
    /// \note \a inputs and \a outputs should not overlap.
    template<typename T>
    NOA_HOST void padFull(const T* inputs, size3_t inputs_shape, T* outputs, size3_t outputs_shape);
}
