/// \file noa/cpu/fft/Resize.h
/// \brief Fourier crop and pad.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// - Centering
//      The "non-centered" layout is used by FFT routines, with the origin (the DC) at index 0.
//      The "centered" layout is often used in files, with the origin in the "middle right" (N/2).
//
// - Redundancy
//      It refers to non-redundant Fourier transforms of real inputs, resulting in transforms with a LOGICAL shapes of
//      {fast, medium, slow} complex elements and PHYSICAL shapes of {fast/2+1, medium, slow} complex elements.
//      Note that with even dimensions, the Nyquist frequency is real and the C2R routines will assume its imaginary
//      part is zero.
//
//  - Example (DC=0)
//      n=8: non-centered, redundant     u=[ 0, 1, 2, 3,-4,-3,-2,-1]     note: frequency -4 is real, -4 = 4
//           non-centered, non-redundant u=[ 0, 1, 2, 3,-4]
//           centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3]
//           centered,     non-redundant u=[ 0, 1, 2, 3,-4]
//
//      n=9  non-centered, redundant     u=[ 0, 1, 2, 3, 4,-4,-3,-2,-1]  note: frequency 4 is complex, -4 = conj(4)
//           non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
//           centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3, 4]
//           centered,     non-redundant u=[ 0, 1, 2, 3, 4]

namespace noa::cpu::fft {
    /// Crops a non-redundant Fourier transform.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    On the \b host. Contiguous input non-centered, non-redundant array.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p input.
    /// \param[out] output  On the \b host. Contiguous output non-centered, non-redundant array.
    /// \param output_shape Logical {fast, medium, slow} shape of \p output.
    ///                     All dimensions should be less or equal than \p input_shape.
    /// \note If \p input_shape and \p output_shape are equal, \p input is copied into \p output.
    /// \note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
    /// \note \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void crop(const T* input, size3_t input_shape, T* outputs, size3_t output_shape);

    /// Crops a redundant Fourier transform.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    On the \b host. Contiguous input non-centered, redundant array.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p input.
    /// \param[out] output  On the \b host. Contiguous output non-centered, redundant array.
    /// \param output_shape Logical {fast, medium, slow} shape of \p output.
    ///                     All dimensions should be less or equal than \p input_shape.
    /// \note If \p input_shape and \p output_shape are equal, \p input is copied into \p output.
    /// \note \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void cropFull(const T* input, size3_t input_shape, T* output, size3_t output_shape);

    /// Pads a non-redundant Fourier transform with zeros.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    On the \b host. Contiguous input non-centered, non-redundant array.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p input.
    /// \param[out] output  On the \b host. Contiguous output non-centered, non-redundant array.
    /// \param output_shape Logical {fast, medium, slow} shape of \p output.
    ///                     All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \note If \p input_shape and \p output_shape are equal, \p input is copied into \p output.
    /// \note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
    /// \note \p input and @a output should not overlap.
    template<typename T>
    NOA_HOST void pad(const T* input, size3_t input_shape, T* output, size3_t output_shape);

    /// Pads a redundant Fourier transform.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \param[in] input    On the \b host. Contiguous input non-centered, redundant array.
    /// \param input_shape  Logical {fast, medium, slow} shape of \p input.
    /// \param[out] output  On the \b host. Contiguous output non-centered, redundant array.
    /// \param output_shape Logical {fast, medium, slow} shape of \p output.
    ///                     All dimensions should be greater or equal than the dimensions of \p input_shape.
    /// \note If \p input_shape and \p output_shape are equal, \p input is copied into \p output.
    /// \note \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void padFull(const T* input, size3_t input_shape, T* output, size3_t output_shape);
}
