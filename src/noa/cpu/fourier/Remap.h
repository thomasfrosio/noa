/// \file noa/cpu/fourier/Remap.h
/// \brief Remap FFTs (e.g. fftshift).
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// F = redundant, non-centered
// FC = redundant, centered
// H = non-redundant, non-centered
// HC = non-redundant, centered
// TODO h2fc is missing, since it seems a bit more complicated and it would be surprising if we ever use it.
//      Moreover, the same can be achieved with h2f and then f2fc.

namespace noa::fourier {
    /// Remaps "half centered to half", i.e. file format to FFT format.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous non-redundant, centered transform.
    /// \param[out] output  On the \b host. Contiguous non-redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void hc2h(const T* input, T* output, size3_t shape);

    /// Remaps "half to half centered", i.e. FFT format to file format.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous non-redundant, non-centered transform.
    /// \param[out] output  On the \b host. Contiguous non-redundant, centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void h2hc(const T* input, T* output, size3_t shape);

    /// Remaps "full centered to full", i.e. iFFTShift.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous redundant, centered transform.
    /// \param[out] output  On the \b host. Contiguous redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void fc2f(const T* input, T* output, size3_t shape);

    /// Remaps "full to full centered", i.e. FFTShift.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous redundant, non-centered transform.
    /// \param[out] output  On the \b host. Contiguous redundant, centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void f2fc(const T* input, T* output, size3_t shape);

    /// Remaps "half to full", i.e. applies the hermitian symmetry to generate the redundant transform.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous non-redundant, non-centered transform.
    /// \param[out] output  On the \b host. Contiguous redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void h2f(const T* input, T* output, size3_t shape);

    /// Remaps "full to half".
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous redundant, non-centered transform.
    /// \param[out] output  On the \b host. Contiguous non-redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void f2h(const T* input, T* output, size3_t shape);

    /// Remaps "full centered to half".
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous redundant, centered transform.
    /// \param[out] output  On the \b host. Contiguous non-redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p input and \p output.
    /// \note               \p input and \p output should not overlap.
    template<typename T>
    NOA_HOST void fc2h(const T* input, T* output, size3_t shape);
}
