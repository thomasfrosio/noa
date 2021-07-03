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
    /// \tparam T           Real or complex.
    /// \param[in] in       The first getElementsFFT(shape) elements will be read.
    /// \param[out] out     The first getElementsFFT(shape) elements will be written.
    /// \param shape        Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note               \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void hc2h(const T* in, T* out, size3_t shape);

    /// Remaps "half to half centered", i.e. FFT format to file format.
    /// \tparam T           Real or complex.
    /// \param[in] in       The first getElementsFFT(shape) elements will be read.
    /// \param[out] out     The first getElementsFFT(shape) elements will be written.
    /// \param shape        Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note               \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void h2hc(const T* in, T* out, size3_t shape);

    /// Remaps "full centered to full", i.e. iFFTShift.
    /// \tparam T       Real or complex.
    /// \param shape    Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note           \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void fc2f(const T* in, T* out, size3_t shape);

    /// Remaps "full to full centered", i.e. FFTShift.
    /// \tparam T       Real or complex.
    /// \param shape    Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note           \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void f2fc(const T* in, T* out, size3_t shape);

    /// Remaps "half to full", i.e. applies the hermitian symmetry to generate the non-redundant transform.
    /// \tparam T           Real or complex.
    /// \param[in] in       Half transform. The first getElementsFFT(shape) elements will be read.
    /// \param[out] out     Full transform. The first getElements(shape) elements will be written.
    /// \param shape        Logical {fast, medium, slow} shape of \a in and \a out.
    ///
    /// \note \a in and \a out should not overlap.
    /// \note If \a T is complex, the complex conjugate is computed to generate the redundant elements.
    template<typename T>
    NOA_HOST void h2f(const T* in, T* out, size3_t shape);

    /// Remaps "full to half".
    /// \tparam T           Real or complex.
    /// \param[in] in       Full transform. The first getElements(shape) elements will be read.
    /// \param[out] out     Half transform. The first getElementsFFT(shape) elements will be written.
    /// \param shape        Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note               \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void f2h(const T* in, T* out, size3_t shape);

    /// Remaps "full centered to half".
    /// \param[in] in       Full transform. The first getElements(shape) elements will be read.
    /// \param[out] out     Half transform. The first getElementsFFT(shape) elements will be written.
    /// \param shape        Logical {fast, medium, slow} shape of \a in and \a out.
    /// \note               \a in and \a out should not overlap.
    template<typename T>
    NOA_HOST void fc2h(const T* in, T* out, size3_t shape);
}
