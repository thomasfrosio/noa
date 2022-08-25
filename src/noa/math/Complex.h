#pragma once

#include "noa/Array.h"

namespace noa::math {
    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    /// \param[out] imag    Imaginary elements.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const Array<Complex<T>>& input, const Array<T>& real, const Array<T>& imag);

    /// Extracts the real part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void real(const Array<Complex<T>>& input, const Array<T>& real);

    /// Extracts the imaginary part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] imag    Imaginary elements.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void imag(const Array<Complex<T>>& input, const Array<T>& imag);

    /// Fuses the real and imaginary components.
    /// \tparam T       half_t, float, double.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    /// \param output   Complex array.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void complex(const Array<T>& real, const Array<T>& imag, const Array<Complex<T>>& output);
}

#define NOA_UNIFIED_COMPLEX_
#include "noa/math/details/Complex.inl"
#undef NOA_UNIFIED_COMPLEX_
