#pragma once

#include "noa/unified/Array.h"

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

namespace noa::math {
    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    /// \returns A pair of arrays with the real and imaginary elements.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    std::pair<Array<T>, Array<T>> decompose(const Array<Complex<T>>& input);

    /// Extracts the real part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    Array<T> real(const Array<Complex<T>>& input);

    /// Extracts the imaginary part of complex numbers.
    /// \tparam T           half_t, float, double.
    /// \param[in] input    Complex array to decompose.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    Array<T> imag(const Array<Complex<T>>& input);

    /// Fuses the real and imaginary components.
    /// \tparam T       half_t, float, double.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    Array<Complex<T>> complex(const Array<T>& real, const Array<T>& imag);
}

#define NOA_UNIFIED_COMPLEX_
#include "noa/unified/math/Complex.inl"
#undef NOA_UNIFIED_COMPLEX_
