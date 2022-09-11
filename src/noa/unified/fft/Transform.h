#pragma once

#include "noa/unified/Array.h"

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = NORM_FORWARD;



    /// Computes the forward R2C transform of (batched) 2D/3D array(s) or column/row vector(s).
    /// \tparam T           float, double.
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered FFT(s).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the row dimension.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void r2c(const Array<T>& input, const Array<Complex<T>>& output, Norm norm = NORM_DEFAULT);

    /// Computes the forward R2C transform of (batched) 2D/3D array(s) or column/row vector(s).
    /// \tparam T           float, double.
    /// \param[in] input    Real space array.
    /// \param norm         Normalization mode.
    /// \return Non-redundant non-centered FFT(s).
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    Array<Complex<T>> r2c(const Array<T>& input, Norm norm = NORM_DEFAULT);

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param[out] output      Real space array.
    /// \param norm             Normalization mode.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the row dimension.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void c2r(const Array<Complex<T>>& input, const Array<T>& output, Norm norm = NORM_DEFAULT);

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param shape            BDHW logical shape of \p input.
    /// \param norm             Normalization mode.
    /// \return Real space array.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    Array<T> c2r(const Array<Complex<T>>& input, size4_t shape, Norm norm = NORM_DEFAULT);

    /// Computes the C2C transform.
    /// \tparam T           float, double.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void c2c(const Array<Complex<T>>& input, const Array<Complex<T>>& output, Sign sign, Norm norm = NORM_DEFAULT);

    /// Computes the C2C transform.
    /// \tparam T           float, double.
    /// \param[in] input    Input complex data.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm         Normalization mode.
    /// \return Non-centered FFT(s).
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    Array<Complex<T>> c2c(const Array<Complex<T>>& input, Sign sign, Norm norm = NORM_DEFAULT);
}

#define NOA_UNIFIED_TRANSFORM_
#include "noa/unified/fft/Transform.inl"
#undef NOA_UNIFIED_TRANSFORM_
