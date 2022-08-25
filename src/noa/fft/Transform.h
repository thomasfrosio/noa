#pragma once

#include "noa/Array.h"

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = NORM_FORWARD;

    /// Returns the optimum even size, greater or equal than \p size.
    /// \note A optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d).
    /// \note If \p size is >16800, this function will simply return the next even number and will not necessarily
    ///       satisfy the aforementioned requirements.
    inline size_t nextFastSize(size_t size);

    /// Returns the next optimum BDHW shape.
    /// \note Dimensions of size 0 or 1 are ignored as well as the batch dimension, e.g. {3,1,53,53}
    ///       is rounded up to {3,1,54,54}.
    template<typename T>
    Int4<T> nextFastShape(Int4<T> shape);

    /// Returns an alias of \p input that fits the corresponding real-space array.
    /// \tparam T           float, double.
    /// \param[in] input    Non-redundant FFT(s) to alias.
    /// \param shape        BDHW logical shape of \p input.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    Array<T> alias(const Array<Complex<T>>& input, size4_t shape);

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
#include "noa/fft/details/Transform.inl"
#undef NOA_UNIFIED_TRANSFORM_
