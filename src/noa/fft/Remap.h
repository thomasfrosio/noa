#pragma once

#include "noa/Array.h"

namespace noa::fft {
    /// Remaps FFT(s).
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t or cdouble_t.
    /// \param remap        Remapping operation. See noa::fft::Remap for more details.
    /// \param[in] input    Input FFT to remap.
    /// \param[out] output  Remapped FFT.
    /// \param shape        BDHW logical shape.
    /// \note If \p remap is \c H2HC, \p input can be equal to \p output, iff the height and depth is even or 1.
    /// \note The redundant dimension is the width dimension and non-redundant FFT are traversed in the rightmost order.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void remap(Remap remap, const Array<T>& input, const Array<T>& output, size4_t shape);

    /// Remaps FFT(s).
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t or cdouble_t.
    /// \param remap        Remapping operation. See noa::fft::Remap for more details.
    /// \param[in] input    FFT to remap.
    /// \param shape        BDHW logical shape.
    /// \return Remapped FFT(s).
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    Array<T> remap(Remap remap, const Array<T>& input, size4_t shape);
}

#define NOA_UNIFIED_REMAP_
#include "noa/fft/details/Remap.inl"
#undef NOA_UNIFIED_REMAP_
