#pragma once

#include "noa/unified/Array.h"

namespace noa::fft::details {
    using Remap = ::noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_resize = (traits::is_float_v<T> || traits::is_complex_v<T>) &&
                                     (REMAP == Remap::H2H || REMAP == Remap::F2F);
}

namespace noa::fft {
    /// Crops or zero-pads an FFT.
    /// \tparam REMAP       FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape BDHW logical shape of \p output.
    ///                     All dimensions should either be <= or >= than \p input_shape.
    /// \note The batch dimension cannot be resized.
    /// \note The redundant dimension is the width dimension and non-redundant FFT are traversed in the rightmost order.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    void resize(const Array<T>& input, size4_t input_shape, const Array<T>& output, size4_t output_shape);

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param output_shape BDHW logical shape of the output.
    ///                     All dimensions should either be <= or >= than \p input_shape.
    /// \note The batch dimension cannot be resized.
    /// \note The redundant dimension is the width dimension and non-redundant FFT are traversed in the rightmost order.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    Array<T> resize(const Array<T>& input, size4_t input_shape, size4_t output_shape);
}

#define NOA_UNIFIED_RESIZE_
#include "noa/unified/fft/Resize.inl"
#undef NOA_UNIFIED_RESIZE_
