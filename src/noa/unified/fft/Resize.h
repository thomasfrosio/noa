#pragma once

#include "noa/unified/Array.h"

namespace noa::fft::details {
    using Remap = ::noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_resize = (traits::is_float_v<T> || traits::is_complex_v<T>) &&
                                     (REMAP == Remap::H2H || REMAP == Remap::F2F ||
                                      REMAP == Remap::HC2HC || REMAP == Remap::FC2FC);
}

namespace noa::fft {
    /// Crops or zero-pads an FFT.
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape BDHW logical shape of \p output.
    ///
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    /// \note The redundant dimension is the width dimension and non-redundant FFT are traversed in the rightmost order.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    void resize(const Array<T>& input, dim4_t input_shape, const Array<T>& output, dim4_t output_shape);

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param output_shape BDHW logical shape of the output.
    ///
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    /// \note The redundant dimension is the width dimension and non-redundant FFT are traversed in the rightmost order.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    [[nodiscard]] Array<T> resize(const Array<T>& input, dim4_t input_shape, dim4_t output_shape);
}

#define NOA_UNIFIED_RESIZE_
#include "noa/unified/fft/Resize.inl"
#undef NOA_UNIFIED_RESIZE_
