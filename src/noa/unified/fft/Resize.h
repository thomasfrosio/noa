#pragma once

#include "noa/unified/Array.h"

namespace noa::fft {
    /// Crops or zero-pads an FFT.
    /// \tparam REMAP       FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  Rightmost logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape Rightmost logical shape of \p output.
    ///                     All dimensions should either be <= or >= than \p input_shape.
    /// \note The outermost dimension cannot be resized, i.e. \p input_shape[0] == \p output_shape[0].
    template<Remap REMAP, typename T, typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void resize(const Array<T>& input, size4_t input_shape, const Array<T>& output, size4_t output_shape);
}

#define NOA_UNIFIED_RESIZE_
#include "noa/unified/fft/Resize.inl"
#undef NOA_UNIFIED_RESIZE_
