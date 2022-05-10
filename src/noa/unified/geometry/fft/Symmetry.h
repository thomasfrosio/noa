#pragma once

#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"
#include "noa/unified/geometry/fft/Transform.h"

namespace noa::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 2D FFT to transform.
    /// \param[out] output      Non-redundant transformed 2D FFT.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] shift        Rightmost 2D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t shift,
                      float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Non-redundant 3D FFT to transform.
    /// \param[out] output      Non-redundant transformed 3D FFT.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param[in] shift        Rightmost 3D real-space forward shift to apply (as phase shift) after the symmetry.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    ///
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p input can be on any device, including the CPU.\n
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void symmetrize3D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float3_t shift,
                      float cutoff = 0.5f, InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);
}

#define NOA_UNIFIED_FFT_SYMMETRY_
#include "noa/unified/geometry/fft/Symmetry.inl"
#undef NOA_UNIFIED_FFT_SYMMETRY_
