#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Symmetry.h"

namespace noa::cpu::transform::fft {
    using Remap = ::noa::fft::Remap;
    using Symmetry = ::noa::transform::Symmetry;

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param shape            Logical {fast, medium} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \note \p input and \p output should not overlap.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize2D(const T* input, T* output, size2_t shape,
                               const Symmetry& symmetry, float2_t shift,
                               float max_frequency, InterpMode interp_mode, bool normalize);

    /// Symmetrizes a non-redundant FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Non-redundant FFT to symmetrize.
    /// \param[out] output      On the \b host. Non-redundant symmetrized FFT.
    /// \param shape            Logical {fast, medium, slow} shape, in \p T elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator to apply.
    /// \param max_frequency    Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are left unchanged.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \note \p input and \p output should not overlap.
    template<Remap REMAP, typename T>
    NOA_HOST void symmetrize3D(const T* input, T* output, size3_t shape,
                               const Symmetry& symmetry, float3_t shift,
                               float max_frequency, InterpMode interp_mode, bool normalize);
}
