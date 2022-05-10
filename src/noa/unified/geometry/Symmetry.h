#pragma once

#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Symmetrizes the 2D (batched) input array.
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       Rightmost center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T>
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);

    /// Symmetrizes the 3D (batched) input array.
    /// \tparam PREFILTER   Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       Rightmost center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<bool PREFILTER = true, typename T>
    void symmetrize3D(const Array<T[]>& input, const Array<T[]>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool normalize = true);
}

#define NOA_UNIFIED_SYMMETRY_
#include "noa/unified/geometry/Symmetry.inl"
#undef NOA_UNIFIED_SYMMETRY_
