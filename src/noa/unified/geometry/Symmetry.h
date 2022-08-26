#pragma once

#include "noa/common/geometry/Symmetry.h"
#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Symmetrizes the 2D (batched) input array.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       HW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param prefilter    Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and its width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void symmetrize2D(const Array<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Symmetrizes the 2D (batched) input array.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void symmetrize2D(const Texture<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float2_t center,
                      bool normalize = true);

    /// Symmetrizes the 3D (batched) input array.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] symmetry Symmetry operator.
    /// \param center       DHW center of the symmetry.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param prefilter    Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \param normalize    Whether \p output should be normalized to have the same range as \p input.
    ///                     If false, output values end up being scaled by the symmetry count.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and its height and width dimensions should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void symmetrize3D(const Array<T[]>& input, const Array<T[]>& output,
                      const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode = INTERP_LINEAR, bool prefilter = true, bool normalize = true);

    /// Symmetrizes the 2D (batched) input array.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t, double, cdouble_t>>>
    void symmetrize3D(const Texture<T>& input, const Array<T>& output,
                      const Symmetry& symmetry, float3_t center,
                      bool normalize = true);
}

#define NOA_UNIFIED_SYMMETRY_
#include "noa/unified/geometry/Symmetry.inl"
#undef NOA_UNIFIED_SYMMETRY_
