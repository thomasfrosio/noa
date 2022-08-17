#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/unified/Array.h"

namespace noa::geometry::details {
    template<int NDIM, typename T, typename M>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<M, float2_t, Array<float2_t>>) ||
             (NDIM == 3 && traits::is_any_v<M, float3_t, Array<float3_t>>));
}

namespace noa::geometry {
    /// Applies one or multiple 2D shifts.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \tparam S           float2_t or Array<float2_t>.
    /// \param[in] input    Input 2D array.
    /// \param[out] output  Output 2D array.
    /// \param[in] shifts   HW forward shifts. One, or if an array is entered, one per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter    Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p shifts can be on any device as long as they are dereferenceable by the CPU.\n
    ///         - All border modes are supported, except BORDER_NOTHING.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p shifts can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<2, T, S>>>
    void shift2D(const Array<T>& input, const Array<T>& output, const S& shifts,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                 T value = T{0}, bool prefilter = true);

    /// Applies one or multiple 2D shifts.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<2, T, S>>>
    void shift2D(const Texture<T>& input, const Array<T>& output, const S& shifts);

    /// Applies one or multiple 3D shifts.
    /// \tparam T           float, double, cfloat_t or cdouble_t.
    /// \tparam S           float3_t or Array<float3_t>.
    /// \param[in] input    Input 3D array.
    /// \param[out] output  Output 3D array.
    /// \param[in] shifts   DHW forward shifts. One, or if an array is entered, one per output batch.
    /// \param interp_mode  Filter mode. See InterpMode.
    /// \param border_mode  Address mode. See BorderMode.
    /// \param value        Constant value to use for out-of-bounds coordinates.
    ///                     Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter    Whether or not the input should be prefiltered.
    ///                     Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p shifts can be on any device as long as they are dereferenceable by the CPU.\n
    ///         - All border modes are supported, except BORDER_NOTHING.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the height and width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p shifts can be on any device, including the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<3, T, S>>>
    void shift3D(const Array<T>& input, const Array<T>& output, const S& shifts,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                 T value = T{0}, bool prefilter = true);

    /// Applies one or multiple 3D shifts.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    template<typename T, typename S, typename = std::enable_if_t<details::is_valid_shift_v<3, T, S>>>
    void shift3D(const Texture<T>& input, const Array<T>& output, const S& shifts);
}

#define NOA_UNIFIED_SHIFT_
#include "noa/unified/geometry/Shift.inl"
#undef NOA_UNIFIED_SHIFT_
