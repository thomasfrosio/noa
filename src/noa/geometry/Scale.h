#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/Array.h"

namespace noa::geometry::details {
    template<int NDIM, typename T, typename S, typename C>
    constexpr bool is_valid_scale_v =
            traits::is_any_v<T, float, cfloat_t, double, cdouble_t> &&
            ((NDIM == 2 && traits::is_any_v<S, float2_t, shared_t<float2_t[]>> && traits::is_any_v<C, float2_t, shared_t<float2_t[]>>) ||
             (NDIM == 3 && traits::is_any_v<S, float3_t, shared_t<float3_t[]>> && traits::is_any_v<C, float3_t, shared_t<float3_t[]>>));
}

namespace noa::geometry {
    /// Applies one or multiple 2D shrinking/stretching.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam S                   float2_t or Array<float2_t>.
    /// \tparam C                   float2_t or Array<float2_t>.
    /// \param[in] input            Input 2D array.
    /// \param[out] output          Output 2D array.
    /// \param[in] scaling_factors  HW forward scaling factors. One, or if an array is entered, one per output batch.
    /// \param[in] scaling_centers  HW scaling centers. One, or if an array is entered, one per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \param prefilter            Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    ///
    /// \see "noa/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p scaling_factors and \p scaling_centers can be on any device as long as
    ///           they are dereferenceable by the CPU.\n
    ///         - All border modes are supported, except BORDER_NOTHING.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and its width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<typename T, typename S, typename C, typename = std::enable_if_t<details::is_valid_scale_v<2, T, S, C>>>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                 T value = T{0}, bool prefilter = true);

    /// Applies one or multiple 3D shrinking/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam S                   float3_t or Array<float3_t>.
    /// \tparam C                   float3_t or Array<float3_t>.
    /// \param[in] input            Input 3D array.
    /// \param[out] output          Output 3D array.
    /// \param[in] scaling_factors  DHW forward scaling factors. One, or if an array is entered, one per output batch.
    /// \param[in] scaling_centers  DHW scaling centers. One, or if an array is entered, one per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the CPU:\n
    ///         - \p input and \p output should not overlap.\n
    ///         - \p input and \p output should be on the same device.\n
    ///         - \p scaling_factors and \p scaling_centers can be on any device as long as
    ///           they are dereferenceable by the CPU.\n
    ///         - All border modes are supported, except BORDER_NOTHING.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and its height and width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<typename T, typename S, typename C, typename = std::enable_if_t<details::is_valid_scale_v<3, T, S, C>>>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 const S& scaling_factors, const C& scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO,
                 T value = T{0}, bool prefilter = true);
}

#define NOA_UNIFIED_SCALE_
#include "noa/geometry/details/Scale.inl"
#undef NOA_UNIFIED_SCALE_
