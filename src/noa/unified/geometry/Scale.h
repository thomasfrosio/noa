#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Applies one or multiple 2D shrinking/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \param[in] input            Input 2D array.
    /// \param[out] output          Output 2D array.
    /// \param[in] scaling_factors  Rightmost forward scaling factors. One per output batch.
    /// \param[in] scaling_centers  Rightmost scaling centers. One per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 const Array<float2_t>& scaling_factors, const Array<float2_t>& scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one or multiple 2D shrinking/stretching.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same rotation for all batches.
    template<bool PREFILTER = true, typename T>
    void scale2D(const Array<T>& input, const Array<T>& output,
                 float2_t scaling_factors, float2_t scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one or multiple 3D shrinking/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \param[in] input            Input 3D array.
    /// \param[out] output          Output 3D array.
    /// \param[in] scaling_factors  Rightmost forward scaling factors. One per output batch.
    /// \param[in] scaling_centers  Rightmost scaling centers. One per output batch.
    /// \param interp_mode          Filter mode. See InterpMode.
    /// \param border_mode          Address mode. See BorderMode.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/unified/geometry/Transform.h" for more details on the input and output parameters.
    /// \see "noa/common/geometry/Geometry.h" for more details on the conventions used for transformations.
    /// \note If the output is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The third-most and innermost dimension of the input should be contiguous.\n
    ///         - If pre-filtering is not required, the input array can be on the CPU.
    ///           Otherwise, should be on the same device as the output.\n
    ///         - In-place transformation (\p input == \p output) is always allowed.\n
    ///         - \p scaling_factors and \p scaling_centers should be accessible by the CPU.\n
    ///         - \p border_mode is limited to BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///           The last two are only supported with \p interp_mode set to INTER_NEAREST or INTER_LINEAR_FAST.\n
    template<bool PREFILTER = true, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 const Array<float3_t>& scaling_factors, const Array<float3_t>& scaling_centers,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});

    /// Applies one 3D rotation to a (batched) array.
    /// \see This function is has the same features and limitations than the overload above,
    ///      but is using the same rotation for all batches.
    template<bool PREFILTER = true, typename T>
    void scale3D(const Array<T>& input, const Array<T>& output,
                 float3_t scaling_factor, float3_t scaling_center,
                 InterpMode interp_mode = INTERP_LINEAR, BorderMode border_mode = BORDER_ZERO, T value = T{0});
}

#define NOA_UNIFIED_SCALE_
#include "noa/unified/geometry/Scale.inl"
#undef NOA_UNIFIED_SCALE_
