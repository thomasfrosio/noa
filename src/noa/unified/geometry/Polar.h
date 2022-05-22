#pragma once

#include "noa/unified/Array.h"

namespace noa::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] cartesian    Input 2D cartesian array to interpolate onto the new coordinate system.
    /// \param[out] polar       Transformed 2D array on the (log-)polar grid.
    ///                         The innermost dimension is the radius rho, from and to \p radius_range.
    ///                         The second-most dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center Rightmost transformation center.
    /// \param radius_range     Radius [start,end) range of the bounding circle to transform, in pixels.
    /// \param angle_range      Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note If \p polar is on the CPU:\n
    ///         - \p cartesian and \p polar should not overlap.\n
    ///         - \p cartesian and \p polar should be on the same device.\n
    /// \note If \p polar is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of the cartesian input should be contiguous.\n
    ///         - If pre-filtering is not required, \p cartesian can be on the CPU.
    ///           Otherwise, should be on the same device as \p polar.\n
    ///         - In-place transformation is always allowed.\n
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const Array<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] polar        Input 2D polar array to interpolate onto the new coordinate system.
    ///                         The innermost dimension is the radius rho, from and to \p radius_range.
    ///                         The second-most dimension is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian   Transformed 2D array on the cartesian grid.
    /// \param cartesian_center Rightmost transformation center.
    /// \param radius_range     Radius [start,end) range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end) range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note If \p cartesian is on the CPU:\n
    ///         - \p polar and \p cartesian should not overlap.\n
    ///         - \p polar and \p cartesian should be on the same device.\n
    /// \note If \p cartesian is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - The innermost dimension of \p polar should be contiguous.\n
    ///         - If pre-filtering is not required, \p polar can be on the CPU.
    ///           Otherwise, should be on the same device as \p cartesian.\n
    ///         - In-place transformation is always allowed.\n
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const Array<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR);
}

#define NOA_UNIFIED_POLAR_
#include "noa/unified/geometry/Polar.inl"
#undef NOA_UNIFIED_POLAR_
