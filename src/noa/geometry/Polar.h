#pragma once

#include "noa/Array.h"

namespace noa::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] cartesian    Input 2D cartesian array to interpolate onto the new coordinate system.
    /// \param[out] polar       Transformed 2D array on the (log-)polar grid.
    ///                         The width dimension is the radius rho, from and to \p radius_range.
    ///                         The height dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center HW transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle to transform, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    /// \param prefilter        Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note If \p polar is on the CPU:\n
    ///         - \p cartesian and \p polar should not overlap.\n
    ///         - \p cartesian and \p polar should be on the same device.\n
    /// \note If \p polar is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p cartesian should be in the rightmost order and its width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, \p cartesian can be on the CPU.
    ///           Otherwise, should be on the same device as \p polar.\n
    ///         - In-place transformation is always allowed.\n
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const Array<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR, bool prefilter = true);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const Texture<T>& cartesian, const Array<T>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false);

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] polar        Input 2D polar array to interpolate onto the new coordinate system.
    ///                         The width is the radius rho, from and to \p radius_range.
    ///                         The height is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian   Transformed 2D array on the cartesian grid.
    /// \param cartesian_center HW transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    /// \param prefilter        Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note If \p cartesian is on the CPU:\n
    ///         - \p polar and \p cartesian should not overlap.\n
    ///         - \p polar and \p cartesian should be on the same device.\n
    /// \note If \p cartesian is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p polar should be in the rightmost order and its width dimension should be contiguous.\n
    ///         - If pre-filtering is not required, \p polar can be on the CPU.
    ///           Otherwise, should be on the same device as \p cartesian.\n
    ///         - In-place transformation is always allowed.\n
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const Array<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR, bool prefilter = true);

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const Texture<T>& polar, const Array<T>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false);
}

#define NOA_UNIFIED_POLAR_
#include "noa/geometry/details/Polar.inl"
#undef NOA_UNIFIED_POLAR_
