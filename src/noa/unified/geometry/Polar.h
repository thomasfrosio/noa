#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam Value               float, double, cfloat_t or cdouble_t.
    /// \param[in,out] cartesian    Input 2D cartesian array to interpolate onto the new coordinate system.
    ///                             Can be overwritten, depending on \p prefilter.
    /// \param[out] polar           Transformed 2D array on the (log-)polar grid.
    ///                             The width dimension is the radius rho, from and to \p radius_range.
    ///                             The height dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center     HW transformation center.
    /// \param radius_range         Radius [start,end] range of the bounding circle to transform, in pixels.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether log-polar coordinates should be computed instead.
    /// \param interp               Interpolation method used to interpolate the values onto the new grid.
    ///                             Out-of-bounds elements are set to zero.
    /// \param prefilter            Whether or not the input should be prefiltered in-place.
    ///                             Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const Array<Value>& cartesian, const Array<Value>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR, bool prefilter = true);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const Texture<Value>& cartesian, const Array<Value>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false);

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// \tparam Value           float, double, cfloat_t or cdouble_t.
    /// \param[in,out] polar    Input 2D polar array to interpolate onto the new coordinate system.
    ///                         Can be overwritten, depending on \p prefilter.
    ///                         The width is the radius rho, from and to \p radius_range.
    ///                         The height is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian   Transformed 2D array on the cartesian grid.
    /// \param cartesian_center HW transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    ///                         Out-of-bounds elements are set to zero.
    /// \param prefilter        Whether or not the input should be prefiltered in-place.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const Array<Value>& polar, const Array<Value>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR, bool prefilter = true);

    /// Transforms 2D array(s) from (log-)polar to cartesian coordinates.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const Texture<Value>& polar, const Array<Value>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log = false);
}
