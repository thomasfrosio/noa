#pragma once

#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] cartesian    On the \b host. Input array to interpolate onto the new coordinate system.
    /// \param cartesian_stride Rightmost stride of \p cartesian.
    /// \param cartesian_shape  Rightmost shape of \p cartesian.
    /// \param[out] polar       On the \b host. Transformed array on the (log-)polar grid.
    /// \param polar_stride     Rightmost stride of \p polar.
    /// \param polar_shape      Rightmost shape of \p polar.
    ///                         The innermost dimension is the radius rho, from and to \p radius_range.
    ///                         The second-most dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center Rightmost transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle to transform, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] polar        On the \b host. Input array to interpolate onto the new coordinate system.
    /// \param polar_stride     Rightmost stride of \p polar.
    /// \param polar_shape      Rightmost shape of \p output.
    ///                         The innermost dimension is the radius rho, from and to \p radius_range.
    ///                         The second-most dimension is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian   On the \b host. Transformed array on the cartesian grid.
    /// \param cartesian_stride Rightmost stride of \p cartesian.
    /// \param cartesian_shape  Rightmost shape of \p cartesian.
    /// \param cartesian_center Rightmost transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool PREFILTER = true, typename T,
             typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    void polar2cartesian(const shared_t<T[]>& polar, size4_t polar_stride, size4_t polar_shape,
                         const shared_t<T[]>& cartesian, size4_t cartesian_stride, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);
}
