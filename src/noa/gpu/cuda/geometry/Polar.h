#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry {
    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] cartesian        Input array to interpolate onto the new coordinate system.
    ///                             If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param cartesian_strides    BDHW strides of \p cartesian.
    /// \param cartesian_shape      BDHW shape of \p cartesian.
    /// \param[out] polar           On the \b device. Transformed array on the (log-)polar grid. Can be equal to \p cartesian.
    /// \param polar_strides        BDHW strides of \p polar.
    /// \param polar_shape          BDHW shape of \p polar.
    ///                             The width dimension is the radius rho, from and to \p radius_range.
    ///                             The height dimension is the angle phi, from and to \p angle_range.
    /// \param cartesian_center     HW transformation center.
    /// \param radius_range         Radius [start,end] range of the bounding circle to transform, in pixels.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether log-polar coordinates should be computed instead.
    /// \param interp               Interpolation method used to interpolate the values onto the new grid.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_strides, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] polar            Input array to interpolate onto the new coordinate system.
    ///                             If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param polar_strides        BDHW strides of \p polar.
    /// \param polar_shape          BDHW shape of \p output.
    ///                             The width is the radius rho, from and to \p radius_range.
    ///                             The height is the angle phi, from and to \p angle_range.
    /// \param[out] cartesian       On the \b device. Transformed array on the cartesian grid. Can be equal to \p polar.
    /// \param cartesian_strides    BDHW strides of \p cartesian.
    /// \param cartesian_shape      BDHW shape of \p cartesian.
    /// \param cartesian_center     HW transformation center.
    /// \param radius_range         Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether this is a log-polar coordinates system.
    /// \param interp               Interpolation method used to interpolate the values onto the new grid.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void polar2cartesian(const shared_t<T[]>& polar, size4_t polar_strides, size4_t polar_shape,
                         const shared_t<T[]>& cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);
}

namespace noa::cuda::geometry {
    /// Transforms a 2D array from cartesian to (log-)polar coordinates.
    /// \tparam T               float or cfloat_t.
    /// \param cartesian        Input texture bound to a CUDA array.
    /// \param cartesian_interp Filter method of \p cartesian.
    /// \param[out] polar       On the \b device. Transformed array on the (log-)polar grid.
    /// \param polar_strides    BDHW strides of \p polar.
    /// \param polar_shape      BDHW shape of \p polar.
    /// \param cartesian_center HW transformation center.
    /// \param radius_range     Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether this is a log-polar coordinates system.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    /// \note The address mode is assumed to be BORDER_ZERO and the texture should use unnormalized coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void cartesian2polar(cudaTextureObject_t cartesian, InterpMode cartesian_interp,
                         T* polar, size4_t polar_strides, size4_t polar_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream);

    /// Transforms 2D array(s) from cartesian to (log-)polar coordinates.
    /// \tparam T                   float or cfloat_t.
    /// \param polar                Input texture bound to a CUDA array.
    /// \param polar_interp         Filter method of \p polar.
    /// \param[out] cartesian       On the \b device. Transformed array on the cartesian grid.
    /// \param cartesian_strides    BDHW strides of \p cartesian.
    /// \param cartesian_shape      BDHW shape of \p cartesian.
    /// \param cartesian_center     HW transformation center.
    /// \param radius_range         Radius [start,end] range of the bounding circle, in pixels.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log                  Whether this is a log-polar coordinates system.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    /// \note The address mode is assumed to be BORDER_ZERO and the texture should use unnormalized coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void polar2cartesian(cudaTextureObject_t polar, InterpMode polar_interp, float2_t polar_shape,
                         T* cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, Stream& stream);
}
