#pragma once

#include "noa/common/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_polar_xform_v = traits::is_any_v<T, float, cfloat_t> && REMAP == HC2FC;
}

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// \tparam REMAP               Only HC2FC is currently supported. The output is denoted as "FC" (full-centered)
    ///                             to emphasize that it has a full shape (equals to \p polar_shape) and can map the
    ///                             entire angular range (e.g. 0 to 2PI).
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \param[in] cartesian        On the \b device. Non-redundant centered FFT to interpolate onto the new coordinate system.
    /// \param cartesian_strides    BDHW strides of \p cartesian.
    /// \param cartesian_shape      BDHW logical shape of \p cartesian.
    /// \param[out] polar           On the \b device. Transformed array on the (log-)polar grid.
    /// \param polar_strides        BDHW strides of \p polar.
    /// \param polar_shape          BDHW shape of \p polar.
    ///                             The width dimension is the radius rho, from and to \p radius_range.
    ///                             The height dimension is the angle phi, from and to \p angle_range.
    /// \param frequency_range      Frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
    ///                             While Nyquist is at 0.5, higher values can be specified.
    /// \param angle_range          Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    ///                             While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
    ///                             this range can include the entire unit circle, e.g. [-pi, pi].
    /// \param log                  Whether log-polar coordinates should be computed instead.
    /// \param interp               Interpolation method used to interpolate the values onto the new grid.
    ///                             Cubic interpolations are not supported.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, T>>>
    void cartesian2polar(const shared_t<T[]>& cartesian, size4_t cartesian_strides, size4_t cartesian_shape,
                         const shared_t<T[]>& polar, size4_t polar_strides, size4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp, Stream& stream);

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param cartesian        Input texture bound to a CUDA array with a non-redundant centered FFT
    /// \param cartesian_interp Filter method of \p cartesian.
    /// \param cartesian_shape  BDHW logical shape of \p cartesian.
    /// \param[out] polar       On the \b device. Transformed array on the (log-)polar grid.
    /// \param polar_strides    BDHW strides of \p polar.
    /// \param polar_shape      BDHW shape of \p polar.
    /// \param frequency_range  Frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    /// \note The address mode is assumed to be BORDER_ZERO and the texture should use unnormalized coordinates.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void cartesian2polar(cudaTextureObject_t cartesian, InterpMode cartesian_interp, size2_t cartesian_shape,
                         T* polar, size4_t polar_strides, size4_t polar_shape,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, Stream& stream);
}
