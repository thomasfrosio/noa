#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename Value>
    constexpr bool is_valid_polar_xform_v = traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> && REMAP == HC2FC;
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// \tparam REMAP           Only HC2FC is currently supported. The output is denoted as "FC" (full-centered)
    ///                         to emphasize that it has a full shape (equals to \p polar_shape) and can map the
    ///                         entire angular range (e.g. 0 to 2PI).
    /// \tparam Value           float, double, cfloat_t or cdouble_t.
    /// \param[in] cartesian    Non-redundant centered 2D FFT to interpolate onto the new coordinate system.
    /// \param cartesian_shape  BDHW logical shape of \p cartesian.
    /// \param[out] polar       Transformed 2D array on the (log-)polar grid.
    ///                         The width dimension is the radius rho, from and to \p radius_range.
    ///                         The height dimension is the angle phi, from and to \p angle_range.
    /// \param frequency_range  Frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
    ///                         While Nyquist is at 0.5, higher values can be specified.
    /// \param angle_range      Angle [start,end] range increasing in the counterclockwise orientation, in radians.
    ///                         While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
    ///                         this range can include the entire unit circle, e.g. [-pi, pi].
    /// \param log              Whether log-polar coordinates should be computed instead.
    /// \param interp           Interpolation method used to interpolate the values onto the new grid.
    ///                         Cubic interpolations are not supported.
    ///
    /// \note Out-of-bounds elements are set to zero.
    /// \note If \p polar is on the CPU:\n
    ///         - \p cartesian and \p polar should not overlap.\n
    ///         - \p cartesian and \p polar should be on the same device.\n
    /// \note If \p polar is on the GPU:\n
    ///         - Double-precision (complex-) floating-points are not supported.\n
    ///         - \p input should be in the rightmost order and the width dimension should be contiguous.\n
    ///         - \p cartesian can be on any device including the CPU.
    ///         - In-place transformation is always allowed.\n
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Value>>>
    void cartesian2polar(const Array<Value>& cartesian, dim4_t cartesian_shape,
                         const Array<Value>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log = false, InterpMode interp = INTERP_LINEAR);

    /// Transforms 2D FFT(s) to (log-)polar coordinates.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, 1) the border mode should be BORDER_ZERO and un-normalized coordinates
    ///          should be used.
    template<Remap REMAP, typename Value, typename = std::enable_if_t<details::is_valid_polar_xform_v<REMAP, Value>>>
    void cartesian2polar(const Texture<Value>& cartesian, dim4_t cartesian_shape,
                         const Array<Value>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log = false);
}
