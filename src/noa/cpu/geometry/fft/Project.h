/// \file noa/cpu/geometry/fft/Project.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_insert_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                       (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);

    template<Remap REMAP, typename T>
    constexpr bool is_valid_extract_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                        (REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cpu::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Inserts 2D Fourier slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    /// \details The slices are scaled and the EWS curvature is applied. Then, they are rotated and added to the
    ///          3D cartesian Fourier volume using tri-linear interpolation. In practice, a density correction
    ///          (i.e. normalization) is often required after this operation. This can easily be achieved by inserting
    ///          the per-slice weights into another volume to keep track of what was inserted and where. Gridding
    ///          correction is also often necessary, see griddingCorrection below.
    ///
    /// \tparam REMAP               Remapping from the slice to the grid layout. Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[in] slice            On the \b host. Non-redundant 2D slice(s) to insert.
    /// \param slice_stride         Rightmost stride of \p slice.
    /// \param slice_shape          Rightmost logical shape of \p slice.
    /// \param[out] grid            On the \b host. Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_stride          Rightmost stride of \p grid.
    /// \param grid_shape           Rightmost logical shape of \p grid.
    /// \param[in] scaling_factors  On the \b host. 2x2 rightmost \e inverse real-space scaling to apply to the
    ///                             slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations        On the \b host. 3x3 rightmost \e forward rotation matrices. One per slice.
    /// \param cutoff               Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius           Rightmost Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note To decrease artefacts, the cartesian grid is usually oversampled. While this can be achieved by zero
    ///       padding the slices before the FFT, this function can also work with an oversampled grid. The oversampling
    ///       ratio is the ratio between the two innermost dimensions of \p grid_shape and \p slice_shape.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T>>>
    void insert3D(const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                  const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                  const shared_t<float22_t[]>& scaling_factors,
                  const shared_t<float33_t[]>& rotations,
                  float cutoff, float2_t ews_radius, Stream& stream);

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of insert3D. The transformation itself is identical to insert3D's, so
    ///          to extract a slice the same scaling factor and rotation should be used here.
    ///
    /// \tparam REMAP               Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \param[out] grid            On the \b host. Non-redundant centered 3D grid from which to extract the slices.
    /// \param grid_stride          Rightmost stride of \p grid.
    /// \param grid_shape           Rightmost logical shape of \p grid.
    /// \param[in] slice            On the \b host. Non-redundant extracted 2D slice(s).
    /// \param slice_stride         Rightmost stride of \p slice.
    /// \param slice_shape          Rightmost logical shape of \p slice.
    /// \param[in] scaling_factors  On the \b host. 2x2 rightmost \e inverse real-space scaling applied to the
    ///                             slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations        On the \b host. 3x3 rightmost \e forward rotation matrices. One per slice.
    /// \param cutoff               Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius           Rightmost Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note To decrease artefacts, the cartesian grid is usually oversampled. While this can be achieved by zero
    ///       padding the grid before the FFT, this function can also work with an oversampled grid. The oversampling
    ///       ratio is the ratio between the two innermost dimensions of \p grid_shape and \p slice_shape.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T>>>
    void extract3D(const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                   const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                   const shared_t<float22_t[]>& scaling_factors,
                   const shared_t<float33_t[]>& rotations,
                   float cutoff, float2_t ews_radius, Stream& stream);

    /// Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    /// \tparam T               float or double.
    /// \param[in] input        On the \b host. Real-space volume.
    /// \param input_stride     Rightmost stride of \p input.
    /// \param[out] output      On the \b host. Gridding-corrected output. Can be equal to \p input.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected
    ///                         using insert3D, whereas pre-correction is meant to be applied on the volume that is
    ///                         about to be forward projected using extract3D.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const shared_t<T[]>& input, size4_t input_stride,
                            const shared_t<T[]>& output, size4_t output_stride,
                            size4_t shape, bool post_correction, Stream& stream);
}
