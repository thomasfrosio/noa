#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_insert_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                       (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
                                        REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T>
    constexpr bool is_valid_extract_v = traits::is_any_v<T, float, cfloat_t> &&
                                        (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::cuda::geometry::fft {
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
    /// \param[in] slice            On the \b device. Non-redundant 2D slice(s) to insert.
    /// \param slice_strides        BDHW strides of \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] grid            On the \b device. Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_strides         BDHW strides of \p grid.
    /// \param grid_shape           BDHW logical shape of \p grid.
    /// \param[in] scaling_factors  On the \b host or \b device. 2x2 HW \e inverse real-space scaling to apply
    ///                             to the slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations        On the \b host or \b device. 3x3 DHW \e forward rotation matrices.
    /// \param cutoff               One per slice. Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius           HW ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note To decrease artefacts, the cartesian grid is usually oversampled. While this can be achieved by zero
    ///       padding the slices before the FFT, this function can also work with an oversampled grid. The oversampling
    ///       ratio is the ratio between the two innermost dimensions of \p grid_shape and \p slice_shape.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T>>>
    void insert3D(const shared_t<T[]>& slice, size4_t slice_strides, size4_t slice_shape,
                  const shared_t<T[]>& grid, size4_t grid_strides, size4_t grid_shape,
                  const shared_t<float22_t[]>& scaling_factors,
                  const shared_t<float33_t[]>& rotations,
                  float cutoff, float2_t ews_radius, Stream& stream);

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of insert3D. The transformation itself is identical to insert3D's, so
    ///          to extract a slice the same scaling factor and rotation should be used here.
    ///
    /// \tparam REMAP               Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam T                   float or cfloat_t.
    /// \param[out] grid            On the \b device. Non-redundant centered 3D grid from which to extract the slices.
    /// \param grid_strides         BDHW strides of \p grid.
    /// \param grid_shape           BDHW logical shape of \p grid.
    /// \param[in] slice            On the \b device. Non-redundant extracted 2D slice(s).
    /// \param slice_strides        BDHW strides of \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[in] scaling_factors  On the \b host or \b device. 2x2 HW \e inverse real-space scaling applied
    ///                             to the slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations        On the \b host or \b device. 3x3 DHW \e forward rotation matrices.
    /// \param cutoff               One per slice. Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius           BDHW ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note To decrease artefacts, the cartesian grid is usually oversampled. While this can be achieved by zero
    ///       padding the grid before the FFT, this function can also work with an oversampled grid. The oversampling
    ///       ratio is the ratio between the two innermost dimensions of \p grid_shape and \p slice_shape.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T>>>
    void extract3D(const shared_t<T[]>& grid, size4_t grid_strides, size4_t grid_shape,
                   const shared_t<T[]>& slice, size4_t slice_strides, size4_t slice_shape,
                   const shared_t<float22_t[]>& scaling_factors,
                   const shared_t<float33_t[]>& rotations,
                   float cutoff, float2_t ews_radius, Stream& stream);

    /// Corrects for the gridding, assuming tri-linear interpolation is used during the insertion or extraction.
    /// \tparam T               float or double.
    /// \param[in] input        On the \b device. Real-space volume.
    /// \param input_strides    BDHW strides of \p input.
    /// \param[out] output      On the \b device. Gridding-corrected output. Can be equal to \p input.
    /// \param output_strides   BDHW strides of \p output.
    /// \param shape            BDHW shape of \p input and \p output.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected
    ///                         using insert3D, whereas pre-correction is meant to be applied on the volume that is
    ///                         about to be forward projected using extract3D.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const shared_t<T[]>& input, size4_t input_strides,
                            const shared_t<T[]>& output, size4_t output_strides,
                            size4_t shape, bool post_correction, Stream& stream);
}

namespace noa::cuda::geometry::fft {
    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \tparam REMAP               Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam T                   float or cfloat_t.
    /// \param grid                 Texture bound to a CUDA array.
    ///                             The interpolation mode is expected to be INTERP_LINEAR.
    ///                             The border mode is expected to be BORDER_ZERO.
    ///                             Un-normalized coordinates should be used.
    /// \param grid_shape           BDHW logical shape of the grid.
    /// \param[in] slice            On the \b device. Non-redundant extracted 2D slice(s).
    /// \param slice_strides        BDHW strides of \p slice.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[in] scaling_factors  On the \b host or \b device. 2x2 HW \e inverse real-space scaling applied
    ///                             to the slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations        On the \b host or \b device. 3x3 DHW \e forward rotation matrices.
    /// \param cutoff               One per slice. Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius           BDHW ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p grid, \p slice, \p scaling_factors, and \p rotations should stay valid until completion.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T>>>
    void extract3D(cudaTextureObject_t grid, int3_t grid_shape,
                   T* slice, size4_t slice_strides, size4_t slice_shape,
                   const float22_t* scaling_factors, const float33_t* rotations,
                   float cutoff, float2_t ews_radius, Stream& stream);
}
