#pragma once

#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry::fft::details {
    using Remap = noa::fft::Remap;
    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_insert_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                       traits::is_any_v<S, Array<float22_t>, float22_t> &&
                                       traits::is_any_v<R, Array<float33_t>, float33_t> &&
                                       (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
                                        REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename T, typename S, typename R>
    constexpr bool is_valid_extract_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
                                        traits::is_any_v<S, Array<float22_t>, float22_t> &&
                                        traits::is_any_v<R, Array<float33_t>, float33_t> &&
                                        (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::geometry::fft {
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
    /// \tparam S                   Array<float22_t> or float22_t.
    /// \tparam R                   Array<float33_t> or float33_t.
    /// \param[in] slice            Non-redundant 2D slice(s) to insert.
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[out] grid            Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_shape           BDHW logical shape of \p grid.
    /// \param[in] scaling_factors  2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                             If an array is passed, it can be empty or have one matrix per slice.
    ///                             Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] rotations        3x3 DHW \e forward rotation matrices.
    ///                             If an array is passed, it should have one matrix per slice.
    ///                             Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff               Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param target_shape         Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius           HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    ///
    /// \note If \p grid is on the GPU, \p scaling_factors and \p rotations can be on any device,
    ///       including the CPU. If \p grid is on the CPU, they should be dereferenceable by the CPU.
    /// \note This function normalizes the slice and grid dimensions, and works with normalized frequencies,
    ///       from -0.5 to 0.5 cycle/pix. By default (empty \p target_shape or \p target_shape == \p grid_shape),
    ///       the slice frequencies are mapped into the grid frequencies. If the grid is larger than the slices,
    ///       the slices are implicitly stretched (over-sampling case). If the grid is smaller than the slices,
    ///       the slices are shrank (under-sampling case).
    ///       However, if \p target_shape is specified, the slice frequencies are instead mapped into the frequencies
    ///       of a 3D FFT volume of shape \p target_shape. In this case, \p grid is just the region to "render" within
    ///       the volume defined by \p target_shape, which can be of any shape, e.g. a subregion of \p target_shape.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ews_radius. To insert the other side, one would have to
    ///       call this function a second time with \p ews_radius * -1.
    /// \note The scaling factors and the rotation are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p ews_radius is 0, the scaling factors can be merged with the
    ///       rotations.
    /// \note The redundant line at x=0 is entirely inserted into the volume. If the projection has an in-plane
    ///       rotation, this results into having this line inserted twice. This emphasizes the need of normalizing
    ///       the output grid, or extracted slice(s), with the corresponding inserted weights, or extracted weights.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, T, S, R>>>
    void insert3D(const Array<T>& slice, dim4_t slice_shape,
                  const Array<T>& grid, dim4_t grid_shape,
                  const S& scaling_factors, const R& rotations,
                  float cutoff = 0.5f,
                  dim4_t target_shape = {},
                  float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of insert3D. The transformation itself is identical to insert3D's,
    ///          so the same parameters should be used here.
    ///
    /// \tparam REMAP               Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam T                   float, double, cfloat_t, cdouble_t.
    /// \tparam S                   Array<float22_t> or float22_t.
    /// \tparam R                   Array<float33_t> or float33_t.
    /// \param[out] grid            Non-redundant centered 3D grid from which to extract the slices.
    /// \param grid_shape           BDHW logical shape of \p grid.
    /// \param[in] slice            Non-redundant 2D extracted slice(s).
    /// \param slice_shape          BDHW logical shape of \p slice.
    /// \param[in] scaling_factors  2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                             If an array is passed, it can be empty or have one matrix per slice.
    ///                             Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] rotations        3x3 DHW \e forward rotation matrices.
    ///                             If an array is passed, it should have one matrix per slice.
    ///                             Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff               Frequency cutoff in \p grid, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param target_shape         Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius           HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                             If negative, the negative curve is computed.
    ///                             If {0,0}, the slices are projections.
    ///
    /// \note If \p slice is on the CPU, \p scaling_factors and \p rotations should be dereferenceable by the CPU.
    ///       If \p slice is on the GPU, they can be on any device, including the CPU.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T, S, R>>>
    void extract3D(const Array<T>& grid, dim4_t grid_shape,
                   const Array<T>& slice, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff = 0.5f,
                   dim4_t target_shape = {},
                   float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This functions has the same features and limitations as the overload taking arrays.
    ///          However, for GPU textures, the border mode should be BORDER_ZERO, the interpolation mode
    ///          should be INTERP_LINEAR or INTERP_LINEAR_FAST, and un-normalized coordinates should be used.
    /// \note If \p slice is on the GPU:
    ///         - Double precision is not supported.
    ///         - \p scaling_factors and \p rotations can be on any device, including the CPU.
    template<Remap REMAP, typename T, typename S, typename R,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, T, S, R>>>
    void extract3D(const Texture<T>& grid, dim4_t grid_shape,
                   const Array<T>& slice, dim4_t slice_shape,
                   const S& scaling_factors, const R& rotations,
                   float cutoff = 0.5f,
                   dim4_t target_shape = {},
                   float2_t ews_radius = {});

    /// Corrects for the gridding, assuming tri-linear interpolation was used during the Fourier insertion.
    /// \details During direct Fourier insertion of slices S into a volume B, two problems arises:
    ///          1) The insertion is not uniform (e.g. inherently more dense at low frequencies). This can be
    ///             easily corrected by inserting the data as well as its associated weights and normalizing the
    ///             inserted data with the inserted weights. This is often referred to as density correction.
    ///             This function is not about that.
    ///          2) The data-points are inserted in Fourier space by interpolation, a process called gridding,
    ///             which is essentially a convolution between the data points and the interpolation filter
    ///             (e.g. triangle pulse for linear interpolation). The interpolation filter is often referred to as
    ///             the gridding kernel. Since convolution in frequency space corresponds to a multiplication in
    ///             real-space, the resulting inverse Fourier transform of the volume B is the product of the final
    ///             wanted reconstruction and the apodization function. The apodization function is the Fourier
    ///             transform of the gridding kernel (e.g. sinc^2 for linear interpolation). This function is there
    ///             to correct for this gridding artefact, assuming tri-linear interpolation.
    /// \tparam T               float or double.
    /// \param[in] input        Inverse Fourier transform of the 3D grid used for direct Fourier insertion.
    /// \param[out] output      Gridding-corrected output. Can be equal to \p input.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected
    ///                         using insert3D, whereas pre-correction is meant to be applied on the volume that is
    ///                         about to be forward projected using extract3D.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double>>>
    void griddingCorrection(const Array<T>& input, const Array<T>& output, bool post_correction);
}

#define NOA_UNIFIED_GEOMETRY_FFT_PROJECT_
#include "noa/unified/geometry/fft/Project.inl"
#undef NOA_UNIFIED_GEOMETRY_FFT_PROJECT_
