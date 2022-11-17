#pragma once

#include "noa/common/geometry/Euler.h"
#include "noa/common/geometry/Transform.h"
#include "noa/unified/Array.h"
#include "noa/unified/Texture.h"

namespace noa::geometry::fft::details {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_v = traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
                                       traits::is_any_v<Scale, Array<float22_t>, float22_t> &&
                                       traits::is_any_v<Rotate, Array<float33_t>, float33_t> &&
                                       (REMAP == Remap::H2H || REMAP == Remap::H2HC ||
                                        REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_insert_thick_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale, Array<float22_t>, float22_t> &&
            traits::is_any_v<Rotate, Array<float33_t>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale, typename Rotate>
    constexpr bool is_valid_extract_v = traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
                                        traits::is_any_v<Scale, Array<float22_t>, float22_t> &&
                                        traits::is_any_v<Rotate, Array<float33_t>, float33_t> &&
                                        (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);

    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1>
    constexpr bool is_valid_insert_insert_extract_v =
            traits::is_any_v<Value, float, double, cfloat_t, cdouble_t> &&
            traits::is_any_v<Scale0, Array<float22_t>, float22_t> &&
            traits::is_any_v<Scale1, Array<float22_t>, float22_t> &&
            traits::is_any_v<Rotate0, Array<float33_t>, float33_t> &&
            traits::is_any_v<Rotate1, Array<float33_t>, float33_t> &&
            (REMAP == Remap::HC2H || REMAP == Remap::HC2HC);
}

namespace noa::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using tri-linear rasterisation.
    /// \details The slices are scaled and the EWS curvature is applied. Then, they are rotated and added to the
    ///          3D cartesian Fourier volume using tri-linear rasterisation. This method, often referred to as
    ///          direct Fourier insertion, explicitly sets the "thickness" of the central slices as the width of
    ///          the interpolation window (referred to as gridding kernel), which in this case is 1 voxel.
    ///          In practice, a density correction (i.e. normalization) is often required after this operation.
    ///          This can easily be achieved by inserting the per-slice weights into another volume to keep track
    ///          of what was inserted and where. Gridding correction can also be beneficial as post-processing
    ///          one the real-space output (see griddingCorrection below).
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout.
    ///                                 Should be H2H, H2HC, HC2H or HC2HC.
    /// \tparam Value                   float, double, cfloat_t, cdouble_t.
    /// \tparam Scale                   Array<float22_t> or float22_t.
    /// \tparam Rotate                  Array<float33_t> or float33_t.
    /// \param[in] slice                Non-redundant 2D slice(s) to insert.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] grid                Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    ///
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
    /// \note The scaling and the rotation matrices are kept separated from one another in order to properly compute the
    ///       curve of the Ewald sphere. Indeed, the scaling is applied first to correct for magnification, so that the
    ///       EWS is computed using the original frequencies (from the scattering) and is therefore spherical even
    ///       under anisotropic magnification. If \p ews_radius is 0, the scaling factors can be merged to the
    ///       rotations.
    /// \note The redundant line at x=0 is entirely inserted into the volume. If the projection has an in-plane
    ///       rotation, this results into having this line inserted twice. This emphasizes the need of normalizing
    ///       the output grid, or extracted slice(s), with the corresponding inserted weights, or extracted weights.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const Array<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& inv_scaling_matrix,
                  const Rotate& fwd_rotation_matrix,
                  float cutoff,
                  dim4_t target_shape = {},
                  float2_t ews_radius = {});

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using bi-linear interpolation and sinc-weighting.
    /// \details This function computes the inverse transformation compared to the overload above using rasterization,
    ///          effectively transforming the 3D grid onto the input slice(s). Briefly, for each input slice, each
    ///          voxel is assigned to a transformed frequency (w,v,u) corresponding to the reference frame of the
    ///          current slice to insert. 1) Given the frequency w, which is the distance of the voxel along the
    ///          normal of the slice, and \p slice_z_radius, it computes a sinc-weight from 1 (on the slice) to 0
    ///          (outside the slice). 2) Then, if the slice does contribute to the voxel, i.e. the sinc-weight is
    ///          non-zero, a bi-linear interpolation is done using the (v,u) frequency component of the voxel.
    ///          The interpolated value is then sinc-weighted and added to the voxel.
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout.
    ///                                 Should be HC2H or HC2HC.
    /// \tparam Value                   float, double, cfloat_t, cdouble_t.
    /// \tparam Scale                   Array<float22_t> or float22_t.
    /// \tparam Rotate                  Array<float33_t> or float33_t.
    /// \param[in] slice                Non-redundant 2D slice(s) to insert.
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[out] grid                Non-redundant 3D grid inside which the slices are inserted.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] fwd_scaling_matrix   2x2 HW \e forward real-space scaling matrix to apply to the slices
    ///                                 before the rotation. If an array is passed, it can be empty or have
    ///                                 one matrix per slice. Otherwise the same scaling matrix is applied
    ///                                 to every slice.
    /// \param[in] inv_rotation_matrix  3x3 DHW \e inverse rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param slice_z_radius           Radius along the normal of the central slices, in cycle/pix.
    ///                                 This is usually from 0.001 to 0.01.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const Array<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrix,
                  const Rotate& inv_rotation_matrix,
                  float slice_z_radius,
                  float cutoff,
                  dim4_t target_shape = {},
                  float2_t ews_radius = {});

    /// Inserts 2D Fourier central slice(s) into a 3D Fourier volume, using bi-linear interpolation and sinc-weighting.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_insert_thick_v<REMAP, Value, Scale, Rotate>>>
    void insert3D(const Texture<Value>& slice, dim4_t slice_shape,
                  const Array<Value>& grid, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrix,
                  const Rotate& inv_rotation_matrix,
                  float slice_z_radius,
                  float cutoff,
                  dim4_t target_shape = {},
                  float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of insert3D. The transformation itself is identical to the
    ///          transformation of insert3D using rasterization, so the same parameters can be used here.
    ///
    /// \tparam REMAP                   Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam Value                   float, double, cfloat_t, cdouble_t.
    /// \tparam Slice                   Array<float22_t> or float22_t.
    /// \tparam Rotate                  Array<float33_t> or float33_t.
    /// \param[out] grid                Non-redundant centered 3D grid from which to extract the slices.
    /// \param grid_shape               BDHW logical shape of \p grid.
    /// \param[in] slice                Non-redundant 2D extracted slice(s).
    /// \param slice_shape              BDHW logical shape of \p slice.
    /// \param[in] inv_scaling_matrix   2x2 HW \e inverse real-space scaling to apply to the slices before the rotation.
    ///                                 If an array is passed, it can be empty or have one matrix per slice.
    ///                                 Otherwise the same scaling matrix is applied to every slice.
    /// \param[in] fwd_rotation_matrix  3x3 DHW \e forward rotation matrices to apply to the slices.
    ///                                 If an array is passed, it should have one matrix per slice.
    ///                                 Otherwise the same rotation matrix is applied to every slice.
    /// \param cutoff                   Frequency cutoff in \p grid, in cycle/pix.
    /// \param target_shape             Actual BDHW logical shape of the 3D volume.
    /// \param ews_radius               HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                 If negative, the negative curve is computed.
    ///                                 If {0,0}, the slices are projections.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Value, Scale, Rotate>>>
    void extract3D(const Array<Value>& grid, dim4_t grid_shape,
                   const Array<Value>& slice, dim4_t slice_shape,
                   const Scale& inv_scaling_matrix,
                   const Rotate& fwd_rotation_matrix,
                   float cutoff = 0.5f,
                   dim4_t target_shape = {},
                   float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a Fourier volume using tri-linear interpolation.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Scale, typename Rotate,
             typename = std::enable_if_t<details::is_valid_extract_v<REMAP, Value, Scale, Rotate>>>
    void extract3D(const Texture<Value>& grid, dim4_t grid_shape,
                   const Array<Value>& slice, dim4_t slice_shape,
                   const Scale& inv_scaling_matrix,
                   const Rotate& fwd_rotation_matrix,
                   float cutoff = 0.5f,
                   dim4_t target_shape = {},
                   float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a virtual volume filled by other slices, using linear interpolation.
    /// \details This function effectively combines the insertion by interpolation and the extraction, but only
    ///          renders the frequencies that are going to be used for the extraction. This function is useful if
    ///          the 3D Fourier volume, where the slices are inserted, is used for extracting slice(s) immediately
    ///          after the insertion. It is much faster than calling insert3D and extract3D, uses less memory
    ///          (the 3D Fourier volume is entirely skipped), and skips a layer of interpolation.
    ///
    /// \tparam REMAP                           Remapping from the slice to the grid layout. Should be HC2H or HC2HC.
    /// \tparam Value                           float, double, cfloat_t, cdouble_t.
    /// \tparam Scale0                          Array<float22_t> or float22_t.
    /// \tparam Rotate0                         Array<float33_t> or float33_t.
    /// \tparam Scale1                          Array<float22_t> or float22_t.
    /// \tparam Rotate1                         Array<float33_t> or float33_t.
    /// \param[in] input_slice                  Non-redundant 2D slice(s) to insert.
    /// \param input_slice_shape                BDHW logical shape of \p input_slice.
    /// \param[out] output_slice                Non-redundant 2D extracted slice(s). See warning below.
    /// \param output_slice_shape               BDHW logical shape of \p output_slice.
    /// \param[in] input_fwd_scaling_matrix     2x2 HW \e forward real-space scaling matrix to apply to the input
    ///                                         slices before the rotation. If an array is passed, it can be empty
    ///                                         or have one matrix per slice. Otherwise the same scaling matrix
    ///                                         is applied to every slice.
    /// \param[in] input_inv_rotation_matrix    3x3 DHW \e inverse rotation matrices to apply to the input slices.
    ///                                         If an array is passed, it should have one matrix per slice.
    ///                                         Otherwise the same rotation matrix is applied to every slice.
    /// \param[in] output_inv_scaling_matrix    2x2 HW \e inverse real-space scaling matrix to apply to the output
    ///                                         slices before the rotation. If an array is passed, it can be empty
    ///                                         or have one matrix per slice. Otherwise the same scaling matrix
    ///                                         is applied to every slice.
    /// \param[in] output_fwd_rotation_matrix   3x3 DHW \e forward rotation matrices to apply to the output slices.
    ///                                         If an array is passed, it should have one matrix per slice.
    ///                                         Otherwise the same rotation matrix is applied to every slice.
    /// \param slice_z_radius                   Radius along the normal of the central slices, in cycle/pix.
    ///                                         This is usually from 0.001 to 0.01.
    /// \param cutoff                           Frequency cutoff of the virtual 3D Fourier volume, in cycle/pix.
    /// \param ews_radius                       HW Ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                                         If negative, the negative curve is computed.
    ///                                         If {0,0}, the slices are projections.
    ///
    /// \warning This function doesn't set the \p output_slice. Instead, it adds the contribution of \p input_slice
    ///          onto the signal already present in \p output_slice. This is different from the overload taking a
    ///          3D grid and allows to still insert slices one at a time.
    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                     REMAP, Value, Scale0, Rotate0, Scale1, Rotate1>>>
    void extract3D(const Array<Value>& input_slice, dim4_t input_slice_shape,
                   const Array<Value>& output_slice, dim4_t output_slice_shape,
                   const Scale0& input_fwd_scaling_matrix, const Rotate0& input_inv_rotation_matrix,
                   const Scale1& output_inv_scaling_matrix, const Rotate1& output_fwd_rotation_matrix,
                   float slice_z_radius,
                   float cutoff = 0.5f,
                   float2_t ews_radius = {});

    /// Extracts 2D Fourier slice(s) from a virtual volume filled by other slices, using linear interpolation.
    /// \details This function has the same features and limitations as the overload taking arrays, but uses textures.
    template<Remap REMAP, typename Value, typename Scale0, typename Rotate0, typename Scale1, typename Rotate1,
             typename = std::enable_if_t<details::is_valid_insert_insert_extract_v<
                    REMAP, Value, Scale0, Rotate0, Scale1, Rotate1>>>
    void extract3D(const Texture<Value>& input_slice, dim4_t input_slice_shape,
                   const Array<Value>& output_slice, dim4_t output_slice_shape,
                   const Scale0& input_fwd_scaling_matrix, const Rotate0& input_inv_rotation_matrix,
                   const Scale1& output_inv_scaling_matrix, const Rotate1& output_fwd_rotation_matrix,
                   float slice_z_radius,
                   float cutoff = 0.5f,
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
    /// \tparam Value           float or double.
    /// \param[in] input        Inverse Fourier transform of the 3D grid used for direct Fourier insertion.
    /// \param[out] output      Gridding-corrected output. Can be equal to \p input.
    /// \param post_correction  Whether the correction is the post- or pre-correction.
    ///                         Post correction is meant to be applied on the volume that was just back-projected
    ///                         using insert3D, whereas pre-correction is meant to be applied on the volume that is
    ///                         about to be forward projected using extract3D.
    template<typename Value, typename = std::enable_if_t<traits::is_any_v<Value, float, double>>>
    void griddingCorrection(const Array<Value>& input, const Array<Value>& output, bool post_correction);
}
