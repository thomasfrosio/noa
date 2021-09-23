/// \file noa/cpu/reconstruct/ProjectForward.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// TODO(TF) Add support for not-centered volumes. This is quite easy on the CPU backend, but requires to either
//          update the Interpolator3D or to use something else that is aware of the FFT layout. The point being
//          that the 2x2x2 interpolation window might be wrapped around at the edges if the transform is not
//          centered.

namespace noa::cpu::reconstruct {
    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of the backward projection. However, the transformation itself is
    ///          identical, but instead of adding the projections at the transformed coordinates, a simple forward
    ///          linear interpolation is performed to extract the values from the volume.
    ///
    /// \tparam IS_PROJ_CENTERED        Whether or not the output \p proj should be centered.
    ///                                 See "noa/cpu/fourier/README.txt" for more details.
    /// \tparam IS_VOLUME_CENTERED      Whether or not the input \p volume is centered.
    ///                                 Only centered volumes are currently supported.
    /// \tparam T                       float, double, cfloat_t or cdouble_t.
    /// \param[in] volume               On the \b host. Non-redundant volume from which the projections are extracted.
    /// \param volume_dim               Logical dimension size, in elements, of \p volume.
    /// \param[out] proj                On the \b host. Extracted, non-redundant projections. One per projection.
    /// \param proj_dim                 Logical dimension size, in elements, of \p proj.
    /// \param[in] proj_rotations       On the \b host. 3x3 rotation matrices of the projections. One per projection.
    ///                                 The rotation center is on the DC component of \p volume.
    ///                                 For a final rotation `A` of the projection within the volume, we need to apply
    ///                                 `inverse(A)` on the projection coordinates. This function assumes \p proj_rotations
    ///                                 is already inverted and directly pre-multiplies the coordinates with the matrix.
    /// \param[in] proj_magnifications  On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D scaling factor to apply on the projection, after rotation. The third value is
    ///                                 the in-plane magnification angle, in radians. Note that if a magnification
    ///                                 correction was used for the backward projection, the same values should be
    ///                                 passed here to revert the operation.
    /// \param[in] proj_shifts          On the \b host. If nullptr or if \p T is real, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply (as phase-shifts) to the projection after
    ///                                 the extraction.
    /// \param proj_count               Number of projections to extract.
    /// \param freq_max                 Maximum frequency to extract. Values are clamped from 0 to 0.5.
    /// \param ewald_sphere_radius      Ewald sphere radius (i.e. `1/wavelength`, in SI) of the projections.
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections, i.e. slices.
    ///
    /// \note Only square projections and cubic volumes are supported. The volume is usually oversampled compared to
    ///       the projections. The oversampling ratio is set to be the ratio between \p volume_dim and \p proj_dim.
    ///       Note that both the projections and the volume are non-redundant transforms, so the physical size in
    ///       the first dimension is `x/2+1`, x being the logical size, i.e. \p volume_dim or \p proj_dim.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                                 const float33_t* proj_rotations, const float3_t* proj_magnifications,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max = 0.5f, float ewald_sphere_radius = 0.f);

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                                 const float33_t* proj_rotations, float3_t proj_magnification,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max = 0.5f, float ewald_sphere_radius = 0.f);
}
