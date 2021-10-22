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
    ///          linear interpolation is performed to extract the values from the volume. As such, to extract a
    ///          projection inserted during the backward projection, the same rotation and scaling should be used here.
    /// \see projectBackward for more details.
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
    ///
    /// \param[in] proj_scaling_factors On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D real-space scaling to apply to the projection before the rotation.
    ///                                 The third value is the in-plane scaling angle, in radians.
    /// \param[in] proj_rotations       On the \b host. 3x3 rotation matrices of the projections. One per projection.
    ///                                 The rotation center is on the DC component.
    ///                                 See "noa/common/transform/README.txt" for more details on the conventions.
    /// \param[in] proj_shifts          On the \b host. If nullptr or if \p T is real, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply (as phase-shifts) to the projection after extraction.
    ///                                 Positive values shift the real-space object to the right.
    ///                                 Usually, this is the opposite value than the one used during backward projection.
    /// \param proj_count               Number of projections to extract.
    /// \param freq_max                 Maximum frequency to insert. Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ewald_sphere_radius      Ewald sphere radius, in 1/pixel (e.g. `pixel_size/wavelength`).
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                                 const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max = 0.5f, float ewald_sphere_radius = 0.f);

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                                 float3_t proj_scaling_factor, const float33_t* proj_rotations,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max = 0.5f, float ewald_sphere_radius = 0.f);
}
