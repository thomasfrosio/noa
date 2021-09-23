/// \file noa/cpu/reconstruct/ProjectBackward.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::reconstruct {
    /// Inserts Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// \details The projections are phase shifted, their magnification is corrected and the EWS curvature is applied.
    ///          Then, they are rotated and added to the cartesian (oversampled) 3D volume using tri-linear interpolation.
    ///
    /// \tparam IS_PROJ_CENTERED        Whether or not the input \p proj and \p proj_weights are centered.
    ///                                 See "noa/cpu/fourier/README.txt" for more details.
    /// \tparam IS_VOLUME_CENTERED      Whether or not the output \p volume and \p volume_weights are centered.
    /// \tparam T                       float or double.
    /// \param[in] proj                 On the \b host. Non-redundant projections to insert. One per projection.
    /// \param[in] proj_weights         On the \b host. Element-wise weights associated to \p proj. One per projection.
    /// \param proj_dim                 Logical dimension size, in elements, of \p proj and \p proj_weights.
    /// \param[out] volume              On the \b host. Non-redundant volume inside which the projections are inserted.
    /// \param[out] volume_weights      On the \b host. Element-wise weights associated to the volume.
    /// \param volume_dim               Logical dimension size, in elements, of \p volume and \p volume_weights.
    /// \param[in] proj_shifts          On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply (as phase-shifts) to the projection before any
    ///                                 other transformation.
    /// \param[in] proj_magnifications  On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D magnification correction of the projection. The magnification is corrected
    ///                                 before the rotation. The third value is the in-plane magnification angle,
    ///                                 in radians. Note that this is not a scaling to apply on the projection, but the
    ///                                 magnification of the projection that needs to be corrected/removed.
    /// \param[in] proj_rotations       On the \b host. 3x3 rotation matrices of the projections. One per projection.
    ///                                 The rotation center is on the DC component of \p volume. For a final rotation `A`
    ///                                 of the projection within the volume, we need to apply `inverse(A)` on the
    ///                                 projection coordinates. This function assumes \p proj_rotations is already
    ///                                 inverted and directly pre-multiplies the coordinates with the matrix.
    ///                                 See "noa/common/transform/README.txt" for more details on the conventions.
    /// \param proj_count               Number of projections to insert.
    /// \param freq_max                 Maximum frequency to insert. Values are clamped from 0 to 0.5.
    /// \param ewald_sphere_radius      Ewald sphere radius (i.e. `1/wavelength`, in SI) of the projections.
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections, i.e. slices.
    ///
    /// \note Since a rotation is applied to the projections, they should have their real-space rotation-center
    ///       shifted at 0, as with any rotation applied in Fourier space. If this is not the case, \p proj_shifts
    ///       can be used to properly phase shift the projection before the rotation.
    /// \note Only square projections and cubic volumes are supported. The volume is usually oversampled compared to
    ///       the projections. The oversampling ratio is set to be the ratio between \p volume_dim and \p proj_dim.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ewald_sphere_radius. To insert the other side, one would have to
    ///       call this function a second time with the negative \p ewald_sphere_radius.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                                  Complex<T>* volume, T* volume_weights, size_t volume_dim,
                                  const float2_t* proj_shifts, const float3_t* proj_magnifications,
                                  const float33_t* proj_rotations, uint proj_count,
                                  float freq_max = 0.5f, float ewald_sphere_radius = 0.f);

    /// Inserts Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification correction is applied to the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_dim,
                                  Complex<T>* volume, T* volume_weights, size_t volume_dim,
                                  const float2_t* proj_shifts, float3_t proj_magnification,
                                  const float33_t* proj_rotations, uint proj_count,
                                  float freq_max = 0.5f, float ewald_sphere_radius = 0.f);
}
