/// \file noa/gpu/cuda/reconstruct/ProjectBackward.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 15 Sep 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::reconstruct {
    /// Adds Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// \details The projections are phase shifted, their magnification is corrected and the EWS curvature is applied.
    ///          Then, they are rotated and added to the cartesian (oversampled) 3D volume using tri-linear interpolation.
    ///
    /// \tparam IS_PROJ_CENTERED        Whether or not the input \p proj and \p proj_weights are centered.
    ///                                 See "noa/cpu/fourier/README.txt" for more details.
    /// \tparam IS_VOLUME_CENTERED      Whether or not the output \p volume and \p volume_weight are centered.
    /// \tparam T                       float or double.
    /// \param[in] proj                 On the \b device. Non-redundant projections to insert. One per projection.
    /// \param[in] proj_weights         On the \b device. Element-wise weights associated to \p proj. One per projection.
    /// \param proj_pitch               Pitch, in elements, of \p proj and \p proj_weights.
    /// \param proj_dim                 Logical dimension size, in elements, of \p proj and \p proj_weights.
    /// \param[in] proj_shifts          On the \b device. If nullptr, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply to the projection before any other transformation.
    /// \param[in] proj_magnifications  On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D magnification correction of the projection. The magnification is corrected
    ///                                 before the rotation. The third value is the in-plane magnification angle,
    ///                                 in radians. Note that this is not a scaling to apply on the projection, but the
    ///                                 magnification of the projection that needs to be corrected/removed.
    /// \param[in] proj_angles          On the \b host. ZYZ Euler angles. One per projection.
    ///                                 Rotation to apply to the projection. See "noa/common/transform/README.txt" for
    ///                                 more details on the transformation conventions.
    /// \param proj_count               Number of projections to insert.
    /// \param[out] volume              On the \b host. If nullptr, it is ignored, as well as \p proj.
    ///                                 Non-redundant volume inside which the projections are inserted.
    /// \param[out] volume_weights      On the \b host. If nullptr, it is ignored, as well as \p proj_weights.
    ///                                 Element-wise weights associated to the volume.
    /// \param volume_pitch             Pitch, in elements, of \p volume and \p volume_weights.
    /// \param volume_dim               Logical dimension size, in elements, of \p volume and \p volume_weights.
    /// \param freq_max                 Maximum frequency to insert. Values are clamped from 0 to 0.5.
    /// \param ewald_sphere_radius      Ewald sphere radius (i.e. `1/wavelength`, in SI) of the projections.
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections, i.e. slices.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///                                 The stream is synchronized when the function returns.
    ///
    /// \note Since a rotation is applied to the projections, they are expected to have their real space rotation
    ///       center phase shifted at 0, as with any rotation applied in Fourier space. If this is not the case,
    ///       \p proj_shifts can be used to properly phase shift the projection before the rotation.
    /// \note Only square projections and cubic volumes are supported. The volume is usually oversampled compared to
    ///       the projections. The oversampling ratio is set to be the ratio between \p volume_dim and \p proj_dim.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ewald_sphere_radius. To insert the other side, one would have to
    ///       call this function a second time with the negative \p ewald_sphere_radius.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                                  const float2_t* proj_shifts, const float3_t* proj_magnifications,
                                  const float3_t* proj_angles, uint proj_count,
                                  Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                                  float freq_max, float ewald_sphere_radius, Stream& stream);

    /// Adds Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification correction is applied to all projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                                  const float2_t* proj_shifts, float3_t proj_magnification,
                                  const float3_t* proj_angles, uint proj_count,
                                  Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                                  float freq_max, float ewald_sphere_radius, Stream& stream);
}

// -- Max to Nyquist, flat EWS -- //
namespace noa::cuda::reconstruct{
    /// Adds Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                                const float2_t* proj_shifts, const float3_t* proj_magnifications,
                                const float3_t* proj_angles, uint proj_count,
                                Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                                Stream& stream) { // overload instead of default value to keep stream at the end
        projectBackward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(
                proj, proj_weights, proj_pitch, proj_dim, proj_shifts, proj_magnifications, proj_angles, proj_count,
                volume, volume_weights, volume_pitch, volume_dim, 0.5f, 0.f, stream);
    }

    /// Adds Fourier "slices" into a Fourier volume using tri-linear interpolation.
    /// Overload which sets the same magnification correction for all projections.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectBackward(const Complex<T>* proj, const T* proj_weights, size_t proj_pitch, size_t proj_dim,
                                const float2_t* proj_shifts, float3_t proj_magnification,
                                const float3_t* proj_angles, uint proj_count,
                                Complex<T>* volume, T* volume_weights, size_t volume_pitch, size_t volume_dim,
                                Stream& stream) {
        projectBackward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(
                proj, proj_weights, proj_pitch, proj_dim, proj_shifts, proj_magnification, proj_angles, proj_count,
                volume, volume_weights, volume_pitch, volume_dim, 0.5f, 0.f, stream);
    }
}
