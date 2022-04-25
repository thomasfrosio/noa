/// \file noa/gpu/cuda/reconstruct/ProjectForward.h
/// \brief Forward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"

// -- Using textures -- //
namespace noa::cuda::reconstruct {
    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// \param volume_interp_mode Interpolation/filter mode of the \p volume texture.
    ///                           It should be INTERP_LINEAR or INTERP_LINEAR_FAST, otherwise an error is thrown.
    /// \see For more details, see the overloads without textures below.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode, size_t volume_dim,
                                 T* proj, size_t proj_pitch, size_t proj_dim,
                                 const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max, float ewald_sphere_radius, Stream& stream);

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode, size_t volume_dim,
                                 T* proj, size_t proj_pitch, size_t proj_dim,
                                 float3_t proj_scaling_factor, const float33_t* proj_rotations,
                                 const float2_t* proj_shifts, uint proj_count,
                                 float freq_max, float ewald_sphere_radius, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::reconstruct {
    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// \details This is the reverse operation of the backward projection. However, the transformation itself is
    ///          identical, but instead of adding the projections at the transformed coordinates, a simple forward
    ///          linear interpolation is performed to extract the values from the volume. As such, to extract a
    ///          projection inserted during the backward projection, the same rotation and scaling should be used here.
    /// \see projectBackward for more details.
    ///
    /// \tparam IS_PROJ_CENTERED        Whether or not the output \p proj should be centered.
    ///                                 See "noa/cpu/fourier/Resize.h" for more details on FFT layouts.
    /// \tparam IS_VOLUME_CENTERED      Whether or not the input \p volume is centered.
    ///                                 Only centered volumes are currently supported.
    /// \tparam T                       float, double, cfloat_t or cdouble_t.
    /// \param[in] volume               On the \b host or \b device. Non-redundant volume from which the projections are extracted.
    /// \param volume_dim               Logical dimension size, in elements, of \p volume.
    /// \param[out] proj                On the \b device. Extracted, non-redundant projections. One per projection.
    /// \param proj_pitch               Pitch, in elements, of \p proj.
    /// \param proj_dim                 Logical dimension size, in elements, of \p proj.
    ///
    /// \param[in] proj_scaling_factors On the \b host. If nullptr, it is ignored. One per projection.
    ///                                 2D real-space scaling to apply to the projection before the rotation.
    ///                                 The third value is the in-plane scaling angle, in radians.
    /// \param[in] proj_rotations       On the \b host. 3x3 rotation matrices of the projections. One per projection.
    ///                                 The rotation center is on the DC component.
    ///                                 See "noa/common/transform/README.txt" for more details on the conventions.
    /// \param[in] proj_shifts          On the \b device. If nullptr or if \p T is real, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply (as phase-shifts) to the projection after extraction.
    ///                                 Positive values shift the real-space object to the right.
    ///                                 Usually, this is the opposite value than the one used during backward projection.
    /// \param proj_count               Number of projections to extract.
    /// \param freq_max                 Maximum frequency to insert. Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ewald_sphere_radius      Ewald sphere radius, in 1/pixel (e.g. `pixel_size/wavelength`).
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///                                 The stream is synchronized when the function returns.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_pitch, size_t volume_dim,
                               T* proj, size_t proj_pitch, size_t proj_dim,
                               const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                               const float2_t* proj_shifts, uint proj_count,
                               float freq_max, float ewald_sphere_radius, Stream& stream) {
        memory::PtrArray<T> array(size3_t{proj_dim / 2 + 1, proj_dim, proj_dim}); // non-redundant
        memory::copy(volume, volume_pitch, array.get(), array.shape(), stream);
        memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        projectForward(texture.get(), INTERP_LINEAR, volume_dim, proj, proj_pitch, proj_dim,
                       proj_scaling_factors, proj_rotations, proj_shifts, proj_count, freq_max, ewald_sphere_radius);
        stream.synchronize();
    }

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_pitch, size_t volume_dim,
                               T* proj, size_t proj_pitch, size_t proj_dim,
                               float3_t proj_scaling_factor, const float33_t* proj_rotations,
                               const float2_t* proj_shifts, uint proj_count,
                               float freq_max, float ewald_sphere_radius, Stream& stream) {
        memory::PtrArray<T> array(size3_t{proj_dim / 2 + 1, proj_dim, proj_dim}); // non-redundant
        memory::copy(volume, volume_pitch, array.get(), array.shape(), stream);
        memory::PtrTexture texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        projectForward(texture.get(), INTERP_LINEAR, volume_dim, proj, proj_pitch, proj_dim,
                       proj_scaling_factor, proj_rotations, proj_shifts, proj_count, freq_max, ewald_sphere_radius);
        stream.synchronize();
    }
}

// -- Max to Nyquist, flat EWS -- //
namespace noa::cuda::reconstruct {
    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                               const float3_t* proj_scaling_factors, const float33_t* proj_rotations,
                               const float2_t* proj_shifts, uint proj_count, Stream& stream) {
        projectForward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(volume, volume_dim, proj, proj_dim, proj_scaling_factors,
                                                             proj_rotations, proj_shifts, proj_count,
                                                             0.5f, 0.f, stream);
    }

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    /// Overload which sets the same magnification correction for all projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                               float3_t proj_scaling_factor, const float33_t* proj_rotations,
                               const float2_t* proj_shifts, uint proj_count, Stream& stream) {
        projectForward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(volume, volume_dim, proj, proj_dim, proj_scaling_factor,
                                                             proj_rotations, proj_shifts, proj_count,
                                                             0.5f, 0.f, stream);
    }
}
