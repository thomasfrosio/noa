/// \file noa/gpu/cuda/reconstruct/ProjectForward.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
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
    NOA_HOST void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                                 size_t volume_pitch, size_t volume_dim,
                                 T* proj, size_t proj_pitch, size_t proj_dim, const float33_t* proj_rotations,
                                 const float3_t* proj_magnifications, const float2_t* proj_shifts, uint proj_count,
                                 float freq_max, float ewald_sphere_radius, Stream& stream);

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_HOST void projectForward(cudaTextureObject_t volume, InterpMode volume_interp_mode,
                                 size_t volume_pitch, size_t volume_dim,
                                 T* proj, size_t proj_pitch, size_t proj_dim, const float33_t* proj_rotations,
                                 float3_t proj_magnification, const float2_t* proj_shifts, uint proj_count,
                                 float freq_max, float ewald_sphere_radius, Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::reconstruct {
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
    /// \param[in] volume               On the \b host or \b device. Non-redundant volume from which the projections are extracted.
    /// \param volume_dim               Logical dimension size, in elements, of \p volume.
    /// \param[out] proj                On the \b device. Extracted, non-redundant projections. One per projection.
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
    /// \param[in] proj_shifts          On the \b device. If nullptr or if \p T is real, it is ignored. One per projection.
    ///                                 2D real-space shifts to apply (as phase-shifts) to the projection after
    ///                                 the extraction.
    /// \param proj_count               Number of projections to extract.
    /// \param freq_max                 Maximum frequency to extract. Values are clamped from 0 to 0.5.
    /// \param ewald_sphere_radius      Ewald sphere radius (i.e. `1/wavelength`, in SI) of the projections.
    ///                                 If negative, the negative curve is computed.
    ///                                 If 0.f, the "projections" are assumed to be actual projections, i.e. slices.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///                                 The stream is synchronized when the function returns.
    ///
    /// \note Only square projections and cubic volumes are supported. The volume is usually oversampled compared to
    ///       the projections. The oversampling ratio is set to be the ratio between \p volume_dim and \p proj_dim.
    ///       Note that both the projections and the volume are non-redundant transforms, so the physical size in
    ///       the first dimension is `x/2+1`, x being the logical size, i.e. \p volume_dim or \p proj_dim.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_pitch, size_t volume_dim,
                               T* proj, size_t proj_pitch, size_t proj_dim,
                               const float33_t* proj_rotations, const float3_t* proj_magnifications,
                               const float2_t* proj_shifts, uint proj_count,
                               float freq_max, float ewald_sphere_radius, Stream& stream) {
        memory::PtrArray<T> array(size3_t{proj_dim / 2 + 1, proj_dim, proj_dim}); // non-redundant
        memory::copy(volume, volume_pitch, array.get(), array.shape(), stream);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        projectForward(texture.get(), INTERP_LINEAR, volume_pitch, volume_dim, proj, proj_pitch, proj_dim,
                       proj_rotations, proj_magnifications, proj_shifts, proj_count, freq_max, ewald_sphere_radius);
        stream.synchronize();
    }

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Same as above, except that the same magnification is applied to all of the projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_pitch, size_t volume_dim,
                               T* proj, size_t proj_pitch, size_t proj_dim,
                               const float33_t* proj_rotations, float3_t proj_magnification,
                               const float2_t* proj_shifts, uint proj_count,
                               float freq_max, float ewald_sphere_radius, Stream& stream) {
        memory::PtrArray<T> array(size3_t{proj_dim / 2 + 1, proj_dim, proj_dim}); // non-redundant
        memory::copy(volume, volume_pitch, array.get(), array.shape(), stream);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        projectForward(texture.get(), INTERP_LINEAR, volume_pitch, volume_dim, proj, proj_pitch, proj_dim,
                       proj_rotations, proj_magnification, proj_shifts, proj_count, freq_max, ewald_sphere_radius);
        stream.synchronize();
    }
}

// -- Max to Nyquist, flat EWS -- //
namespace noa::cuda::reconstruct {
    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                               const float33_t* proj_rotations, const float3_t* proj_magnifications,
                               const float2_t* proj_shifts, uint proj_count, Stream& stream) {
        projectForward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(volume, volume_dim, proj, proj_dim, proj_rotations,
                                                             proj_magnifications, proj_shifts, proj_count,
                                                             0.5f, 0.f, stream);
    }

    /// Extracts Fourier "slices" from a Fourier volume using tri-linear interpolation.
    /// Overload which sets the maximum frequency to Nyquist and uses a "flat" Ewald sphere.
    /// Overload which sets the same magnification correction for all projections.
    template<bool IS_PROJ_CENTERED = false, bool IS_VOLUME_CENTERED = true, typename T>
    NOA_IH void projectForward(const T* volume, size_t volume_dim, T* proj, size_t proj_dim,
                               const float33_t* proj_rotations, float3_t proj_magnification,
                               const float2_t* proj_shifts, uint proj_count, Stream& stream) {
        projectForward<IS_PROJ_CENTERED, IS_VOLUME_CENTERED>(volume, volume_dim, proj, proj_dim, proj_rotations,
                                                             proj_magnification, proj_shifts, proj_count,
                                                             0.5f, 0.f, stream);
    }
}
