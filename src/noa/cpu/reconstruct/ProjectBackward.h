/// \file noa/cpu/reconstruct/ProjectBackward.h
/// \brief Backward projections.
/// \author Thomas - ffyr2w
/// \date 25 Aug 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/common/View.h"

namespace noa::cpu::reconstruct::fft {
    using Remap = noa::fft::Remap;

    /// Inserts 2D Fourier slice(s) into a 3D Fourier volume, using tri-linear interpolation.
    /// \details The slices are phase shifted, their magnification is corrected and the EWS curvature is applied.
    ///          Then, they are rotated and added to the cartesian (oversampled) 3D volume using tri-linear interpolation.
    ///
    /// \tparam REMAP           Remapping from the slice layout to the volume layout. Should be H2H, H2HC, HC2H, HC2HC.
    /// \tparam C1,C0           cfloat_t or cdouble_t. \p C1 can be const-qualified.
    /// \tparam R1,R0           float or double. \p R1 can be const-qualified.
    /// \param[in] slice        On the \b host. Non-redundant 2D slice(s) to insert.
    /// \param[in] slice_weight On the \b host. Element-wise 2D weight(s) associated with \p slice.
    /// \param[out] grid        On the \b host. Non-redundant 3D grid inside which the slices are inserted.
    /// \param[out] grid_weight On the \b host. Element-wise 3D weight associated with \p grid.
    /// \param[in] shifts       On the \b host. 2D real-space shifts to apply (as phase-shifts) to the projection
    ///                         before any other transformation. If nullptr, it is ignored. One per slice.
    /// \param[in] scales       On the \b host. 2x2 rightmost forward real-space scaling to apply to the
    ///                         slices before the rotation. If nullptr, it is ignored. One per slice.
    /// \param[in] rotations    On the \b host. 3x3 rightmost forward rotation matrices. One per slice.
    /// \param cutoff           Frequency cutoff in \p grid and \p grid_weight, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    /// \param ews_radius       Rightmost ewald sphere radius, in 1/pixels (i.e. pixel_size / wavelength).
    ///                         If negative, the negative curve is computed.
    ///                         If {0,0}, the slices are projections.
    ///
    /// \note Since a rotation is applied to the projections, they should have their real-space rotation-center
    ///       phase-shifted at 0, as with any rotation applied in Fourier space. If this is not the case,
    ///       \p shifts can be used to properly phase shift the projection before the rotation.
    /// \note To decrease artefacts, the output cartesian grid is usually oversampled compared to the input slices.
    ///       The oversampling ratio is set to be the ratio between the two innermost dimensions of \p grid_shape and
    ///       \p slice_shape.
    /// \note In order to have both left and right beams assigned to different values, this function only computes one
    ///       "side" of the EWS, as specified by \p ews_radius. To insert the other side, one would have to
    ///       call this function a second time with \p ews_radius * -1.
    template<Remap REMAP, typename C0, typename C1, typename R0, typename R1>
    NOA_HOST void insert(const View<C0>& slice, const View<R0>& slice_weight,
                         const View<C1>& grid, const View<R1>& grid_weight,
                         const float2_t* shifts, const float22_t* scales, const float33_t* rotations,
                         float cutoff, float2_t ews_radius, Stream& stream);

    /// Corrects for the gridding, assuming tri-linear interpolation was used during the Fourier insertion.
    /// \tparam T
    /// \param[in] input
    /// \param[out] output
    /// \param stream
    template<typename T>
    NOA_HOST void postCompensation(const View<const T>& input, const View<T>& output, Stream& stream);
}
