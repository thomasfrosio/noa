/// \file noa/cpu/mask/Sphere.h
/// \brief Sphere masks.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace noa::mask {
    /// Applies a spherical mask to the input array(s).
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the sphere is removed.
    /// \tparam T           float or double.
    /// \param[in] inputs   Input arrays. One per batch.
    /// \param[out] outputs Output arrays. One per batch. \a inputs == \a outputs is valid.
    /// \param shape        Physical {fast, medium, slow} shape of \a inputs and \a outputs (does not include the batch).
    /// \param shifts       Shifts relative to the center of the \a shape (the center is at shape / 2).
    ///                     Positive shifts translate the sphere to the right.
    ///                     If \a shape describes a 2D array, \a shift.z is ignored.
    /// \param radius       Radius of the sphere.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    ///                     For instance: taper_size = 6 gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
    /// \param batches      Number of batches.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere(const T* inputs, T* outputs, size3_t shape, float3_t shifts,
                         float radius, float taper_size, uint batches);

    /// Computes a spherical mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere(T* output_mask, size3_t shape, float3_t shifts, float radius, float taper_size);
}
