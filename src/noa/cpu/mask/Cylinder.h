/// \file noa/cpu/mask/Cylinder.h
/// \brief Cylinder masks.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::mask {
    /// Applies a cylindrical mask to the input array(s).
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the cylinder is removed.
    /// \tparam T           float or double.
    /// \param[in] inputs   Input arrays. One per batch.
    /// \param[out] outputs Output arrays. One per batch. \a inputs == \a outputs is valid.
    /// \param shape        Physical {fast, medium, slow} shape of \a inputs and \a outputs (does not include the batch).
    /// \param shifts       Shifts relative to the center of the \a shape (the center is at shape / 2).
    ///                     Positive shifts translate the cylinder to the right.
    /// \param radius_xy    Radius of the cylinder.
    /// \param radius_z     Length of the cylinder.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    ///                     For instance: taper_size = 6 gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
    /// \param batches      Number of batches.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(const T* inputs, T* outputs, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z, float taper_size, uint batches);

    /// Computes a cylindrical mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(T* output_mask, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z, float taper_size);
}
