/// \file noa/cpu/mask/Rectangle.h
/// \brief Rectangle masks.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::filter {
    /// Applies a rectangular mask to the input array(s).
    /// \tparam INVERT      Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T           float or double.
    /// \param[in] inputs   On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] outputs On the \b host. Contiguous output arrays. One per batch. Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param shifts       Shifts, in elements, corresponding to \p shape.
    ///                     Positive shifts translate the rectangle to the right.
    ///                     The center of the rectangle without shifts is at shape / 2.
    ///                     If \p shape describes a 2D array, \p shift.z is ignored.
    /// \param radius       Radii, in elements, of the rectangle.
    /// \param taper_size   Width, in elements, of the raised-cosine, including the first zero.
    ///                     For instance: taper_size = 6, gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
    /// \param batches      Number of batches.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                            float taper_size, uint batches);

    /// Computes a rectangular mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius, float taper_size);
}
