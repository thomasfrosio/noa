/// \file noa/cpu/mask/Shape.h
/// \brief Shape masks.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::filter {
    /// Returns or applies a spherical mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the sphere is removed.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch. If nullptr, write mask in \p outputs.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of batches to mask.
    /// \param center           Center of the sphere, starting from 0, in elements.
    /// \param radius           Radius, in \p T elements, of the sphere.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                         size_t batches, float3_t center, float radius, float taper_size, Stream& stream);

    /// Returns or applies a rectangular mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch. If nullptr, write mask in \p outputs.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the rectangle, in \p T elements.
    /// \param radius           Radius, in \p T elements, of the rectangle.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                            size_t batches, float3_t center, float3_t radius, float taper_size, Stream& stream);

    /// Returns or applies a cylindrical mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the cylinder is removed.
    /// \tparam T               half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch. If nullptr, write mask in \p outputs.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the cylinder, in \p T elements.
    /// \param radius_xy        Radius of the cylinder.
    /// \param radius_z         Length of the cylinder.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                           size_t batches, float3_t center, float radius_xy, float radius_z, float taper_size,
                           Stream& stream);
}
