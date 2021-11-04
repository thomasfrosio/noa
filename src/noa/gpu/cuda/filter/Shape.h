/// \file noa/gpu/cuda/mask/Shape.h
/// \brief Shape masks.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::filter {
    /// Returns or applies a circular mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the sphere is removed.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch.
    ///                         If nullptr, the mask is directly written in \p outputs.
    /// \param input_pitch      Pitch, in \p T elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the circle, in \p T elements.
    /// \param radius           Radius, in \p T elements, of the circle.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                           size2_t shape, size_t batches,
                           float2_t center, float radius, float taper_size, Stream& stream);

    /// Returns or applies a spherical mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the sphere is removed.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch.
    ///                         If nullptr, the mask is directly written in \p outputs.
    /// \param input_pitch      Pitch, in \p T elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the sphere, in \p T elements.
    /// \param radius           Radius, in \p T elements, of the sphere.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                           size3_t shape, size_t batches,
                           float3_t center, float radius, float taper_size, Stream& stream);

    /// Returns or applies a rectangular mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch.
    ///                         If nullptr, the mask is directly written in \p outputs.
    /// \param input_pitch      Pitch, in \p T elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the rectangle, in \p T elements.
    /// \param radius           Radius, in \p T elements, of the rectangle.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                              size2_t shape, size_t batches,
                              float2_t center, float2_t radius, float taper_size, Stream& stream);

    /// Returns or applies a rectangular mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch.
    ///                         If nullptr, the mask is directly written in \p outputs.
    /// \param input_pitch      Pitch, in \p T elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the rectangle, in \p T elements.
    /// \param radius           Radius, in \p T elements, of the rectangle.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                              size3_t shape, size_t batches,
                              float3_t center, float3_t radius, float taper_size, Stream& stream);

    /// Returns or applies a cylindrical mask.
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the cylinder is removed.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Array(s) to mask. One per batch.
    ///                         If nullptr, the mask is directly written in \p outputs.
    /// \param input_pitch      Pitch, in \p T elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Masked array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in \p T elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring \p batches.
    /// \param batches          Number of contiguous batches to mask.
    /// \param center           Center of the cylinder, in \p T elements.
    /// \param radius_xy        Radius of the cylinder.
    /// \param radius_z         Length of the cylinder.
    /// \param taper_size       Width, in \p T elements, of the raised-cosine, including the first zero.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                             size3_t shape, size_t batches,
                             float3_t center, float radius_xy, float radius_z, float taper_size, Stream& stream);
}