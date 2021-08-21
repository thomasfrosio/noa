/// \file noa/gpu/cuda/transform/Scale.h
/// \brief Scale arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 22 Jul 2021

#pragma once

#include <memory>

#include "noa/common/Definitions.h"
#include "noa/common/transform/Geometry.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Apply.h"

// -- Using textures -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D scaling/stretching.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape of \p texture and \p outputs.
    /// \param[in] scaling_factors  On the \p host. Scaling factors. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       cuda::transform::apply2D() instead.
    template<typename T>
    NOA_HOST void scale2D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* outputs, size_t output_pitch, size2_t shape,
                          const float2_t* scaling_factors, const float2_t* scaling_centers, uint nb_transforms,
                          Stream& stream) {
        if (nb_transforms == 1) {
            float23_t inv_transform(noa::transform::translate(0.5f + scaling_centers[0]) *
                                    float33_t(noa::transform::scale(1.f / scaling_factors[0])) *
                                    noa::transform::translate(-scaling_centers[0]));
            apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, inv_transform, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float23_t[]> h_inv_transforms = std::make_unique<float23_t[]>(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                h_inv_transforms[i] = float23_t(noa::transform::translate(0.5f + scaling_centers[i]) *
                                                float33_t(noa::transform::scale(1.f / scaling_factors[i])) *
                                                noa::transform::translate(-scaling_centers[i]));
            memory::PtrDevice<float23_t> d_inv_transforms(nb_transforms);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_transforms, stream);
            apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, d_inv_transforms.get(), nb_transforms, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void scale2D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* output, size_t output_pitch, size2_t shape,
                          float2_t scaling_factor, float2_t scaling_center, Stream& stream) {
        float23_t inv_transform(noa::transform::translate(0.5f + scaling_center) *
                                float33_t(noa::transform::scale(1.f / scaling_factor)) *
                                noa::transform::translate(-scaling_center));
        apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                       output, output_pitch, shape, inv_transform, stream);
    }

    /// Applies one or multiple 3D scaling/stretching.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium, slow} shape of \p texture and \p outputs.
    /// \param[in] scaling_factors  On the \p host. Scaling factors. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       cuda::transform::apply3D() instead.
    template<typename T>
    NOA_HOST void scale3D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* outputs, size_t output_pitch, size3_t shape,
                          const float3_t* scaling_factors, const float3_t* scaling_centers, uint nb_transforms,
                          Stream& stream) {
        if (nb_transforms == 1) {
            float34_t inv_transform(noa::transform::translate(0.5f + scaling_centers[0]) *
                                    float44_t(noa::transform::scale(1.f / scaling_factors[0])) *
                                    noa::transform::translate(-scaling_centers[0]));
            apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, inv_transform, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float34_t[]> h_inv_transforms = std::make_unique<float34_t[]>(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i)
                h_inv_transforms[i] = float34_t(noa::transform::translate(0.5f + scaling_centers[i]) *
                                                float44_t(noa::transform::scale(1.f / scaling_factors[i])) *
                                                noa::transform::translate(-scaling_centers[i]));
            memory::PtrDevice<float34_t> d_inv_transforms(nb_transforms);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_transforms, stream);
            apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, d_inv_transforms.get(), nb_transforms, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 3D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void scale3D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                          T* output, size_t output_pitch, size3_t shape,
                          float3_t scaling_factor, float3_t scaling_center, Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + scaling_center) *
                                float44_t(noa::transform::scale(1.f / scaling_factor)) *
                                noa::transform::translate(-scaling_center));
        apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                       output, output_pitch, shape, inv_transform, stream);
    }
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D scaling/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter2D(), which is then used as input for the interpolation.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \p host. Scaling factors. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or
    ///                             BORDER_MIRROR. BORDER_PERIODIC and BORDER_MIRROR are only supported with
    ///                             INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       cuda::transform::apply2D() instead.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                          const float2_t* scaling_factors, const float2_t* scaling_centers, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        if (nb_transforms == 1) {
            float23_t inv_transform(noa::transform::translate(0.5f + scaling_centers[0]) *
                                    float33_t(noa::transform::scale(1.f / scaling_factors[0])) *
                                    noa::transform::translate(-scaling_centers[0]));
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      inv_transform, interp_mode, border_mode, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float23_t[]> h_inv_transforms = std::make_unique<float23_t[]>(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i) {
                h_inv_transforms[i] = float23_t(noa::transform::translate(0.5f + scaling_centers[i]) *
                                                float33_t(noa::transform::scale(1.f / scaling_factors[i])) *
                                                noa::transform::translate(-scaling_centers[i]));
            }
            memory::PtrDevice<float23_t> d_inv_transforms(nb_transforms);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_transforms, stream);
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      d_inv_transforms.get(), nb_transforms, interp_mode, border_mode, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                          float2_t scaling_factor, float2_t scaling_center,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        float23_t inv_transform(noa::transform::translate(0.5f + scaling_center) *
                                float33_t(noa::transform::scale(1.f / scaling_factor)) *
                                noa::transform::translate(-scaling_center));
        apply2D<PREFILTER, false>(input, input_pitch, shape, output, output_pitch, shape,
                                  inv_transform, interp_mode, border_mode, stream);
    }

    /// Applies one or multiple 3D scaling/stretching.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter3D(), which is then used as input for the interpolation.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per transformation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium, slow} shape of \p input and \p outputs.
    /// \param[in] scaling_factors  On the \p host. Scaling factors. One per transformation.
    /// \param[in] scaling_centers  On the \p host. Scaling centers in \p input. One per transformation.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. Any of InterpMode.
    /// \param border_mode          Border/address mode. Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or
    ///                             BORDER_MIRROR. BORDER_PERIODIC and BORDER_MIRROR are only supported with
    ///                             INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \note If the input and output window are meant to have different shapes and/or centers, use
    ///       cuda::transform::apply3D() instead.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                          const float3_t* scaling_factors, const float3_t* scaling_centers, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        if (nb_transforms == 1) {
            float34_t inv_transform(noa::transform::translate(0.5f + scaling_centers[0]) *
                                    float44_t(noa::transform::scale(1.f / scaling_factors[0])) *
                                    noa::transform::translate(-scaling_centers[0]));
            apply3D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      inv_transform, interp_mode, border_mode, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float34_t[]> h_inv_transforms = std::make_unique<float34_t[]>(nb_transforms);
            for (uint i = 0; i < nb_transforms; ++i) {
                h_inv_transforms[i] = float34_t(noa::transform::translate(0.5f + scaling_centers[i]) *
                                                float44_t(noa::transform::scale(1.f / scaling_factors[i])) *
                                                noa::transform::translate(-scaling_centers[i]));
            }
            memory::PtrDevice<float34_t> d_inv_transforms(nb_transforms);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_transforms, stream);
            apply3D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      d_inv_transforms.get(), nb_transforms, interp_mode, border_mode, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 3D scaling/stretching.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void scale3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                          float3_t scaling_factor, float3_t scaling_center,
                          InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + scaling_center) *
                                float44_t(noa::transform::scale(1.f / scaling_factor)) *
                                noa::transform::translate(-scaling_center));
        apply3D<PREFILTER, false>(input, input_pitch, shape, output, output_pitch, shape,
                                  inv_transform, interp_mode, border_mode, stream);
    }
}
