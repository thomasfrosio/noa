/// \file noa/gpu/cuda/transform/Rotate.h
/// \brief Rotate arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 22 Jul 2021

#pragma once

#include <memory>

#include "noa/common/Definitions.h"
#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"

#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Apply.h"

// -- Using textures -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D rotations.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per rotation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape of \p texture and \p outputs.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param[in] rotation_centers On the \b host. Rotation centers in \p input. One per rotation.
    /// \param nb_rotations         Number of rotations to compute.
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
    NOA_HOST void rotate2D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                           T* outputs, size_t output_pitch, size2_t shape,
                           const float* rotations, const float2_t* rotation_centers, uint nb_rotations,
                           Stream& stream) {
        // On the output it does: add the -0.5 offset to account for texture offset, translate rotation center
        // to the origin, rotate, translate back to the rotation center. Of course here, take the invert of that.
        constexpr bool TEXTURE_OFFSET = false;
        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float23_t(noa::transform::translate(0.5f + rotation_centers[index]) *
                             float33_t(noa::transform::rotate(-rotations[index])) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1) {
            apply2D<TEXTURE_OFFSET>(texture, shape, texture_interp_mode, texture_border_mode,
                                    outputs, output_pitch, shape, getInvertTransform_(0), stream);
            stream.synchronize(); // sync even if you don't have to
        } else {
            std::unique_ptr<float23_t[]> h_inv_transforms = std::make_unique<float23_t[]>(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                h_inv_transforms[i] = getInvertTransform_(i);
            memory::PtrDevice<float23_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply2D<TEXTURE_OFFSET>(texture, shape, texture_interp_mode, texture_border_mode,
                                    outputs, output_pitch, shape, d_inv_transforms.get(), nb_rotations, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void rotate2D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                         T* output, size_t output_pitch, size2_t shape,
                         float rotation, float2_t rotation_center, Stream& stream) {
        float23_t inv_transform(noa::transform::translate(0.5f + rotation_center) *
                                float33_t(noa::transform::rotate(-rotation)) *
                                noa::transform::translate(-rotation_center));
        apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                       output, output_pitch, shape, inv_transform, stream);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Border/address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] outputs         On the \b device. Output arrays. One per rotation.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium, slow} shape of \p texture and \p outputs.
    /// \param[in] rotations        On the \b host. 3x3 inverse rotation matrices. One per rotation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This function assumes \p rotations is already
    ///                             inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] rotation_centers On the \b host. Rotation centers in \p input. One per rotation.
    /// \param nb_rotations         Number of rotations to compute.
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
    NOA_HOST void rotate3D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                           T* outputs, size_t output_pitch, size3_t shape,
                           const float33_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                           Stream& stream) {
        constexpr bool TEXTURE_OFFSET = false;
        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float34_t(noa::transform::translate(0.5f + rotation_centers[index]) *
                             float44_t(rotations[index]) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1) {
            apply3D<TEXTURE_OFFSET>(texture, shape, texture_interp_mode, texture_border_mode,
                                    outputs, output_pitch, shape, getInvertTransform_(0), stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float34_t[]> h_inv_transforms = std::make_unique<float34_t[]>(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                h_inv_transforms[i] = getInvertTransform_(i);
            memory::PtrDevice<float34_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply3D<TEXTURE_OFFSET>(texture, shape, texture_interp_mode, texture_border_mode,
                                    outputs, output_pitch, shape, d_inv_transforms.get(), nb_rotations, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 3D rotation.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void rotate3D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
                           T* output, size_t output_pitch, size3_t shape,
                           float33_t rotation, float3_t rotation_center, Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + rotation_center) *
                                float44_t(rotation) *
                                noa::transform::translate(-rotation_center));
        apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                       output, output_pitch, shape, inv_transform, stream);
    }
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter2D(), which is then used as input for the interpolation.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per rotation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium} shape of \p input and \p outputs.
    /// \param[in] rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param[in] rotation_centers On the \b host. Rotation centers in \p input. One per rotation.
    /// \param nb_rotations         Number of rotations to compute.
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
    /// \note If the input and output window are meant to have different shapes
    ///       and/or centers, use cuda::transform::apply2D() instead.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                           const float* rotations, const float2_t* rotation_centers, uint nb_rotations,
                           InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float23_t(noa::transform::translate(0.5f + rotation_centers[index]) *
                             float33_t(noa::transform::rotate(-rotations[index])) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1) {
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      getInvertTransform_(0), interp_mode, border_mode, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float23_t[]> h_inv_transforms = std::make_unique<float23_t[]>(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                h_inv_transforms[i] = getInvertTransform_(i);
            memory::PtrDevice<float23_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      d_inv_transforms.get(), nb_rotations, interp_mode, border_mode, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                           float rotation, float2_t rotation_center, InterpMode interp_mode, BorderMode border_mode,
                           Stream& stream) {
        float23_t transform(noa::transform::translate(0.5f + rotation_center) *
                            float33_t(noa::transform::rotate(-rotation)) *
                            noa::transform::translate(-rotation_center));
        apply2D<PREFILTER, false>(input, input_pitch, shape, output, output_pitch, shape,
                                  transform, interp_mode, border_mode, stream);
    }

    /// Applies one or multiple 3D rotations.
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true,
    ///                             a temporary array of the same shape as \p input is allocated and used to store the
    ///                             output of bspline::prefilter3D(), which is then used as input for the interpolation.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input array. If \p PREFILTER is true and \p interp_mode is INTERP_CUBIC_BSPLINE or
    ///                             INTERP_CUBIC_BSPLINE_FAST, should be on the \b device. Otherwise, can be on the
    ///                             \b host or \b device.
    /// \param input_pitch          Pitch, in elements, of \p input.
    /// \param[out] outputs         On the \b device. Output arrays. One per rotation. Can be equal to \p input.
    /// \param output_pitch         Pitch, in elements, of \p outputs.
    /// \param shape                Logical {fast, medium, slow} shape of \p input and \p outputs.
    /// \param[in] rotations        On the \b host. 3x3 inverse rotation matrices. One per rotation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This function assumes \p rotations is already
    ///                             inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param[in] rotation_centers On the \b host. Rotation centers in \p input. One per rotation.
    /// \param nb_rotations         Number of rotations to compute.
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
    NOA_HOST void rotate3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                           const float33_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                           InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        auto getInvertTransform_ = [rotations, rotation_centers](uint index) {
            return float34_t(noa::transform::translate(0.5f + rotation_centers[index]) *
                             float44_t(rotations[index]) *
                             noa::transform::translate(-rotation_centers[index]));
        };

        if (nb_rotations == 1) {
            apply3D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      getInvertTransform_(0), interp_mode, border_mode, stream);
            stream.synchronize();
        } else {
            std::unique_ptr<float34_t[]> h_inv_transforms = std::make_unique<float34_t[]>(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i)
                h_inv_transforms[i] = getInvertTransform_(i);
            memory::PtrDevice<float34_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply3D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      d_inv_transforms.get(), nb_rotations, interp_mode, border_mode, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 3D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                           float33_t rotation, float3_t rotation_center, InterpMode interp_mode, BorderMode border_mode,
                           Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + rotation_center) *
                                float44_t(rotation) *
                                noa::transform::translate(-rotation_center));
        apply3D<PREFILTER, false>(input, input_pitch, shape, output, output_pitch, shape,
                                  inv_transform, interp_mode, border_mode, stream);
    }
}
