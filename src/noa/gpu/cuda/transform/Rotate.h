/// \file noa/gpu/cuda/transform/Rotate.h
/// \brief Rotate arrays or CUDA textures.
/// \author Thomas - ffyr2w
/// \date 22 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"
#include "noa/cpu/memory/PtrHost.h"

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
    /// \param[in] rotations        On the \p host. Rotation angles, in radians. One per rotation.
    /// \param[in] rotation_centers On the \p host. Rotation centers in \p input. One per rotation.
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
        if (nb_rotations == 1) {
            // On the output it does: add the -0.5 offset to account for texture offset, translate rotation center
            // to the origin, rotate, translate back to the rotation center. Of course here, take the invert of that.
            float23_t inv_transform(noa::transform::translate(0.5f + rotation_centers[0]) *
                                    float33_t(noa::transform::rotate(-rotations[0])) *
                                    noa::transform::translate(-rotation_centers[0]));

            apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, inv_transform, stream); // TEXTURE_OFFSET = false
            stream.synchronize(); // sync even if you don't have to
        } else {
            noa::memory::PtrHost<float23_t> h_inv_transforms(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i) {
                h_inv_transforms[i] = float23_t(noa::transform::translate(0.5f + rotation_centers[i]) *
                                                float33_t(noa::transform::rotate(-rotations[i])) *
                                                noa::transform::translate(-rotation_centers[i]));
            }
            memory::PtrDevice<float23_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply2D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, d_inv_transforms.get(), nb_rotations, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void rotate2D(cudaTextureObject_t texture, InterpMode texture_interp_mode, BorderMode texture_border_mode,
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
    /// \param[in] rotations        On the \p host. ZYZ Euler angles, in radians. One trio per rotation.
    /// \param[in] rotation_centers On the \p host. Rotation centers in \p input. One per rotation.
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
                           const float3_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                           Stream& stream) {
        if (nb_rotations == 1) {
            float34_t inv_transform(noa::transform::translate(0.5f + rotation_centers[0]) *
                                    float44_t(noa::transform::toMatrix<true>(rotations[0])) *
                                    noa::transform::translate(-rotation_centers[0]));
            apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
                           outputs, output_pitch, shape, inv_transform, stream);
            stream.synchronize();
        } else {
            noa::memory::PtrHost<float34_t> h_inv_transforms(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i) {
                h_inv_transforms[i] = float34_t(noa::transform::translate(0.5f + rotation_centers[i]) *
                                                float44_t(noa::transform::toMatrix<true>(rotations[i])) *
                                                noa::transform::translate(-rotation_centers[i]));
            }
            memory::PtrDevice<float34_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply3D<false>(texture, shape, texture_interp_mode, texture_border_mode,
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
                           float3_t rotation, float3_t rotation_center, Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + rotation_center) *
                                float44_t(noa::transform::toMatrix<true>(rotation)) *
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
        if (nb_rotations == 1) {
            float23_t transform(noa::transform::translate(0.5f + rotation_centers[0]) *
                                float33_t(noa::transform::rotate(-rotations[0])) *
                                noa::transform::translate(-rotation_centers[0]));
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      transform, interp_mode, border_mode, stream);
            stream.synchronize();
        } else {
            noa::memory::PtrHost<float23_t> h_inv_transforms(nb_rotations);
            for (uint i = 0; i < nb_rotations; ++i) {
                h_inv_transforms[i] = float23_t(noa::transform::translate(0.5f + rotation_centers[i]) *
                                                float33_t(noa::transform::rotate(-rotations[i])) *
                                                noa::transform::translate(-rotation_centers[i]));
            }
            memory::PtrDevice<float23_t> d_inv_transforms(nb_rotations);
            memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
            apply2D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                      d_inv_transforms.get(), nb_rotations, interp_mode, border_mode, stream);
            stream.synchronize();
        }
    }

    /// Applies a single 2D rotation.
    /// \see This function has the same features and limitations than the overload above.
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
    /// \param[in] rotations        On the \b host. ZYZ Euler angles, in radians. One trio per rotation.
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
                           const float3_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                           InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        noa::memory::PtrHost<float34_t> h_inv_transforms(nb_rotations);
        for (uint i = 0; i < nb_rotations; ++i) {
            h_inv_transforms[i] = float34_t(noa::transform::translate(0.5f + rotation_centers[i]) *
                                            float44_t(noa::transform::toMatrix<true>(rotations[i])) *
                                            noa::transform::translate(-rotation_centers[i]));
        }
        memory::PtrDevice<float34_t> d_inv_transforms(nb_rotations);
        memory::copy(h_inv_transforms.get(), d_inv_transforms.get(), nb_rotations, stream);
        apply3D<PREFILTER, false>(input, input_pitch, shape, outputs, output_pitch, shape,
                                  d_inv_transforms.get(), nb_rotations, interp_mode, border_mode, stream);
        stream.synchronize();
    }

    /// Applies a single 3D rotation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void rotate3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                           float3_t rotation, float3_t rotation_center, InterpMode interp_mode, BorderMode border_mode,
                           Stream& stream) {
        float34_t inv_transform(noa::transform::translate(0.5f + rotation_center) *
                                float44_t(noa::transform::toMatrix<true>(rotation)) *
                                noa::transform::translate(-rotation_center));
        apply3D<PREFILTER, false>(input, input_pitch, shape, output, output_pitch, shape,
                                  inv_transform, interp_mode, border_mode, stream);
    }
}

// -- Using texture - for centered Fourier transforms -- //
namespace noa::cuda::transform {
    /// Applies one or multiple 2D rotations to a non-redundant centered Fourier transform.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p output should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param texture          Non-redundant and centered transform.
    ///                         Should use the INTERP_NEAREST, INTERP_COSINE or INTERP_LINEAR filtering mode.
    ///                         Should use the BORDER_ZERO filtering mode (although it doesn't effect on \p output).
    ///                         Un-normalized coordinates should be used.
    ///                         It's center of rotation is expected to be at index `(0, shape/2)`.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p texture and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        On the \b host. Rotation angles, in radians. One per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 2D rotation to a non-redundant centered Fourier transform.
    /// \see This function has the same features and limitations than the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             float rotation, float freq_cutoff, Stream& stream);

    /// Applies one or multiple 3D rotations to a non-redundant centered Fourier transform.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p output should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param texture          Non-redundant and centered transform.
    ///                         Should use the INTERP_NEAREST, INTERP_COSINE or INTERP_LINEAR filtering mode.
    ///                         Should use the BORDER_ZERO filtering mode (although it doesn't effect on \p output).
    ///                         Un-normalized coordinates should be used.
    ///                         It's center of rotation is expected to be at index `(0, shape/2, shape/2)`.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p texture and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        On the \b host. ZYZ Euler angles, in radians. One trio per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(cudaTextureObject_t texture, T* outputs, size_t output_pitch, size_t shape,
                             const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 3D rotation to a non-redundant centered Fourier transform.
    /// \see This function has the same features and limitations than the overload above. The only difference is that
    ///      since it computes a single rotation, it doesn't need to allocate a temporary array to store the rotation
    ///      matrices. As such, this function is asynchronous relative to the host and may return before completion.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                             float3_t rotation, float freq_cutoff, Stream& stream);
}

// -- Using arrays - for centered Fourier transforms -- //
namespace noa::cuda::transform {
    /// Rotates a 2D non-redundant centered Fourier transform, either inplace or out-of-place.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p outputs should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param input            On the \b device. Non-redundant and centered Fourier transform.
    ///                         It's center of rotation is expected to be at index `(0, shape/2)`.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Can be equal to \p input.
    ///                         Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p input and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        Rotation angles, in radians. One per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                             const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 2D rotation to a non-redundant centered Fourier transform.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate2DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                             float rotation, float freq_cutoff, Stream& stream);

    /// Rotates a 3D non-redundant centered Fourier transform, either inplace or out-of-place.
    /// \tparam T               float or cfloat_t. With cfloat_t, \a texture should have its descriptor set to float2.
    /// \tparam REMAP           Whether or not \p outputs should be remapped to a non-centered layout (i.e. ifftshift)
    ///                         so that it can be passed directly to the c2r routines.
    /// \param input            On the \b device. Non-redundant and centered Fourier transform.
    ///                         It's center of rotation is expected to be at index `(0, shape/2, shape/2)`.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param outputs          On the \b device. Non-redundant rotated, using bilinear interpolation, Fourier transform.
    ///                         One per rotation. Can be equal to \p input.
    ///                         Output arrays can be centered or non-centered depending on \p REMAP.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical shape of \p input and \p outputs (one per rotation).
    ///                         All dimensions should therefore have the same logical size.
    /// \param rotations        ZYZ Euler angles, in radians. One trio per rotation.
    /// \param nb_rotations     Number of rotations.
    /// \param freq_cutoff      Frequency cutoff. From 0 to 0.5 (0.5 being the Nyquist frequency).
    /// \param stream           Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                             const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream);

    /// Applies a single 3D rotation to a non-redundant centered Fourier transform.
    template<bool REMAP = false, typename T>
    NOA_HOST void rotate3DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                             float3_t rotation, float freq_cutoff, Stream& stream);
}
