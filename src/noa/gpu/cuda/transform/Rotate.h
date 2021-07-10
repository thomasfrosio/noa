#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/transform/Interpolate.h"

namespace noa::cuda::transform {
    // -- Texture versions of rotate -- //

    ///
    /// \param texture
    /// \param output
    /// \param output_pitch
    /// \param shape
    /// \param rotation
    /// \param rotation_center
    /// \param interp_mode
    /// \param stream
    /// \note This function is asynchronous relative to the host and may return before completion.
    NOA_HOST void rotate(cudaTextureObject_t texture, float* output, size_t output_pitch, size3_t shape,
                         float3_t rotation, float3_t rotation_center, InterpMode interp_mode, Stream& stream);

    // -- Simplest versions of rotate -- //

    /// 3D rotates the input array, either in-place or out-of-place.
    /// \param[in] input        Input array.
    /// \param input_pitch      Pitch, in elements, of \a input.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param output_pitch     Pitch, in elements, of \a output.
    /// \param shape            Logical {fast, medium, slow} shape of \a input and \a output.
    /// \param rotation         ZYZ euler angles, in radians.
    /// \param rotation_center  Rotation center within \a shape.
    /// \param interp_mode      Any of the interpolation mode in \a InterpMode.
    /// \param border_mode      BORDER_CLAMP or BORDER_ZERO.
    ///                         If \a interp_mode is INTERP_(NEAREST|LINEAR), BORDER_(PERIODIC|MIRROR) are also supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/transform/Euler.h" and "noa/common/transform/Geometry.h" for more details on the conventions.
    NOA_IH void rotate(const float* input, size_t input_pitch, float* output, size_t output_pitch,
                       size3_t shape, float3_t rotation, float3_t rotation_center,
                       InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        memory::PtrArray<float> i_array(shape);
        memory::PtrTexture<float> i_texture;

        if (interp_mode == INTERP_CUBIC_BSPLINE) {
            bspline::prefilter(input, input_pitch, output, output_pitch, shape, 1, stream);
            memory::copy(output, output_pitch, i_array.get(), shape, stream);
        } else {
            memory::copy(input, input_pitch, i_array.get(), shape, stream);
        }
        stream.synchronize();
        i_texture.reset(i_array.get(), interp_mode, border_mode);

        rotate(i_texture.get(), output, output_pitch, shape, rotation, rotation_center, interp_mode, stream);
        stream.synchronize();
    }

    /// 2D rotates the input array, either in-place or out-of-place.
    /// \param[in] input        Input array.
    /// \param input_pitch      Pitch, in elements, of \a input.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param output_pitch     Pitch, in elements, of \a output.
    /// \param shape            Logical {fast, medium, slow} shape of \a input and \a output.
    /// \param angle            Angle, in radians.
    /// \param interp_mode      Any of the interpolation mode in \a InterpMode.
    /// \param border_mode      BORDER_CLAMP or BORDER_ZERO.
    ///                         If \a interp_mode is INTERP_(NEAREST|LINEAR), BORDER_(PERIODIC|MIRROR) are also supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \see noa/common/transform/Geometry.h for more details on the followed conventions.
    NOA_HOST void rotate(const float* input, size_t input_pitch, float* output, size_t output_pitch, size3_t shape,
                         float angle, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// 3D rotates the input array, either in-place or out-of-place. Version for contiguous layouts.
    NOA_IH void rotate(const float* input, float* output, size3_t shape,
                       float3_t angles, InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        rotate(input, shape.x, output, shape.x, shape, angles, interp_mode, border_mode, stream);
    }

    /// 2D rotates the input array, either in-place or out-of-place. Version for contiguous layouts.
    NOA_HOST void rotate(const float* input, float* output, size3_t shape,
                         float angle, InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        rotate(input, shape.x, output, shape.x, shape, angle, interp_mode, border_mode, stream);
    }

    /// Computes multiple 3D rotations of the input array. Each rotation is saved in an output array.
    /// \tparam FFT_LAYOUT
    /// \tparam T
    /// \param input
    /// \param input_pitch
    /// \param outputs
    /// \param outputs_pitch
    /// \param shape
    /// \param rotations
    /// \param rotation_centers
    /// \param nb_rotations
    template<typename T>
    NOA_HOST void rotate(const T* input, size_t input_pitch, T* outputs, size_t outputs_pitch, size3_t shape,
                         const float3_t* rotations, const float3_t* rotation_centers, uint nb_rotations,
                         InterpMode interp_mode, BorderMode border_mode, Stream& stream);
}

namespace noa::cuda::transform::fourier {

}
