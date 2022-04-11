#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry::fft {
    using Remap = noa::fft::Remap;

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant 2D FFT to transform.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant transformed 2D FFT. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    ///                         The outermost dimension is the batch.
    /// \param[in] matrices     On the \b host or \b device. 2x2 inverse rightmost rotation/scaling matrix. One per batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host or \b device. One per batch. If nullptr or if \p T is real, it is ignored.
    ///                         Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     const shared_t<float22_t[]>& matrices, const shared_t<float2_t[]>& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     float22_t matrix, float2_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant 3D FFT to transform.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant transformed 3D FFT. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    ///                         The outermost dimension is the batch.
    /// \param[in] matrices     On the \b host or \b device. 3x3 inverse rightmost rotation/scaling matrix. One per batch.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] shifts       On the \b host or \b device. One per batch. If nullptr or if \p T is real, it is ignored.
    ///                         Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     const shared_t<float33_t[]>& matrices, const shared_t<float3_t[]>& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     float33_t matrix, float3_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream);
}

namespace noa::cuda::geometry::fft {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant 2D FFT to transform.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant transformed 2D FFT. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    /// \param[in] matrix       2x2 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    /// \todo ADD TESTS!
    template<Remap REMAP, typename T>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be HC2HC or HC2H.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        On the \b host or \b device. Non-redundant 3D FFT to transform.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant transformed 3D FFT. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements, of \p input and \p output.
    /// \param[in] matrix       3x3 inverse rightmost rotation/scaling matrix.
    ///                         If a scaling is encoded in the transformation, remember that for a scaling S in real
    ///                         space, a scaling of 1/S should be used in Fourier space.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param[in] shift        Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are set to 0.
    /// \param interp_mode      Interpolation/filtering mode. Cubic modes are currently not supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    ///
    /// \bug In this implementation, rotating non-redundant FFTs will not generate exactly the same results as if
    ///      redundant FFTs were used. This bug affects only a few elements at the Nyquist frequencies (the ones on
    ///      the central axes, e.g. x=0) on the input and weights the interpolated values towards zero.
    template<Remap REMAP, typename T>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, InterpMode interp_mode, bool normalize, Stream& stream);
}

// -- Textures -- //
namespace noa::cuda::geometry::fft {
    /// Rotates/scales a non-redundant 2D (batched) FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Filter method of \p texture.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrices         On the \b host or \b device. 2x2 inverse rightmost rotation/scaling matrix. One per batch.
    /// \param[in] shifts           On the \b host or \b device. One per batch. If nullptr or if \p T is real, it is ignored.
    /// \param cutoff               Maximum output frequency to consider, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                             Frequencies higher than this value are set to 0.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const float22_t* matrices, const float2_t* shifts, float cutoff, Stream& stream);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float22_t matrix, float2_t shift, float cutoff, Stream& stream);

    /// Rotates/scales a non-redundant 3D (batched) FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Filter method of \p texture.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrices         On the \b host or \b device. 3x3 inverse rightmost rotation/scaling matrix. One per batch.
    /// \param[in] shifts           On the \b host or \b device. One per batch. If nullptr or if \p T is real, it is ignored.
    /// \param cutoff               Maximum output frequency to consider, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                             Frequencies higher than this value are set to 0.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const float33_t* matrices, const float3_t* shifts, float cutoff, Stream& stream);

    /// Applies a single 3D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float33_t matrix, float3_t shift, float cutoff, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant 2D (batched) FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Filter method of \p texture.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrix           2x2 inverse rightmost rotation/scaling matrix. One per batch.
    /// \param[in] shift            Rightmost 2D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff               Maximum output frequency to consider, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                             Frequencies higher than this value are set to 0.
    /// \param normalize            Whether \p output should be normalized to have the same range as the input.
    ///                             If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float22_t matrix, const Symmetry& symmetry, float2_t shift,
                     float cutoff, bool normalize, Stream& stream);

    /// Rotates/scales and then symmetrizes a non-redundant 3D (batched) FFT.
    /// \tparam REMAP               Remap operation. Should be HC2HC or HC2H.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array. Should use unnormalized coordinates.
    /// \param texture_interp_mode  Filter method of \p texture.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrix           3x3 inverse rightmost rotation/scaling matrix. One per batch.
    /// \param[in] shift            Rightmost 3D real-space forward shift to apply (as phase shift) after the transformation.
    /// \param cutoff               Maximum output frequency to consider, in cycle/pix.
    ///                             Values are clamped from 0 (DC) to 0.5 (Nyquist).
    ///                             Frequencies higher than this value are set to 0.
    /// \param normalize            Whether \p output should be normalized to have the same range as the input.
    ///                             If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<Remap REMAP, typename T>
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float33_t matrix, const Symmetry& symmetry, float3_t shift,
                     float cutoff, bool normalize, Stream& stream);
}
