/// \file noa/gpu/cuda/geometry/Transform.h
/// \brief Apply affine transforms to images and volumes.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/geometry/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::geometry {
    /// Applies one or multiple 2D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p matrices, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 2D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MAT                 float23_t or float33_t.
    /// \param[in] input            Input 2D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The innermost dimension should be contiguous.
    /// \param input_shape          Rightmost, in elements, shape of \p input.
    /// \param[out] output          On the \b device. Output 2D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrices         One the \b host or \b device. 2x3 or 3x3 inverse rightmost affine matrices. One per batch.
    /// \param interp_mode          Filter method. All modes are supported.
    /// \param border_mode          Address mode. BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     const shared_t<MAT[]>& matrices,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 2D affine (batched) transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies one or multiple 3D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p matrices, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 3D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MAT                 float34_t or float44_t.
    /// \param[in] input            Input 3D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The third-most and innermost dimensions should be contiguous.
    /// \param input_shape          Rightmost shape of \p input.
    /// \param[out] output          On the \b device. Output 3D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape of \p output. The outermost dimension is the batch dimension.
    /// \param[in] matrices         One the \b host or \b device. 3x4 or 4x4 inverse rightmost affine matrices. One per batch.
    /// \param interp_mode          Interpolation/filter method. All interpolation modes are supported.
    /// \param border_mode          Address mode. BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     const shared_t<MAT[]>& matrices,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies one 3D affine transform to a (batched) array.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode, Stream& stream);
}

// -- Symmetry -- //
namespace noa::cuda::geometry {
    using Symmetry = noa::geometry::Symmetry;
    /// Shifts, then rotates/scales and applies the symmetry on the 2D input array.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p matrices, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, the input can be batched
    ///          as well, resulting in a fully batched operation. However if the input is not batched, it is broadcast
    ///          to all output batches, effectively applying multiple transformations to the same 2D input array.
    ///
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        Input 2D array. If pre-filtering is required, should be on the \b device.
    ///                         Otherwise, can be on the \b host or \b device.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    ///                         The innermost dimension should be contiguous.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Transformed array. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    /// \param shift            Rightmost forward shift to apply before the other transformations.
    ///                         Positive shifts translate the object to the right.
    /// \param matrix           Rightmost inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param center           Rightmost index of the transformation center.
    ///                         Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode      Interpolation/filter method. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     InterpMode interp_mode, bool normalize, Stream& stream);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input array.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p matrices, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, the input can be batched
    ///          as well, resulting in a fully batched operation. However if the input is not batched, it is broadcast
    ///          to all output batches, effectively applying multiple transformations to the same 3D input array.
    ///
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        Input 3D array. If pre-filtering is required, should be on the \b device.
    ///                         Otherwise, can be on the \b host or \b device.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    ///                         The third-most and innermost dimensions should be contiguous.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Transformed array. Can be equal to \p input.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    /// \param shift            Rightmost forward shift to apply before the other transformations.
    ///                         Positive shifts translate the object to the right.
    /// \param matrix           Rightmost inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry     Symmetry operator to apply after the rotation/scaling.
    /// \param center           Rightmost index of the transformation center.
    ///                         Both \p matrix and \p symmetry operates around this center.
    /// \param interp_mode      Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool normalize, Stream& stream);
}

// -- Using textures -- //
namespace noa::cuda::geometry {
    /// Applies one or multiple 2D affine transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float23_t or float33_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Rightmost shape of \p texture.
    /// \param texture_interp_mode  Filter method of \p texture.
    /// \param texture_border_mode  Address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrices         One the \b host or \b device. 2x3 or 3x3 inverse rightmost affine matrices. One per batch.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const MAT* matrices, Stream& stream);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float23_v<MAT> || traits::is_float33_v<MAT>>>
    void transform2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, Stream& stream);

    /// Applies one or multiple 3D affine transforms.
    /// \tparam T                   float or cfloat_t.
    /// \tparam MATRIX              float34_t or float44_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Rightmost shape of \p texture.
    /// \param texture_interp_mode  Filter method of \p texture. Any of InterpMode.
    /// \param texture_border_mode  Address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] output          On the \b device. Output array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] matrix           One the \b host or \b device. 3x4 or 4x4 inverse rightmost affine matrices. One per batch.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const MAT* matrices, Stream& stream);

    /// Applies a single 3D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<typename T, typename MAT,
             typename = std::enable_if_t<traits::is_float34_v<MAT> || traits::is_float44_v<MAT>>>
    void transform3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, Stream& stream);
}

// -- Symmetry - using textures -- //
namespace noa::cuda::geometry {
    /// Shifts, then rotates/scales and applies the symmetry on the 2D input texture.
    /// \tparam T                       float or cfloat.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/addressing mode of the texture. Any of InterpMode.
    /// \param[out] output              On the \b device. Symmetrized array.
    /// \param output_stride            Rightmost stride, in elements, of \p output.
    /// \param output_shape             Rightmost shape, in elements, of \p output.
    /// \param shift                    Shift to apply before the other transformations.
    /// \param matrix                   Inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry             Symmetry operator to apply after the rotation/scaling.
    /// \param center                   Index of the transformation center.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    template<typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float2_t shift, float22_t matrix, const Symmetry& symmetry, float2_t center,
                     bool normalize, Stream& stream);

    /// Shifts, then rotates/scales and applies the symmetry on the 3D input texture.
    /// \tparam T                       float or cfloat.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/addressing mode of the texture. Any of InterpMode.
    /// \param[out] output              On the \b device. Symmetrized array.
    /// \param output_stride            Rightmost stride, in elements, of \p output.
    /// \param output_shape             Rightmost shape, in elements, of \p output.
    /// \param shift                    Shift to apply before the other transformations.
    /// \param matrix                   Inverse rotation/scaling to apply after the shift.
    /// \param[in] symmetry             Symmetry operator to apply after the rotation/scaling.
    /// \param center                   Index of the transformation center.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       As such, one must make sure the texture and the CUDA array it is bound to stays valid until completion.
    template<typename T>
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize, Stream& stream);
}
