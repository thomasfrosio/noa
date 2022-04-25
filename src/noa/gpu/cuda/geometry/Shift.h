/// \file noa/gpu/cuda/geometry/Shift.h
/// \brief Pixel shifts for 2D and 3D (batched) arrays.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// -- Using arrays -- //
namespace noa::cuda::geometry {
    /// Applies one or multiple 2D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    /// \details The input and output arrays should be 2D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 2D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input 2D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The innermost dimension should be contiguous.
    /// \param input_shape          Rightmost, in elements, shape of \p input.
    /// \param[out] output          On the \b device. Output 2D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] shifts           On the \b device. Rightmost forward shifts. One per batch.
    /// \param interp_mode          Filter method. All modes are supported.
    /// \param border_mode          Address mode. BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<float2_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 2D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies one or multiple 3D translations.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a additional translation in \p translations, one can move the center of the output window
    ///          relative to the input window, effectively combining a translation and an extraction.
    /// \details The input and output arrays should be 3D arrays. If the output is batched, a different matrix will
    ///          be applied for every batch. In this case, the input can be batched as well, resulting in a fully
    ///          batched operation. However if the input is not batched, it is broadcast to all output batches,
    ///          effectively applying multiple transformations to the same 3D input array.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered.
    ///                             Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T                   float or cfloat_t.
    /// \param[in] input            Input 3D array. If pre-filtering is required, should be on the \b device.
    ///                             Otherwise, can be on the \b host or \b device.
    /// \param input_stride         Rightmost stride, in elements, of \p input.
    ///                             The innermost dimension should be contiguous.
    /// \param input_shape          Rightmost, in elements, shape of \p input.
    /// \param[out] output          On the \b device. Output 3D array. Can be equal to \p input.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] shifts           On the \b device. Rightmost forward shifts. One per batch.
    /// \param interp_mode          Filter method. All modes are supported.
    /// \param border_mode          Address mode. BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    ///                             The last two are only supported with INTER_NEAREST and INTER_LINEAR_FAST.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when the function returns.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 const shared_t<float3_t[]>& shifts, InterpMode interp_mode, BorderMode border_mode, Stream& stream);

    /// Applies a single 3D translation.
    /// \see This function has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                 float3_t shift, InterpMode interp_mode, BorderMode border_mode, Stream& stream);
}

// -- Using textures -- //
namespace noa::cuda::geometry {
    /// Applies one or multiple 2D translations.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Rightmost shape, in elements, of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. All modes are supported.
    /// \param texture_border_mode  Address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] output          On the \b device. Output 2D array.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] shifts           On the \b host or \b device. Rightmost forward shifts. One per batch.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<typename T>
    void shift2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 const float2_t* shifts, Stream& stream);

    /// Translates a single 2D array.
    /// \see This function has the same features and limitations than the first overload above.
    template<typename T>
    void shift2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, Stream& stream);

    /// Applies one or multiple 3D translations.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_shape        Rightmost shape, in elements, of \p texture.
    ///                             This is only used if \p texture_border_mode is BORDER_PERIODIC or BORDER_MIRROR.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. All modes are supported.
    /// \param texture_border_mode  Address mode of \p texture.
    ///                             Should be BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC or BORDER_MIRROR.
    /// \param[out] output          On the \b device. Output arrays. One per translation.
    /// \param output_stride        Rightmost stride, in elements, of \p output.
    /// \param output_shape         Rightmost shape, in elements, of \p output. The outermost dimension is the batch.
    /// \param[in] shifts           On the \b host or \b device. Rightmost forward shifts. One per batch.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///
    /// \see "noa/common/geometry/Transform.h" for more details on the conventions used for transformations.
    /// \see "noa/gpu/cuda/memory/PtrTexture.h" for more details on CUDA textures and how to use them.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note BORDER_PERIODIC and BORDER_MIRROR are only supported with INTER_NEAREST and INTER_LINEAR_FAST, and
    ///       require \a texture to use normalized coordinates. All the other cases require unnormalized coordinates.
    template<typename T>
    void shift3D(cudaTextureObject_t texture, size3_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 const float3_t* shifts, Stream& stream);

    /// Translates a single 3D array.
    /// \see This function has the same features and limitations than the first overload above.
    template<typename T>
    void shift3D(cudaTextureObject_t texture, size3_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 float3_t shift, Stream& stream);
}
