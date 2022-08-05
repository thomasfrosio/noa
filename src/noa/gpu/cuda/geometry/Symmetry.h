/// \file noa/gpu/cuda/geometry/Symmetry.h
/// \brief Symmetrizes arrays.
/// \author Thomas - ffyr2w
/// \date 05 Sep 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/common/geometry/Symmetry.h"

namespace noa::cuda::geometry {
    using Symmetry = ::noa::geometry::Symmetry;

    /// Symmetrizes the 2D (batched) input array.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        Input 2D array. If pre-filtering is required, should be on the \b device.
    ///                         Otherwise, can be on the \b host or \b device.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    ///                         The width dimension should be contiguous.
    /// \param[out] output      On the \b device. Output 2D array. Can be equal to \p input.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param symmetry         Symmetry operator.
    /// \param center           HW center of the symmetry.
    /// \param interp_mode      Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize2D(const shared_t<T[]>& input, size4_t input_strides,
                      const shared_t<T[]>& output, size4_t output_strides,
                      size4_t shape, const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool normalize, Stream& stream);

    /// Symmetrizes the 3D (batched) input array.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered.
    ///                         Only used if \p interp_mode is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST.
    /// \tparam T               float, cfloat_t.
    /// \param[in] input        Input 3D array. If pre-filtering is required, should be on the \b device.
    ///                         Otherwise, can be on the \b host or \b device.
    /// \param input_strides    BDHW strides, in elements, of \p input.
    ///                         The width dimension should be contiguous.
    /// \param[out] output      On the \b device. Output 2D array. Can be equal to \p input.
    /// \param output_strides   BDHW strides, in elements, of \p output.
    /// \param shape            BDHW shape, in elements, of \p input and \p output.
    /// \param[in] symmetry     Symmetry operator.
    /// \param center           HW center of the symmetry.
    /// \param interp_mode      Interpolation/filter mode. All interpolation modes are supported.
    /// \param normalize        Whether \p output should be normalized to have the same range as \p input.
    ///                         If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize3D(const shared_t<T[]>& input, size4_t input_strides,
                      const shared_t<T[]>& output, size4_t output_strides,
                      size4_t shape, const Symmetry& symmetry, float3_t center,
                      InterpMode interp_mode, bool normalize, Stream& stream);
}

// -- Using textures -- //
namespace noa::cuda::geometry {
    /// Symmetrizes the 2D texture.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param[out] output          On the \b device. Symmetrized output array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape, in elements, of \p texture and \p output.
    /// \param[in] symmetry         Symmetry operator.
    /// \param center               HW center of the symmetry.
    /// \param normalize            Whether \p output should be normalized to have the same range as the input.
    ///                             If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size4_t output_strides, size4_t output_shape,
                      const Symmetry& symmetry, float2_t center, bool normalize, Stream& stream);

    /// Symmetrizes the 3D texture.
    /// \tparam T                   float or cfloat_t.
    /// \param texture              Input texture bound to a CUDA array.
    /// \param texture_interp_mode  Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param[out] output          On the \b device. Symmetrized output array.
    /// \param output_strides       BDHW strides, in elements, of \p output.
    /// \param output_shape         BDHW shape, in elements, of \p texture and \p output.
    /// \param[in] symmetry         Symmetry operator.
    /// \param center               HW center of the symmetry.
    /// \param normalize            Whether \p output should be normalized to have the same range as the input.
    ///                             If false, output values end up being scaled by the symmetry count.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, cfloat_t>>>
    void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size4_t output_strides, size4_t output_shape,
                      const Symmetry& symmetry, float3_t center, bool normalize, Stream& stream);
}
