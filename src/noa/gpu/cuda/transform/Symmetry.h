/// \file noa/gpu/cuda/transform/Symmetry.h
/// \brief Symmetrizes arrays.
/// \author Thomas - ffyr2w
/// \date 05 Sep 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/common/transform/Symmetry.h"

// -- Using textures -- //
namespace noa::cuda::transform {
    using Symmetry = ::noa::transform::Symmetry;

    /// Symmetrizes the 2D texture.
    /// \tparam T                       float or cfloat_t.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param[out] output              On the \b device. Symmetrized output array.
    /// \param output_pitch             Pitch, in elements, of \p output.
    /// \param shape                    Physical {fast, medium} shape of \p texture and \p output, in elements.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices, usually retrieved from Symmetry.
    ///                                 The identity matrix is implicitly considered and should NOT be included here.
    ///                                 The matrices are converted for 2x2 matrices internally.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param symmetry_center          Index of the symmetry center.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size2_t shape,
                               const float33_t* symmetry_matrices, size_t symmetry_count, float2_t symmetry_center,
                               Stream& stream);

    /// Symmetrizes the 3D texture.
    /// \tparam T                       float or cfloat_t.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param output                   On the \b device. Symmetrized output array.
    /// \param output_pitch             Pitch, in elements, of \p output.
    /// \param shape                    Physical {fast, medium, slow} shape of \p texture and \p output, in elements.
    /// \param[in] symmetry_matrices    On the \b device. Symmetry matrices, usually retrieved from Symmetry.
    ///                                 The identity matrix is implicitly considered and should NOT be included here.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param symmetry_center          Index of the symmetry center.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size3_t shape,
                               const float33_t* symmetry_matrices, size_t symmetry_count, float3_t symmetry_center,
                               Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Symmetrizes the 2D input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter2D().
    /// \tparam T               float or cfloat_t.
    /// \param[in] inputs       Input array(s) to symmetrize. One per batch.
    ///                         If pre-filtering is required (see \p PREFILTER), should be on the \b device. Otherwise,
    ///                         can be on the \b host or \b device.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Symmetrized output array(s). One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Physical {fast, medium} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Index of the symmetry center. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    /// \note In-place computation is allowed, i.e. \p inputs and \p outputs can overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                               size2_t shape, size_t batches, const Symmetry& symmetry, float2_t symmetry_center,
                               InterpMode interp_mode, Stream& stream);

    /// Symmetrizes the 3D input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter3D().
    /// \tparam T               float or cfloat_t.
    /// \param[in] inputs       Input array(s) to symmetrize. One per batch.
    ///                         If pre-filtering is required (see \p PREFILTER), should be on the \b device. Otherwise,
    ///                         can be on the \b host or \b device.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Symmetrized output array(s). One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Index of the symmetry center. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    /// \note In-place computation is allowed, i.e. \p inputs and \p outputs can overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                               size3_t shape, size_t batches, const Symmetry& symmetry, float3_t symmetry_center,
                               InterpMode interp_mode, Stream& stream);
}
