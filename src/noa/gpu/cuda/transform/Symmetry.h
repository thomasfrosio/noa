/// \file noa/gpu/cuda/transform/Symmetry.h
/// \brief Symmetrizes arrays.
/// \author Thomas - ffyr2w
/// \date 05 Sep 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/transform/Symmetry.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

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
    /// \param[in] symmetry_matrices    On the \b device. Matrices from the get() method of Symmetry. The identity
    ///                                 matrix is implicitly considered and should not be included here. They are converted to
    ///                                 2x2 matrices, so really they should describe a C or D symmetry.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param symmetry_center          Center of the symmetry.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size2_t shape,
                               const float33_t* symmetry_matrices, uint symmetry_count, float2_t symmetry_center,
                               Stream& stream);

    /// Symmetrizes the 3D texture.
    /// \tparam T                       float or cfloat_t.
    /// \param texture                  Input texture bound to a CUDA array.
    /// \param texture_interp_mode      Interpolation/filter method of \p texture. Any of InterpMode.
    /// \param output                   On the \b device. Symmetrized output array.
    /// \param output_pitch             Pitch, in elements, of \p output.
    /// \param shape                    Physical {fast, medium, slow} shape of \p texture and \p output, in elements.
    /// \param[in] symmetry_matrices    On the \b device. Matrices from the get() method of Symmetry. The identity
    ///                                 matrix is implicitly considered and should not be included here.
    /// \param symmetry_count           Number of matrices. If 0, \p symmetry_matrices is not read.
    /// \param symmetry_center          Center of the symmetry.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    /// \note The \p texture is expected to be set with BORDER_ZERO and unnormalized coordinates.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                               T* output, size_t output_pitch, size3_t shape,
                               const float33_t* symmetry_matrices, uint symmetry_count, float3_t symmetry_center,
                               Stream& stream);
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    /// Symmetrizes the 2D input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter2D().
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host or device. Input array(s) to symmetrize. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] outputs     On the \b device. Symmetrized output array(s). One per batch.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Physical {fast, medium} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator. Should be a C or D symmetry.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    /// \note In-place computation is allowed, i.e. \p inputs and \p outputs can overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                               size2_t shape, uint batches, Symmetry symmetry, float2_t symmetry_center,
                               InterpMode interp_mode, Stream& stream);

    /// Symmetrizes the 2D input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                             size2_t shape, uint batches, const char* symmetry, float2_t symmetry_center,
                             InterpMode interp_mode, Stream& stream) {
        Symmetry s(symmetry);
        symmetrize2D<PREFILTER>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                std::move(s), symmetry_center, interp_mode, stream);
    }

    /// Symmetrizes the 3D input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter3D().
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host or device. Input array(s) to symmetrize. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] outputs     On the \b device. Symmetrized output array(s). One per batch.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    /// \note In-place computation is allowed, i.e. \p inputs and \p outputs can overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                               size3_t shape, uint batches, Symmetry symmetry, float3_t symmetry_center,
                               InterpMode interp_mode, Stream& stream);

    /// Symmetrizes the 3D input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                             size3_t shape, uint batches, const char* symmetry, float3_t symmetry_center,
                             InterpMode interp_mode, Stream& stream) {
        Symmetry s(symmetry);
        symmetrize3D<PREFILTER>(inputs, input_pitch, outputs, output_pitch, shape, batches,
                                std::move(s), symmetry_center, interp_mode, stream);
    }

    /// Symmetrizes the input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE or INTERP_CUBIC_BSPLINE_FAST. In this case and if true, the
    ///                         input(s) are pre-filtered using bspline::prefilter(2|3)D().
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host or device. Input array(s) to symmetrize. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] outputs     On the \b device. Symmetrized output array(s). One per batch.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. Any of InterpMode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when this function returns.
    /// \note In-place computation is allowed, i.e. \p inputs and \p outputs can overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                           size3_t shape, uint batches, Symmetry symmetry, float3_t symmetry_center,
                           InterpMode interp_mode, Stream& stream) {
        auto ndim = getNDim(shape);
        if (ndim == 2)
            symmetrize2D<PREFILTER>(inputs, input_pitch, outputs, output_pitch,
                                    size2_t(shape.x, shape.y), batches,
                                    symmetry, float2_t(symmetry_center.x, symmetry_center.y), interp_mode, stream);
        else if (ndim == 3)
            symmetrize3D<PREFILTER>(inputs, input_pitch, outputs, output_pitch,
                                    shape, batches, symmetry, symmetry_center, interp_mode, stream);
        else
            NOA_THROW("Number of dimensions ({}) not supported", ndim);
    }

    /// Symmetrizes the input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                           size3_t shape, uint batches, const char* symmetry, float3_t symmetry_center,
                           InterpMode interp_mode, Stream& stream) {
        Symmetry s(symmetry);
        symmetrize<PREFILTER>(inputs, input_pitch, outputs, output_pitch,
                              shape, batches, std::move(s), symmetry_center, interp_mode, stream);
    }
}
