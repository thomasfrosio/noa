/// \file noa/cpu/transform/Apply.h
/// \brief Apply linear and affine transforms to arrays.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/transform/Symmetry.h"

// -- Using arrays -- //
namespace noa::cpu::transform {
    /// Applies one or multiple 2D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter2D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam MATRIX              float23_t or float33_t.
    /// \param[in] input            On the \p host. Input 2D array.
    /// \param input_shape          Logical {fast, medium} shape of \p input.
    /// \param[out] outputs         On the \b host. Output 2D arrays. One per transformation. Shouldn't be equal to \p input.
    /// \param output_shape         Logical {fast, medium} shape of \p outputs.
    /// \param[in] transforms       One the \b host. 2x3 or 3x3 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MATRIX>
    NOA_HOST void apply2D(const T* input, size2_t input_shape, T* outputs, size2_t output_shape,
                          const MATRIX* transforms, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, T value);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MATRIX>
    NOA_IH void apply2D(const T* input, size2_t input_shape,
                        T* output, size2_t output_shape,
                        MATRIX transform, InterpMode interp_mode, BorderMode border_mode, T value) {
        apply2D<PREFILTER>(input, input_shape, output, output_shape, &transform, 1, interp_mode, border_mode, value);
    }

    /// Applies one or multiple 3D affine transforms.
    /// \details This function allows to specify an output window that doesn't necessarily have the same shape
    ///          than the input window. The output window starts at the same index than the input window, so by
    ///          entering a translation in \p transforms, one can move the center of the output window relative
    ///          to the input window, effectively combining a transformation and an extraction.
    ///
    /// \tparam PREFILTER           Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                             is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                             shape as \p input is allocated and used to store the output of bspline::prefilter3D(),
    ///                             which is then used as input for the interpolation.
    /// \tparam T                   float, double, cfloat_t or cdouble_t.
    /// \tparam MATRIX              float34_t or float44_t.
    /// \param[in] input            On the \p host. Input 2D array.
    /// \param input_shape          Logical {fast, medium, slow} shape of \p input.
    /// \param[out] outputs         On the \b host. Output 2D arrays. One per transformation. Shouldn't be equal to \p input.
    /// \param output_shape         Logical {fast, medium, slow} shape of \p outputs.
    /// \param[in] transforms       One the \b host. 3x4 or 4x4 inverse affine matrices. One per transformation.
    ///                             For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                             on the output array coordinates. This functions assumes \p transforms are already
    ///                             inverted and pre-multiplies the coordinates with these matrices directly.
    /// \param nb_transforms        Number of transforms to compute.
    /// \param interp_mode          Interpolation/filter method. All "accurate" interpolation modes are supported.
    /// \param border_mode          Border/address mode. All border modes are supported, except BORDER_NOTHING.
    /// \param value                Constant value to use for out-of-bounds coordinates.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \see "noa/common/transform/Geometry.h" for more details on the conventions used for transformations.
    template<bool PREFILTER = true, typename T, typename MATRIX>
    NOA_HOST void apply3D(const T* input, size3_t input_shape, T* outputs, size3_t output_shape,
                          const MATRIX* transforms, uint nb_transforms,
                          InterpMode interp_mode, BorderMode border_mode, T value);

    /// Applies a single 2D affine transform.
    /// \see This function is has the same features and limitations than the overload above.
    template<bool PREFILTER = true, typename T, typename MATRIX>
    NOA_IH void apply3D(const T* input, size3_t input_shape,
                        T* output, size3_t output_shape,
                        MATRIX transform, InterpMode interp_mode, BorderMode border_mode, T value) {
        apply3D<PREFILTER>(input, input_shape, output, output_shape, &transform, 1, interp_mode, border_mode, value);
    }
}

// -- Apply symmetry -- //
namespace noa::cpu::transform {
    using Symmetry = ::noa::transform::Symmetry;

    /// Shifts, rotate/scale and then apply the symmetry on the 2D input array.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                         shape as \p input is allocated and used to store the output of bspline::prefilter2D(),
    ///                         which is then used as input for the interpolation.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param input            On the \b host. Input array to transform.
    /// \param output           On the \b host. Transformed output arrays.
    /// \param shape            Physical {fast, medium} shape of \p input and \p output, in elements.
    /// \param center           Transformation center. Both \p matrix and \p symmetry operates around this center.
    /// \param shifts           Shifts to apply.
    /// \param matrix           Rotation/scaling to apply after the shifts.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This functions assumes \p matrix is already
    ///                         inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param symmetry         Symmetry operator to apply after the rotation/scaling.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If there's no symmetry, equivalent results can be accomplished with the more flexible affine transforms.
    ///       Similarly, if the order of the transformations is not the desired one, a solution is to first transform
    ///       the array using the various transformation functions and then apply the symmetry on the transformed
    ///       array using the symmetrize functions in "noa/cpu/transform/Symmetry.h".
    template<bool PREFILTER = true, typename T>
    NOA_HOST void apply2D(const T* input, T* output, size2_t shape,
                          float2_t center, float2_t shifts, float22_t matrix, Symmetry symmetry,
                          InterpMode interp_mode);

    /// Shifts, rotate/scale and then apply the symmetry on the 3D input array.
    /// \tparam PREFILTER       Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                         shape as \p input is allocated and used to store the output of bspline::prefilter3D(),
    ///                         which is then used as input for the interpolation.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param input            On the \b host. Input array to transform.
    /// \param output           On the \b host. Transformed output arrays.
    /// \param shape            Physical {fast, medium, slow} shape of \p input and \p output, in elements.
    /// \param center           Transformation center. Both \p matrix and \p symmetry operates around this center.
    /// \param shifts           Shifts to apply.
    /// \param matrix           Rotation/scaling to apply after the shifts.
    ///                         For a final transformation `A` in the output array, we need to apply `inverse(A)`
    ///                         on the output array coordinates. This functions assumes \p matrix is already
    ///                         inverted and pre-multiplies the coordinates with the matrix directly.
    /// \param symmetry         Symmetry operator to apply after the rotation/scaling.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    ///
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    /// \note If there's no symmetry, equivalent results can be accomplished with the more flexible affine transforms.
    ///       Similarly, if the order of the transformations is not the desired one, a solution is to first transform
    ///       the array using the various transformation functions and then apply the symmetry on the transformed
    ///       array using the symmetrize functions in "noa/cpu/transform/Symmetry.h".
    template<bool PREFILTER = true, typename T>
    NOA_HOST void apply3D(const T* input, T* output, size3_t shape,
                          float3_t center, float3_t shifts, float33_t matrix, Symmetry symmetry,
                          InterpMode interp_mode);
}
