/// \file noa/cpu/transform/Symmetry.h
/// \brief Symmetrizes arrays.
/// \author Thomas - ffyr2w
/// \date 04 Sep 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Exception.h"
#include "noa/common/transform/Symmetry.h"

namespace noa::cpu::transform {
    using Symmetry = ::noa::transform::Symmetry;

    /// Symmetrizes the 2D input array(s).
    /// \tparam PREFILTER       Whether or not the input(s) should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE. In this case and if true, the input(s) are pre-filtered using
    ///                         bspline::prefilter2D().
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host. Input array(s) to symmetrize. One per batch.
    /// \param[out] outputs     On the \b host. Symmetrized output array(s). One per batch.
    /// \param shape            Physical {fast, medium} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator. Should be a C or D symmetry.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize2D(const T* inputs, T* outputs, size2_t shape, uint batches,
                               Symmetry symmetry, float2_t symmetry_center, InterpMode interp_mode);

    /// Symmetrizes the 2D input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize2D(const T* inputs, T* outputs, size2_t shape, uint batches,
                             const char* symmetry, float2_t symmetry_center, InterpMode interp_mode) {
        Symmetry s(symmetry);
        symmetrize2D<PREFILTER>(inputs, outputs, shape, batches, std::move(s), symmetry_center, interp_mode);
    }

    /// Symmetrizes the 3D input array(s).
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host. Input array(s) to symmetrize. One per batch.
    /// \param[out] outputs     On the \b host. Symmetrized output array(s). One per batch.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize3D(const T* inputs, T* outputs, size3_t shape, uint batches,
                               Symmetry symmetry, float3_t symmetry_center, InterpMode interp_mode);

    /// Symmetrizes the 3D input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize3D(const T* inputs, T* outputs, size3_t shape, uint batches,
                             const char* symmetry, float3_t symmetry_center, InterpMode interp_mode) {
        Symmetry s(symmetry);
        symmetrize3D<PREFILTER>(inputs, outputs, shape, batches, std::move(s), symmetry_center, interp_mode);
    }

    /// Symmetrizes the input array(s).
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host. Input array(s) to symmetrize. One per batch.
    /// \param[out] outputs     On the \b host. Symmetrized output array(s). One per batch.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param symmetry_center  Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize(const T* inputs, T* outputs, size3_t shape, uint batches,
                           Symmetry symmetry, float3_t symmetry_center, InterpMode interp_mode) {
        auto ndim = getNDim(shape);
        if (ndim == 2)
            symmetrize2D<PREFILTER>(inputs, outputs, size2_t(shape.x, shape.y), batches,
                                    symmetry, float2_t(symmetry_center.x, symmetry_center.y), interp_mode);
        else if (ndim == 3)
            symmetrize3D<PREFILTER>(inputs, outputs, shape, batches, symmetry, symmetry_center, interp_mode);
        else
            NOA_THROW("Number of dimensions ({}) not supported", ndim);
    }

    /// Symmetrizes the input array(s).
    /// Identical to the overload above, except the symmetry operator is entered as a string.
    template<bool PREFILTER = true, typename T>
    NOA_IH void symmetrize(const T* inputs, T* outputs, size3_t shape, uint batches,
                           const char* symmetry, float3_t symmetry_center, InterpMode interp_mode) {
        Symmetry s(symmetry);
        symmetrize<PREFILTER>(inputs, outputs, shape, batches, std::move(s), symmetry_center, interp_mode);
    }
}
