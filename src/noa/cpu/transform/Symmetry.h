/// \file noa/cpu/transform/Symmetry.h
/// \brief Symmetry transformations for images and volumes.
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
    /// \tparam PREFILTER       Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                         shape as \p inputs (one batch only) is allocated and used to store the prefiltered
    ///                         output which is then used as input for the interpolation.
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host. Input array(s) to symmetrize. One per batch.
    /// \param[out] outputs     On the \b host. Symmetrized output array(s). One per batch.
    /// \param shape            Physical {fast, medium} shape of \p inputs and \p outputs, in elements, for one batch.
    /// \param batches          Number of contiguous batches to process.
    /// \param symmetry         Symmetry operator.
    /// \param center           Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    ///
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize2D(const T* inputs, T* outputs, size2_t shape, uint batches,
                               const Symmetry& symmetry, float2_t center, InterpMode interp_mode);

    /// Symmetrizes the 3D input array(s).
    /// \tparam PREFILTER       Whether or not the input should be prefiltered. This is only used if \p interp_mode
    ///                         is INTERP_CUBIC_BSPLINE. In this case and if true, a temporary array of the same
    ///                         shape as \p inputs (one batch only) is allocated and used to store the prefiltered
    ///                         output which is then used as input for the interpolation.
    /// \tparam T               float, double, cfloat, cdouble_t.
    /// \param[in] inputs       On the \b host. Input array(s) to symmetrize. One per batch.
    /// \param[out] outputs     On the \b host. Symmetrized output array(s). One per batch.
    /// \param shape            Physical {fast, medium, slow} shape of \p inputs and \p outputs, in elements.
    /// \param batches          Number of contiguous batches to process.
    /// \param[in] symmetry     Symmetry operator.
    /// \param center           Center of the symmetry. The same center is used for every batch.
    /// \param interp_mode      Interpolation/filter mode. All "accurate" interpolation modes are supported.
    ///
    /// \note In-place computation is not allowed, i.e. \p inputs and \p outputs should not overlap.
    /// \note During transformation, out-of-bound elements are set to 0, i.e. BORDER_ZERO is used.
    template<bool PREFILTER = true, typename T>
    NOA_HOST void symmetrize3D(const T* inputs, T* outputs, size3_t shape, uint batches,
                               const Symmetry& symmetry, float3_t center, InterpMode interp_mode);
}
