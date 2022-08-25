#pragma once

#include "noa/Array.h"

namespace noa::memory {
    /// Permutes, in memory, the axes of an array.
    /// \tparam T           Any data type.
    /// \param[in] input    Array to permute.
    /// \param[out] output  Permuted array. Its shape and strides should be permuted already.
    /// \param permutation  Permutation. Axes are numbered from 0 to 3.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    /// \note On the GPU, the following permutations are fast: 0123, 0132, 0312, 0321, 0213, 0231.
    ///       Anything else calls copy(), which is much slower.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void permute(const Array<T>& input, const Array<T>& output, uint4_t permutation);
}

#define NOA_UNIFIED_TRANSPOSE_
#include "noa/memory/details/Permute.inl"
#undef NOA_UNIFIED_TRANSPOSE_
