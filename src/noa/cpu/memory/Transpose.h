/// \file noa/cpu/memory/Transpose.h
/// \brief Reverse or permute the axes of an array.
/// \author Thomas - ffyr2w
/// \date 29 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Transposes, in memory, the axes of an array.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b host. Input array to permute.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Output permuted array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param permutation      Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    ///
    /// \example Transpose the two innermost dimensions
    /// \code
    /// const size4_t shape{2,63,64,65};
    /// PtrHost<T> input(shape.elements()), output(shape.elements());
    /// // initialize the input...
    /// const uint4_t permutation{0,1,3,2};
    /// const size4_t transposed_shape = indexing::reorder(shape, permutation); // {2,63,65,64}
    /// transpose(input, shape.stride(), shape, output, transposed_shape.stride(), permutation, stream);
    /// \endcode
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void transpose(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                   const shared_t<T[]>& output, size4_t output_stride, uint4_t permutation, Stream& stream);
}

#define NOA_TRANSPOSE_INL_
#include "noa/cpu/memory/Transpose.inl"
#undef NOA_TRANSPOSE_INL_
