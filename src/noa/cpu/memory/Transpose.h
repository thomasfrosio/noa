/// \file noa/cpu/memory/Transpose.h
/// \brief Reverse or permute the axes of an array.
/// \author Thomas - ffyr2w
/// \date 29 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Returns the transposed shape.
    constexpr NOA_IH size3_t transpose(size3_t shape, uint3_t permutation);

    /// Reverses or permutes the axes of an array.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, or any (complex) floating-point.
    /// \param[in] inputs       On the \b host. Input arrays to permute. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs, ignoring the batches.
    /// \param[out] outputs     On the \b host. Output permuted arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param permutation      Specifies the particular transposition to be performed. Values should be 0, 1 and 2,
    ///                         which represent the fast, medium and slow axes as entered in \a shape.
    ///                         For 3D arrays, all 6 permutations are supported: 012, 021, 102, 120, 201, 210.
    ///                         For 2D arrays, only 012 and 102 are supported.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For in-place permutations, only 012, 021, 102 and 210 are supported. Anything else throws an error.
    /// \note The in-place 021 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 102 permutation requires the axis 0 and 1 to have the same size.
    ///       The in-place 210 permutation requires the axis 0 and 2 to have the same size.
    template<typename T>
    NOA_HOST void transpose(const T* inputs, size3_t input_pitch, size3_t shape,
                            T* outputs, size3_t output_pitch, uint3_t permutation,
                            size_t batches, Stream& stream);
}

#define NOA_TRANSPOSE_INL_
#include "noa/cpu/memory/Transpose.inl"
#undef NOA_TRANSPOSE_INL_
