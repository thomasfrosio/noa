/// \file noa/gpu/cuda/memory/Transpose.h
/// \brief Permutes the axes of an array.
/// \author Thomas - ffyr2w
/// \date 29 Jun 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace noa::cuda::memory::details {
    template<typename T>
    void transpose0213(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream);
    template<typename T>
    void transpose0132(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream);
    template<typename T>
    void transpose0312(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream);
    template<typename T>
    void transpose0231(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream);
    template<typename T>
    void transpose0321(const shared_t<T[]>& input, size4_t input_stride,
                       const shared_t<T[]>& output, size4_t output_stride,
                       size4_t shape, Stream& stream);
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void transpose0213(const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream);
    template<typename T>
    void transpose0132(const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream);
    template<typename T>
    void transpose0321(const shared_t<T[]>& output, size4_t output_stride, size4_t shape, Stream& stream);
}

namespace noa::cuda::memory {
    /// Transposes, in memory, the axes of an array.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b device. Input array to permute.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param input_shape      Rightmost shape of \p input.
    /// \param[out] output      On the \b device. Output permuted array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param permutation      Rightmost permutation. Axes are numbered from 0 to 3, 3 being the innermost dimension.
    ///                         6 permutations are supported: 0123, 0132, 0312, 0321, 0213, 0231.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \note For in-place permutations, only 0123, 0213, 0132, and 0321 are supported. Anything else throws an error.
    /// \note The in-place 0213 permutation requires the axis 1 and 2 to have the same size.
    ///       The in-place 0132 permutation requires the axis 3 and 2 to have the same size.
    ///       The in-place 0321 permutation requires the axis 3 and 1 to have the same size.
    template<typename T>
    void transpose(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                   const shared_t<T[]>& output, size4_t output_stride, uint4_t permutation, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 3))
            NOA_THROW("Permutation {} is not valid", permutation);

        const uint idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
        if (input == output) {
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return details::inplace::transpose0213(output, output_stride, input_shape, stream);
                case 132:
                    return details::inplace::transpose0132(output, output_stride, input_shape, stream);
                case 321:
                    return details::inplace::transpose0321(output, output_stride, input_shape, stream);
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            switch (idx) {
                case 123:
                    return copy(input, input_stride, output, output_stride, input_shape, stream);
                case 213:
                    return details::transpose0213(input, input_stride, output, output_stride, input_shape, stream);
                case 132:
                    return details::transpose0132(input, input_stride, output, output_stride, input_shape, stream);
                case 312:
                    return details::transpose0312(input, input_stride, output, output_stride, input_shape, stream);
                case 231:
                    return details::transpose0231(input, input_stride, output, output_stride, input_shape, stream);
                case 321:
                    return details::transpose0321(input, input_stride, output, output_stride, input_shape, stream);
                default:
                    NOA_THROW("Permutation {} is not supported", permutation);
            }
        }
    }
}
