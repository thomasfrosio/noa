#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace noa::cuda::memory::details {
    template<typename T>
    void permute_0213(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0132(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0312(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0231(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0321(const T* input, const Strides4<i64>& input_strides,
                      T* output, const Strides4<i64>& output_strides,
                      const Shape4<i64>& shape, Stream& stream);

    template<typename T>
    void permute_0213_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0132_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream);
    template<typename T>
    void permute_0321_inplace(
            T* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& shape, Stream& stream);
}

namespace noa::cuda::memory {
    // Permutes, in memory, the axes of an array.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_numeric_v<T>>>
    void permute(const T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                 T* output, const Strides4<i64>& output_strides,
                 const Vec4<i64>& permutation, Stream& stream) {
        if (noa::any(permutation > 3) || noa::math::sum(permutation) != 6)
            NOA_THROW("Permutation {} is not valid", permutation);

        NOA_ASSERT(all(input_shape > 0));
        const auto idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
        if (input == output) {
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return details::permute_0213_inplace(output, output_strides, input_shape, stream);
                case 132:
                    return details::permute_0132_inplace(output, output_strides, input_shape, stream);
                case 321:
                    return details::permute_0321_inplace(output, output_strides, input_shape, stream);
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            switch (idx) {
                case 123:
                    return copy(input, input_strides, output, output_strides, input_shape, stream);
                case 213:
                    return details::permute_0213(input, input_strides, output, output_strides, input_shape, stream);
                case 132:
                    return details::permute_0132(input, input_strides, output, output_strides, input_shape, stream);
                case 312:
                    return details::permute_0312(input, input_strides, output, output_strides, input_shape, stream);
                case 231:
                    return details::permute_0231(input, input_strides, output, output_strides, input_shape, stream);
                case 321:
                    return details::permute_0321(input, input_strides, output, output_strides, input_shape, stream);
                default:
                    // Much slower...
                    const auto output_shape = noa::indexing::reorder(input_shape, permutation);
                    const auto input_strides_permuted = noa::indexing::reorder(input_strides, permutation);
                    copy(input, input_strides_permuted, output, output_strides, output_shape, stream);
            }
        }
    }
}
