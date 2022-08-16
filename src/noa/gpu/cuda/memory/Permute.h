#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace noa::cuda::memory::details {
    template<typename T>
    void permute0213(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream);
    template<typename T>
    void permute0132(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream);
    template<typename T>
    void permute0312(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream);
    template<typename T>
    void permute0231(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream);
    template<typename T>
    void permute0321(const shared_t<T[]>& input, size4_t input_strides,
                     const shared_t<T[]>& output, size4_t output_strides,
                     size4_t shape, Stream& stream);
}

namespace noa::cuda::memory::details::inplace {
    template<typename T>
    void permute0213(const shared_t<T[]>& output, size4_t output_strides, size4_t shape, Stream& stream);
    template<typename T>
    void permute0132(const shared_t<T[]>& output, size4_t output_strides, size4_t shape, Stream& stream);
    template<typename T>
    void permute0321(const shared_t<T[]>& output, size4_t output_strides, size4_t shape, Stream& stream);
}

namespace noa::cuda::memory {
    // Permutes, in memory, the axes of an array.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void permute(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, uint4_t permutation, Stream& stream) {
        if (any(permutation > 3) || noa::math::sum(permutation) != 6)
            NOA_THROW("Permutation {} is not valid", permutation);

        const uint idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
        if (input == output) {
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return details::inplace::permute0213(output, output_strides, input_shape, stream);
                case 132:
                    return details::inplace::permute0132(output, output_strides, input_shape, stream);
                case 321:
                    return details::inplace::permute0321(output, output_strides, input_shape, stream);
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            switch (idx) {
                case 123:
                    return copy(input, input_strides, output, output_strides, input_shape, stream);
                case 213:
                    return details::permute0213(input, input_strides, output, output_strides, input_shape, stream);
                case 132:
                    return details::permute0132(input, input_strides, output, output_strides, input_shape, stream);
                case 312:
                    return details::permute0312(input, input_strides, output, output_strides, input_shape, stream);
                case 231:
                    return details::permute0231(input, input_strides, output, output_strides, input_shape, stream);
                case 321:
                    return details::permute0321(input, input_strides, output, output_strides, input_shape, stream);
                default:
                    // Much slower...
                    const size4_t output_shape = indexing::reorder(input_shape, permutation);
                    const size4_t input_strides_permuted = indexing::reorder(input_strides, permutation);
                    copy(input, input_strides_permuted, output, output_strides, output_shape, stream);
            }
        }
    }
}
