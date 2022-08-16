#pragma once

#ifndef NOA_TRANSPOSE_INL_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/common/Exception.h"

namespace noa::cpu::memory::details {
    template<typename T>
    void permute(const T*, size4_t, size4_t, T*, size4_t, uint4_t);
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void permute0213(T*, size4_t, size4_t);
    template<typename T>
    void permute0132(T*, size4_t, size4_t);
    template<typename T>
    void permute0321(T*, size4_t, size4_t);
}

namespace noa::cpu::memory {
    template<typename T, typename>
    void permute(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, uint4_t permutation, Stream& stream) {
        if (any(permutation > 3) || noa::math::sum(permutation) != 6)
            NOA_THROW("Permutation {} is not valid", permutation);

        if (input == output) {
            const uint idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return stream.enqueue([=]() {
                        details::inplace::permute0213<T>(output.get(), output_strides, input_shape);
                    });
                case 132:
                    return stream.enqueue([=]() {
                        details::inplace::permute0132<T>(output.get(), output_strides, input_shape);
                    });
                case 321:
                    return stream.enqueue([=]() {
                        details::inplace::permute0321<T>(output.get(), output_strides, input_shape);
                    });
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            return stream.enqueue([=]() {
                details::permute<T>(input.get(), input_strides, input_shape,
                                    output.get(), output_strides, permutation);
            });
        }
    }
}
