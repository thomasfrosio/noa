#pragma once

#ifndef NOA_TRANSPOSE_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"

namespace noa::cpu::memory::details {
    template<typename T>
    void transpose(const T*, size4_t, size4_t, T*, size4_t, uint4_t);
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void transpose0213(T*, size4_t, size4_t);
    template<typename T>
    void transpose0132(T*, size4_t, size4_t);
    template<typename T>
    void transpose0321(T*, size4_t, size4_t);
}

namespace noa::cpu::memory {
    template<typename T>
    void transpose(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                   const shared_t<T[]>& output, size4_t output_stride, uint4_t permutation, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 3))
            NOA_THROW("Permutation {} is not valid", permutation);

        if (input == output) {
            const uint idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return stream.enqueue([=]() {
                        details::inplace::transpose0213<T>(output.get(), output_stride, input_shape);
                    });
                case 132:
                    return stream.enqueue([=]() {
                        details::inplace::transpose0132<T>(output.get(), output_stride, input_shape);
                    });
                case 321:
                    return stream.enqueue([=]() {
                        details::inplace::transpose0321<T>(output.get(), output_stride, input_shape);
                    });
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            if (all(permutation == uint4_t{0, 1, 2, 3})) {
                return copy(input, input_stride, output, output_stride, input_shape, stream);
            } else {
                return stream.enqueue([=]() {
                    details::transpose<T>(input.get(), input_stride, input_shape,
                                          output.get(), output_stride, permutation);
                });
            }
        }
    }
}
