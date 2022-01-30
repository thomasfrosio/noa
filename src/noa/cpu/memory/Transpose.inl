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
    constexpr size4_t transpose(size4_t shape, uint4_t permutation) {
        return {shape[permutation[0]], shape[permutation[1]], shape[permutation[2]], shape[permutation[3]]};
    }

    template<typename T>
    void transpose(const T* inputs, size4_t input_stride, size4_t input_shape,
                   T* outputs, size4_t output_stride, uint4_t permutation, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 3))
            NOA_THROW("Permutation {} is not valid", permutation);

        if (inputs == outputs) {
            const uint idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return stream.enqueue(details::inplace::transpose0213<T>, outputs, output_stride, input_shape);
                case 132:
                    return stream.enqueue(details::inplace::transpose0132<T>, outputs, output_stride, input_shape);
                case 321:
                    return stream.enqueue(details::inplace::transpose0321<T>, outputs, output_stride, input_shape);
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            if (all(permutation == uint4_t{0, 1, 2, 3}))
                return copy(inputs, input_stride, outputs, output_stride, input_shape, stream);
            else
                return stream.enqueue(details::transpose<T>, inputs, input_stride, input_shape,
                                      outputs, output_stride, permutation);
        }
    }
}
