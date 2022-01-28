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
    void transpose213(T*, size4_t, size4_t);
    template<typename T>
    void transpose132(T*, size4_t, size4_t);
    template<typename T>
    void transpose321(T*, size4_t, size4_t);
}

namespace noa::cpu::memory {
    constexpr size4_t transpose(size4_t shape, uint4_t permutation) {
        return {shape[permutation[0]], shape[permutation[1]], shape[permutation[2]], shape[permutation[3]]};
    }

    template<typename T>
    void transpose(const T* inputs, size4_t input_stride, size4_t input_shape,
                   T* outputs, size4_t output_stride, uint3_t permutation, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 3))
            NOA_THROW("Permutation {} is not valid", permutation);
        else if (permutation[0] != 0)
            NOA_THROW("Permuting the batch (i.e. outermost) dimension is not supported");

        const uint idx = permutation[1] * 100 + permutation[2] * 10 + permutation[3];
        if (inputs == outputs) {
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return stream.enqueue(details::inplace::transpose213<T>, outputs, output_stride, input_shape);
                case 132:
                    return stream.enqueue(details::inplace::transpose132<T>, outputs, output_stride, input_shape);
                case 321:
                    return stream.enqueue(details::inplace::transpose321<T>, outputs, output_stride, input_shape);
                case 312:
                case 231:
                    NOA_THROW("The in-place permutation {} is not yet supported", permutation);
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        } else {
            if (idx == 123)
                return copy(inputs, input_stride, outputs, output_stride, input_shape, stream);
            else
                return stream.enqueue(details::transpose<T>, inputs, input_stride, input_shape,
                                      outputs, output_stride, permutation);
        }
    }
}
