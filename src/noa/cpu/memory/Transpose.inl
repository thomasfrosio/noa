#pragma once

#ifndef NOA_TRANSPOSE_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/Copy.h"

namespace noa::cpu::memory::details {
    template<typename T>
    void transpose021(const T*, size3_t, T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose102(const T*, size3_t, T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose120(const T*, size3_t, T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose201(const T*, size3_t, T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose210(const T*, size3_t, T*, size3_t, size3_t, size_t);
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void transpose021(T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose102(T*, size3_t, size3_t, size_t);
    template<typename T>
    void transpose210(T*, size3_t, size3_t, size_t);
}

namespace noa::cpu::memory {
    constexpr size3_t transpose(size3_t shape, uint3_t permutation) {
        const uint idx = permutation.x * 100 + permutation.y * 10 + permutation.z;
        switch (idx) {
            case 12U:
                return shape;
            case 21U:
                return {shape.x, shape.z, shape.y};
            case 102U:
                return {shape.y, shape.x, shape.z};
            case 120U:
                return {shape.y, shape.z, shape.x};
            case 201U:
                return {shape.z, shape.x, shape.y};
            case 210U:
                return {shape.z, shape.y, shape.x};
            default:
                NOA_THROW("Permutation {} is not valid", permutation);
        }
    }

    template<typename T>
    void transpose(const T* inputs, size3_t input_pitch, size3_t shape,
                   T* outputs, size3_t output_pitch, uint3_t permutation,
                   size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if (any(permutation > 2U))
            NOA_THROW("Permutation {} is not valid", permutation);

        const uint idx = permutation.x * 100 + permutation.y * 10 + permutation.z;
        if (inputs == outputs) {
            switch (idx) {
                case 12U:
                    return;
                case 21U:
                    return stream.enqueue(details::inplace::transpose021<T>, outputs, output_pitch, shape, batches);
                case 102U:
                    return stream.enqueue(details::inplace::transpose102<T>, outputs, output_pitch, shape, batches);
                case 210U:
                    return stream.enqueue(details::inplace::transpose210<T>, outputs, output_pitch, shape, batches);
                case 120U:
                case 201U:
                    NOA_THROW("The in-place permutation {} is not yet supported", permutation);
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        } else {
            switch (idx) {
                case 12U:
                    return copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);
                case 21U:
                    return stream.enqueue(details::transpose021<T>, inputs, input_pitch,
                                          outputs, output_pitch, shape, batches);
                case 102U:
                    return stream.enqueue(details::transpose102<T>, inputs, input_pitch,
                                          outputs, output_pitch, shape, batches);
                case 120U:
                    return stream.enqueue(details::transpose120<T>, inputs, input_pitch,
                                          outputs, output_pitch, shape, batches);
                case 201U:
                    return stream.enqueue(details::transpose201<T>, inputs, input_pitch,
                                          outputs, output_pitch, shape, batches);
                case 210U:
                    return stream.enqueue(details::transpose210<T>, inputs, input_pitch,
                                          outputs, output_pitch, shape, batches);
                default:
                    NOA_THROW("Permutation {} is not valid", permutation);
            }
        }
    }
}
