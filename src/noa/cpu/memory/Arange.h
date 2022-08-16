#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    // Returns evenly spaced values within a given interval.
    template<typename T>
    inline void arange(T* src, size_t elements, T start = T(0), T step = T(1)) {
        T value = start;
        for (size_t i = 0; i < elements; ++i, value += step)
            src[i] = value;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T>
    inline void arange(T* src, size4_t strides, size4_t shape, T start = T(0), T step = T(1)) {
        if (indexing::areContiguous(strides, shape))
            return arange(src, shape.elements(), start, step);

        T value = start;
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l, value += step)
                        src[indexing::at(i, j, k, l, strides)] = value;
        }
    }

    // Returns evenly spaced values within a given interval.
    template<typename T>
    inline void arange(const shared_t<T[]>& src, size_t elements, T start, T step, Stream& stream) {
        stream.enqueue([=]() {
            arange(src.get(), elements, start, step);
        });
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T>
    inline void arange(const shared_t<T[]>& src, size4_t strides, size4_t shape, T start, T step, Stream& stream) {
        stream.enqueue([=]() {
            arange(src.get(), strides, shape, start, step);
        });
    }
}
