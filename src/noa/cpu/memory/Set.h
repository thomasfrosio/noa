#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    // Sets an array to a given value.
    template<typename T>
    inline void set(T* first, T* last, T value) {
        // the cast is not necessary for basic types, but for Complex<>, IntX<> or FloatX<>, it could help...
        if constexpr (traits::is_complex_v<T> || traits::is_intX_v<T> ||
                      traits::is_floatX_v<T> || traits::is_floatXX_v<T>) {
            using value_t = traits::value_type_t<T>;
            if (value == T{0})
                return std::fill(reinterpret_cast<value_t*>(first), reinterpret_cast<value_t*>(last), value_t{0});
        }
        // std::fill is calling memset, https://godbolt.org/z/1zEzTnoTK
        return std::fill(first, last, value);
    }

    // Sets an array to a given value.
    template<typename T>
    inline void set(T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }

    // Sets an array to a given value.
    template<typename T>
    inline void set(const shared_t<T[]>& src, size_t elements, T value, Stream& stream) {
        stream.enqueue([=]() { return set(src.get(), elements, value); });
    }

    // Sets an array to a given value.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void set(T* src, size4_t strides, size4_t shape, T value) {
        if constexpr (SWAP_LAYOUT) {
            const size4_t order = indexing::order(strides, shape);
            shape = indexing::reorder(shape, order);
            strides = indexing::reorder(strides, order);
        }
        if (indexing::areContiguous(strides, shape))
            return set(src, shape.elements(), value);

        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        src[indexing::at(i, j, k, l, strides)] = value;
    }

    // Sets an array to a given value.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void set(const shared_t<T[]>& src, size4_t strides, size4_t shape, T value, Stream& stream) {
        stream.enqueue([=]() { return set<SWAP_LAYOUT>(src.get(), strides, shape, value); });
    }
}
