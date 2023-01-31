#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/utils/EwiseUnary.h"

namespace noa::cpu::memory {
    // Sets an array to a given value.
    template<typename T>
    inline void set(T* first, T* last, T value) {
        // the cast is not necessary for basic types, but for Complex<>, IntX<> or FloatX<>, it could help...
        if constexpr (traits::is_complex_v<T> || traits::is_intX_v<T> ||
                      traits::is_floatX_v<T> || traits::is_floatXX_v<T>) {
            using value_t = traits::value_type_t<T>;
            if (noa::all(value == T{0}))
                return std::fill(reinterpret_cast<value_t*>(first), reinterpret_cast<value_t*>(last), value_t{0});
        }
        // std::fill is calling memset, https://godbolt.org/z/1zEzTnoTK
        return std::fill(first, last, value);
    }

    // Sets an array to a given value.
    template<typename T>
    inline void set(T* src, dim_t elements, T value) {
        NOA_ASSERT(src || !elements);
        set(src, src + elements, value);
    }

    // Sets an array to a given value.
    template<typename T>
    void set(const shared_t<T[]>& src, dim_t elements, T value, Stream& stream) {
        stream.enqueue([=]() { return set(src.get(), elements, value); });
    }

    // Sets an array to a given value.
    template<typename Value>
    void set(Value* src, const dim4_t& strides, const dim4_t& shape, Value value) {
        cpu::utils::ewiseUnary(src, strides, src, strides, shape, [=](auto) { return value; });
    }

    // Sets an array to a given value.
    template<typename T>
    void set(const shared_t<T[]>& src, dim4_t strides, dim4_t shape, T value, Stream& stream) {
        stream.enqueue([=]() { return set(src.get(), strides, shape, value); });
    }
}
