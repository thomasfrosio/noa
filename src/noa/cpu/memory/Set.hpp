#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

namespace noa::cpu::memory {
    // Sets an array to a given value.
    template<typename T>
    void set(T* first, T* last, T value) {
        // the cast is not necessary for basic types, but for Complex<> or VecX<>, it could help...
        if constexpr (traits::is_complex_v<T> || traits::is_intX_v<T> ||
                      traits::is_realX_v<T> || traits::is_matXX_v<T>) {
            using value_t = traits::value_type_t<T>;
            if (noa::all(value == T{0}))
                return std::fill(reinterpret_cast<value_t*>(first), reinterpret_cast<value_t*>(last), value_t{0});
        }
        // std::fill is calling memset, https://godbolt.org/z/1zEzTnoTK
        return std::fill(first, last, value);
    }

    // Sets an array to a given value.
    template<typename T>
    void set(T* src, i64 elements, T value) {
        NOA_ASSERT(src || !elements);
        set(src, src + elements, value);
    }

    // Sets an array to a given value.
    template<typename Value>
    void set(Value* src, const Strides4<i64>& strides, const Shape4<i64>& shape, Value value, i64 threads) {
        cpu::utils::ewise_unary(src, strides, src, strides, shape, [=](auto) { return value; }, threads);
    }
}
