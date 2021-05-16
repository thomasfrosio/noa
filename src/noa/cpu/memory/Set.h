#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Memory {
    /**
     * Initializes or sets device memory to a value.
     * @tparam T            Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
     * @param[out] first    The beginning of range to set.
     * @param[out] last     The end of range to set.
     * @param value         The value to assign.
     */
    template<typename T>
    NOA_IH void set(T* first, T* last, T value) {
        if constexpr (Noa::Traits::is_scalar_v<T> || Noa::Traits::is_complex_v<T> ||
                      Noa::Traits::is_intX_v<T> || Noa::Traits::is_floatX_v<T>) {
            if (value == T{0})
                // calling memset, https://godbolt.org/z/1zEzTnoTK
                // the cast is not necessary for basic types, but for Complex<>, Int<> or Float<>, it could help...
                std::fill(reinterpret_cast<char*>(first), reinterpret_cast<char*>(last), 0);
            else
                std::fill(first, last, value);
        } else {
            std::fill(first, last, value);
        }
    }

    /**
     * Initializes or sets device memory to a value.
     * @tparam T            Most types are supported.
     * @param[out] first    The beginning of range to set.
     * @param elements      Number of elements to set.
     * @param value         The value to assign.
     */
    template<typename T>
    NOA_IH void set(T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }
}
