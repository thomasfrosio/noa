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
    NOA_IH void set(const T* first, const T* last, T value) {
        if (value == 0)
            std::fill(first, last, 0); // calling memset, https://godbolt.org/z/1zEzTnoTK
        else
            std::fill(first, last, value);
    }

    /**
     * Initializes or sets device memory to a value.
     * @tparam T            Most types are supported.
     * @param[out] first    The beginning of range to set.
     * @param elements      Number of elements to set.
     * @param value         The value to assign.
     */
    template<typename T>
    NOA_IH void set(const T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }
}
