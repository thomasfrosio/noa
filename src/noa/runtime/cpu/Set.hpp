#pragma once

#include <algorithm>
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Ewise.hpp"

#include "noa/base/Complex.hpp"
namespace noa::cpu {
    // Fills an array to a given value.
    template<typename T>
    void fill(T* first, T* last, T value) {
        // std::fill is calling memset, https://godbolt.org/z/1zEzTnoTK
        return std::fill(first, last, value);
    }

    // Fills an array with a given value.
    template<typename T>
    void fill(T* src, isize elements, T value) {
        NOA_ASSERT(src or not elements);
        fill(src, src + elements, value);
    }

    // Fills an array with a given value.
    template<typename T>
    void fill(T* src, const Strides4& strides, const Shape4& shape, T value, isize n_threads) {
        ewise(shape, Fill<T>{value},
              make_tuple(),
              make_tuple(Accessor<T, 4>(src, strides)),
              n_threads);
    }
}
