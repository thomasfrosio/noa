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
        return std::fill(first, last, value);
    }

    // Fills an array with a given value.
    template<typename T>
    void fill(T* src, isize elements, T value) {
        NOA_ASSERT(src or not elements);
        fill(src, src + elements, value);
    }

    // Fills an array with a given value.
    template<typename T, usize N>
    void fill(T* src, const Strides<isize, N>& strides, const Shape<isize, N>& shape, T value, isize n_threads) {
        ewise(shape, Fill<T>{value},
              noa::make_tuple(),
              noa::make_tuple(Accessor<T, N, isize>(src, strides)),
              n_threads);
    }
}
