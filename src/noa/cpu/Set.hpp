#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <algorithm>
#include "noa/core/Types.hpp"
#include "noa/cpu/Ewise.hpp"

namespace noa::cpu {
    // Fills an array to a given value.
    template<typename T>
    void fill(T* first, T* last, T value) {
        // the cast is not necessary for basic types, but for Complex<> or VecX<>, it could help...
        if constexpr (nt::is_complex_v<T> or nt::is_vec_int_v<T> or nt::is_vec_real_v<T> or nt::is_mat_v<T>) {
            using value_t = nt::value_type_t<T>;
            if (all(value == T{0}))
                return std::fill(reinterpret_cast<value_t*>(first), reinterpret_cast<value_t*>(last), value_t{0});
        }
        // std::fill is calling memset, https://godbolt.org/z/1zEzTnoTK
        return std::fill(first, last, value);
    }

    // Fills an array with a given value.
    template<typename T>
    void fill(T* src, i64 elements, T value) {
        NOA_ASSERT(src or !elements);
        fill(src, src + elements, value);
    }

    // Fills an array with a given value.
    template<typename T>
    void fill(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape, T value, i64 threads) {
        ewise(shape, [=](T& e) { e = value; },
              make_tuple(AccessorI64<T, 4>(src, strides)),
              make_tuple(),
              threads);
    }
}
#endif
