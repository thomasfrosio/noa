#pragma once

#include <algorithm>
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/cpu/Allocators.hpp"

namespace noa::cpu {
    template<typename T>
    auto median(T* input, Strides4<i64> strides, Shape4<i64> shape, bool overwrite) {
        // Make it in rightmost order.
        const auto order = ni::order(strides, shape);
        strides = ni::reorder(strides, order);
        shape = ni::reorder(shape, order);

        const auto n_elements = shape.n_elements();

        // Allocate buffer if necessary.
        using mut_t = std::remove_const_t<T>;
        mut_t* to_sort;
        AllocatorHeap::allocate_type<mut_t> buffer;
        if constexpr (std::is_const_v<T>) {
            buffer = AllocatorHeap::allocate<mut_t>(n_elements);
            copy(input, strides, buffer.get(), shape.strides(), shape, 1);
            to_sort = buffer.get();
        } else {
            if (overwrite and ni::are_contiguous(strides, shape)) {
                to_sort = input;
            } else {
                buffer = AllocatorHeap::allocate<mut_t>(n_elements);
                copy(input, strides, buffer.get(), shape.strides(), shape, 1);
                to_sort = buffer.get();
            }
        }

        std::nth_element(to_sort, to_sort + n_elements / 2, to_sort + n_elements);
        mut_t half = to_sort[n_elements / 2];
        if (is_odd(n_elements))
            return half;
        std::nth_element(to_sort, to_sort + (n_elements - 1) / 2, to_sort + n_elements);
        return static_cast<T>(to_sort[(n_elements - 1) / 2] + half) / mut_t{2}; // cast to silence integer promotion
    }
}
