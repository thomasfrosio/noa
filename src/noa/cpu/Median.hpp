#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/indexing/Layout.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/cpu/AllocatorHeap.hpp"

#include <algorithm>

namespace noa::cpu {
    template<typename Value>
    Value median(Value* input, Strides4<i64> strides, Shape4<i64> shape, bool overwrite) {
        // Make it in rightmost order.
        const auto order = ni::order(strides, shape);
        strides = ni::reorder(strides, order);
        shape = ni::reorder(shape, order);

        // Allocate buffer only if necessary.
        const auto n_elements = shape.n_elements();
        Value* to_sort;
        typename AllocatorHeap<Value>::alloc_unique_type buffer;
        if (overwrite and ni::are_contiguous(strides, shape)) {
            to_sort = input;
        } else {
            buffer = AllocatorHeap<Value>::allocate(n_elements);
            copy(input, strides, buffer.get(), shape.strides(), shape, 1);
            to_sort = buffer.get();
        }

        std::nth_element(to_sort, to_sort + n_elements / 2, to_sort + n_elements);
        Value half = to_sort[n_elements / 2];
        if (is_odd(n_elements)) {
            return half;
        } else {
            std::nth_element(to_sort, to_sort + (n_elements - 1) / 2, to_sort + n_elements);
            return static_cast<Value>(to_sort[(n_elements - 1) / 2] + half) / Value{2}; // cast to silence integer promotion
        }
    }
}
#endif
