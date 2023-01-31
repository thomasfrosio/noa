#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/traits/Utilities.h"

namespace noa {
    template<typename T>
    NOA_FHD constexpr void swap(T& a, T& b) {
        T tmp = a;
        a = b;
        b = tmp;
    }

    // In-place stable sort for small arrays.
    template<int N, typename T, typename U, typename = std::enable_if_t<N <= 4>>
    NOA_HD constexpr void small_stable_sort(T* begin, U&& comp) noexcept {
        if constexpr (N <= 1) {
            return;
        } else if constexpr (N == 2) {
            if (comp(begin[1], begin[0]))
                swap(begin[0], begin[1]);
        } else if constexpr (N == 3) {
            // Insertion sort:
            if (comp(begin[1], begin[0]))
                swap(begin[0], begin[1]);
            if (comp(begin[2], begin[1])) {
                swap(begin[1], begin[2]);
                if (comp(begin[1], begin[0]))
                    swap(begin[1], begin[0]);
            }
        } else if constexpr (N == 4) {
            // Sorting network, using insertion sort:
            if (comp(begin[3], begin[2]))
                swap(begin[3], begin[2]);
            if (comp(begin[2], begin[1]))
                swap(begin[2], begin[1]);
            if (comp(begin[3], begin[2]))
                swap(begin[3], begin[2]);
            if (comp(begin[1], begin[0]))
                swap(begin[1], begin[0]);
            if (comp(begin[2], begin[1]))
                swap(begin[2], begin[1]);
            if (comp(begin[3], begin[2]))
                swap(begin[3], begin[2]);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }

    // In-place stable sort for small arrays. Sort in ascending order.
    template<int N, typename T, typename = std::enable_if_t<N <= 4>>
    NOA_HD constexpr void small_stable_sort(T* begin) noexcept {
        small_stable_sort<N>(begin, [](const T& a, const T& b) { return a < b; });
    }
}
