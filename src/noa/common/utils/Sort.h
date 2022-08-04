#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa {
    /// In-place stable sort for small arrays.
    template<int N, typename T, typename U>
    NOA_HD constexpr void smallStableSort(T* begin, U&& comp) noexcept {
        constexpr auto swap = [](T& a, T& b) {
            T tmp = a;
            a = b;
            b = tmp;
        };

        if constexpr (N <= 0) {
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
            static_assert(traits::always_false_v<T>);
        }
    }

    /// In-place stable sort for small arrays. Sort in ascending order.
    template<int N, typename T>
    NOA_HD constexpr void smallStableSort(T* begin) noexcept {
        smallStableSort<N>(begin, [](const T& a, const T& b) { return a < b; });
    }
}
