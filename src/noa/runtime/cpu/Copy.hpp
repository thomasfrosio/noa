#pragma once

#include <algorithm>
#include "noa/runtime/cpu/Ewise.hpp"

namespace noa::cpu {
    /// Copies all elements in the range [first, last) starting from first and proceeding to last - 1.
    /// The behavior is undefined if dst_first is within the range [first, last).
    template<typename T>
    void copy(const T* first, const T* last, T* dst_first) {
        std::copy(first, last, dst_first);
    }

    /// Copies src into dst.
    template<typename T>
    void copy(const T* src, T* dst, isize n_elements) {
        copy(src, src + n_elements, dst);
    }

    /// Copies all logical elements from src to dst.
    template<typename T, usize N>
    void copy(
        const T* src, const Strides<isize, N>& src_strides,
        T* dst, const Strides<isize, N>& dst_strides,
        const Shape<isize, N>& shape, i32 n_threads
    ) {
        if (shape.n_elements() == 1) {
            *dst = *src;
            return;
        }
        ewise(shape, Copy{},
              noa::make_tuple(AccessorRestrict<const T, N, isize>(src, src_strides)),
              noa::make_tuple(AccessorRestrict<T, N, isize>(dst, dst_strides)),
              n_threads);
    }
}
