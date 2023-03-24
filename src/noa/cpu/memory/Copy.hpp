#pragma once

#include <algorithm>

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

namespace noa::cpu::memory {
    // Copies all elements in the range [first, last) starting from first and proceeding to last - 1.
    // The behavior is undefined if dst_first is within the range [first, last).
    template<typename T>
    void copy(const T* first, const T* last, T* dst_first) {
        NOA_ASSERT(first && last && dst_first);
        std::copy(first, last, dst_first);
    }

    // Copies src into dst.
    template<typename T>
    void copy(const T* src, T* dst, i64 elements) {
        NOA_ASSERT(!elements || (src && dst && !noa::indexing::are_overlapped(src, elements, dst, elements)));
        copy(src, src + elements, dst);
    }

    // Copies all logical elements from src to dst.
    template<typename T>
    void copy(const T* src, const Strides4<i64>& src_strides,
              T* dst, const Strides4<i64>& dst_strides,
              const Shape4<i64>& shape, i64 threads) {
        NOA_ASSERT(noa::all(shape > 0) && src && dst &&
                   !noa::indexing::are_overlapped(src, src_strides, dst, dst_strides, shape));
        cpu::utils::ewise_unary(src, src_strides, dst, dst_strides, shape, noa::copy_t{}, threads);
    }
}
