#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm>

#include "noa/core/Types.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/cpu/Ewise.hpp"
#include "noa/core/Operators.hpp"

namespace noa::cpu {
    /// Copies all elements in the range [first, last) starting from first and proceeding to last - 1.
    /// The behavior is undefined if dst_first is within the range [first, last).
    template<typename T>
    void copy(const T* first, const T* last, T* dst_first) {
        NOA_ASSERT(first and last and dst_first);
        std::copy(first, last, dst_first);
    }

    /// Copies src into dst.
    template<typename T>
    void copy(const T* src, T* dst, i64 elements) {
        NOA_ASSERT(not elements or (src and dst and not ni::are_overlapped(src, elements, dst, elements)));
        copy(src, src + elements, dst);
    }

    /// Copies all logical elements from src to dst.
    template<typename T>
    void copy(
            const T* src, const Strides4<i64>& src_strides,
            T* dst, const Strides4<i64>& dst_strides,
            const Shape4<i64>& shape, i64 threads
    ) {
        NOA_ASSERT(all(shape > 0) and src and dst and not ni::are_overlapped(src, src_strides, dst, dst_strides, shape));
        ewise(shape, Copy{},
              make_tuple(AccessorRestrictI64<const T, 4>(src, src_strides)),
              make_tuple(AccessorRestrictI64<T, 4>(dst, dst_strides)),
              threads);
    }
}
#endif
