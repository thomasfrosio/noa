#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    // Copies all elements in the range [first, last) starting from first and proceeding to last - 1.
    // The behavior is undefined if dst_first is within the range [first, last).
    template<typename T>
    inline void copy(const T* first, const T* last, T* dst_first) {
        std::copy(first, last, dst_first);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const T* src, T* dst, size_t elements) {
        copy(src, src + elements, dst);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const shared_t<T[]>& src, const shared_t<T[]>& dst, size_t elements, Stream& stream) {
        stream.enqueue([=](){
            copy(src.get(), dst.get(), elements);
        });
    }

    // Copies all logical elements from \p src to \p dst.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void copy(const T* src, size4_t src_strides, T* dst, size4_t dst_strides, size4_t shape) {
        if constexpr (SWAP_LAYOUT) {
            // Loop through the destination in the most cache-friendly way:
            const size4_t order = indexing::order(dst_strides, shape);
            shape = indexing::reorder(shape, order);
            dst_strides = indexing::reorder(dst_strides, order);
            src_strides = indexing::reorder(src_strides, order);
            // We could have selected the source, but since the destination is less likely to be
            // broadcast, it seems safer. This could make things worse for some edge cases.
        }

        if (indexing::areContiguous(src_strides, shape) && indexing::areContiguous(dst_strides, shape))
            return copy(src, src + shape.elements(), dst);

        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = 0; l < shape[3]; ++l)
                        dst[indexing::at(i, j, k, l, dst_strides)] = src[indexing::at(i, j, k, l, src_strides)];
    }

    // Copies all logical elements from \p src to \p dst.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void copy(const shared_t<T[]>& src, size4_t src_strides,
                     const shared_t<T[]>& dst, size4_t dst_strides, size4_t shape, Stream& stream) {
        stream.enqueue([=]() {
            return copy<SWAP_LAYOUT>(src.get(), src_strides, dst.get(), dst_strides, shape);
        });
    }
}
