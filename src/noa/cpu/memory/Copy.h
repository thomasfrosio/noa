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
        NOA_ASSERT(first && last && dst_first);
        std::copy(first, last, dst_first);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const T* src, T* dst, dim_t elements) {
        NOA_ASSERT(!elements || !indexing::isOverlap(src, dim_t{1}, elements, dst, dim_t{1}, elements));
        copy(src, src + elements, dst);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const shared_t<T[]>& src, const shared_t<T[]>& dst, dim_t elements, Stream& stream) {
        stream.enqueue([=](){
            copy(src.get(), dst.get(), elements);
        });
    }

    // Copies all logical elements from \p src to \p dst.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void copy(const T* src, dim4_t src_strides, T* dst, dim4_t dst_strides, dim4_t shape) {
        if constexpr (SWAP_LAYOUT) {
            // Loop through the destination in the most cache-friendly way:
            const dim4_t order = indexing::order(dst_strides, shape);
            shape = indexing::reorder(shape, order);
            dst_strides = indexing::reorder(dst_strides, order);
            src_strides = indexing::reorder(src_strides, order);
            // We could have selected the source, but since the destination is less likely to be
            // broadcast, it seems safer. This could make things worse for some edge cases.
        }

        if (indexing::areContiguous(src_strides, shape) && indexing::areContiguous(dst_strides, shape))
            return copy(src, src + shape.elements(), dst);

        {
            NOA_ASSERT(all(shape > 0) && !indexing::isOverlap(src, src_strides, dst, dst_strides, shape));
            const AccessorReferenceRestrict<const T, 4, dim_t> input(src, src_strides.get());
            const AccessorReferenceRestrict<T, 4, dim_t> output(dst, dst_strides.get());
            for (dim_t i = 0; i < shape[0]; ++i)
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = 0; k < shape[2]; ++k)
                        for (dim_t l = 0; l < shape[3]; ++l)
                            output(i, j, k, l) = input(i, j, k, l);
        }
    }

    // Copies all logical elements from \p src to \p dst.
    template<bool SWAP_LAYOUT = true, typename T>
    inline void copy(const shared_t<T[]>& src, dim4_t src_strides,
                     const shared_t<T[]>& dst, dim4_t dst_strides, dim4_t shape, Stream& stream) {
        stream.enqueue([=]() {
            return copy<SWAP_LAYOUT>(src.get(), src_strides, dst.get(), dst_strides, shape);
        });
    }
}
