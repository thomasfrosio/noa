#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/utils/EwiseUnary.h"

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
        NOA_ASSERT(!elements ||
                   (src && dst && !indexing::isOverlap(src, dim_t{1}, elements, dst, dim_t{1}, elements)));
        copy(src, src + elements, dst);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const shared_t<T[]>& src, const shared_t<T[]>& dst, dim_t elements, Stream& stream) {
        stream.enqueue([=](){
            copy(src.get(), dst.get(), elements);
        });
    }

    // Copies all logical elements from src to dst.
    template<typename Value>
    inline void copy(const Value* src, dim4_t src_strides, Value* dst, dim4_t dst_strides, dim4_t shape) {
        NOA_ASSERT(all(shape > 0) && src && dst &&
                   !indexing::isOverlap(src, src_strides, dst, dst_strides, shape));
        cpu::utils::ewiseUnary(src, src_strides, dst, dst_strides, shape, noa::copy_t{});
    }

    // Copies all logical elements from src to dst.
    template<typename Value>
    void copy(const shared_t<Value[]>& src, const dim4_t& src_strides,
              const shared_t<Value[]>& dst, const dim4_t& dst_strides,
              const dim4_t& shape, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && src && dst &&
                   !indexing::isOverlap(src, src_strides, dst, dst_strides, shape));
        const auto threads_omp = stream.threads();
        stream.enqueue([=]() {
            const Value* src_ptr = src.get();
            Value* dst_ptr = dst.get();
            cpu::utils::ewiseUnary(src_ptr, src_strides, dst_ptr, dst_strides, shape,
                                   noa::copy_t{}, threads_omp);
        });
    }
}
