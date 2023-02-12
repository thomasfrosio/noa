#pragma once

#include <algorithm>

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

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
    inline void copy(const T* src, T* dst, i64 elements) {
        NOA_ASSERT(!elements ||
                   (src && dst && !noa::indexing::are_overlapped(src, 1, elements, dst, 1, elements)));
        copy(src, src + elements, dst);
    }

    // Copies src into dst.
    template<typename T>
    inline void copy(const Shared<T[]>& src, const Shared<T[]>& dst, i64 elements, Stream& stream) {
        stream.enqueue([=](){
            copy(src.get(), dst.get(), elements);
        });
    }

    // Copies all logical elements from src to dst.
    template<typename Value>
    inline void copy(const Value* src, const Strides4<i64>& src_strides,
                     Value* dst, const Strides4<i64>& dst_strides,
                     const Shape4<i64>& shape) {
        NOA_ASSERT(noa::all(shape > 0) && src && dst &&
                   !noa::indexing::are_overlapped(src, src_strides, dst, dst_strides, shape));
        cpu::utils::ewise_unary(src, src_strides, dst, dst_strides, shape, noa::copy_t{});
    }

    // Copies all logical elements from src to dst.
    template<typename Value>
    void copy(const Shared<Value[]>& src, const Strides4<i64>& src_strides,
              const Shared<Value[]>& dst, const Strides4<i64>& dst_strides,
              const Shape4<i64>& shape, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0) && src && dst &&
                   !noa::indexing::are_overlapped(src.get(), src_strides, dst.get(), dst_strides, shape));
        const auto threads_omp = stream.threads();
        stream.enqueue([=]() {
            cpu::utils::ewise_unary<const Value>(
                    src.get(), src_strides, dst.get(), dst_strides, shape,
                    noa::copy_t{}, threads_omp);
        });
    }
}
