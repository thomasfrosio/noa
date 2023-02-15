#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/memory/Arange.hpp"

namespace noa::cpu::memory {
    // Returns evenly spaced values within a given interval.
    template<typename Value>
    inline void arange(Value* src, i64 elements, Value start, Value step) {
        NOA_ASSERT(src || !elements);
        Value value = start;
        for (i64 i = 0; i < elements; ++i, value += step)
            src[i] = value;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename Value>
    inline void arange(Value* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
                       Value start, Value step, i64 threads) {
        if (indexing::are_contiguous(strides, shape))
            return arange(src, shape.elements(), start, step);

        NOA_ASSERT(src && noa::all(shape > 0));
        const auto kernel = noa::algorithm::memory::arange_4d<i64, i64>(src, strides, shape, start, step);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }
}
