#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/memory/Linspace.hpp"

namespace noa::cpu::memory {
    // Returns evenly spaced values within a given interval.
    template<typename Value>
    Value linspace(Value* src, i64 elements, Value start, Value stop, bool endpoint = true) {
        NOA_ASSERT(src || !elements);
        if (elements <= 1) {
            if (elements)
                *src = start;
            return Value(0);
        }
        auto[count, delta, step] = noa::algorithm::memory::linspace_step(elements, start, stop, endpoint);
        for (i64 i = 0; i < count; ++i)
            src[i] = start + static_cast<Value>(i) * step;
        if (endpoint)
            src[elements - 1] = stop;
        return step;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename Value>
    Value linspace(Value* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
                   Value start, Value stop, bool endpoint = true) {
        if (indexing::are_contiguous(strides, shape))
            return linspace(src, shape.elements(), start, stop, endpoint);

        const auto elements = shape.elements();
        NOA_ASSERT(src || !elements);
        if (elements <= 1) {
            if (elements)
                *src = start;
            return Value(0);
        }

        const auto [kernel, step] = noa::algorithm::memory::linspace_4d<i64, i64>(
                src, strides, shape, start, stop, endpoint);
        noa::cpu::utils::iwise_4d(shape, kernel);
        return step;
    }

    // Returns evenly spaced values within a given interval.
    template<typename Value>
    inline Value linspace(const Shared<Value[]>& src, i64 elements,
                          Value start, Value stop, bool endpoint, Stream& stream) {
        auto[a_, b_, step] = noa::algorithm::memory::linspace_step(elements, start, stop, endpoint);
        stream.enqueue([=]() {
            linspace(src.get(), elements, start, stop, endpoint);
        });
        return step;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename Value>
    inline Value linspace(const Shared<Value[]>& src, const Strides4<i64>& strides, const Shape4<i64>& shape,
                          Value start, Value stop, bool endpoint, Stream& stream) {
        auto[a_, b_, step] = noa::algorithm::memory::linspace_step(shape.elements(), start, stop, endpoint);
        stream.enqueue([=]() {
            linspace(src.get(), strides, shape, start, stop, endpoint);
        });
        return step;
    }
}
