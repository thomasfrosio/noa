#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/memory/Arange.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/algorithms/memory/Iota.hpp"

namespace noa::cpu::memory {
    // Returns a tiled sequence [0, elements).
    template<typename T>
    inline void iota(T* src, i64 elements, i64 tile) {
        if (tile == elements)
            return arange(src, elements);

        NOA_ASSERT(src || !elements);
        for (i64 i = 0; i < elements; ++i, ++src)
            *src = static_cast<T>(i % tile);
    }

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T>
    inline void iota(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape, const Vec4<i64>& tile) {
        if (noa::all(tile == shape.vec()))
            return arange(src, strides, shape);

        NOA_ASSERT(src && noa::all(shape > 0));
        const auto kernel = noa::algorithm::memory::iota_4d<i64, i64>(src, strides, shape, tile);
        noa::cpu::utils::iwise_4d(shape, kernel);
    }

    // Returns a tiled sequence [0, elements).
    template<typename T>
    inline void iota(const Shared<T[]>& src, i64 elements, i64 tile, Stream& stream) {
        stream.enqueue([=]() {
            iota(src.get(), elements, tile);
        });
    }

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T>
    inline void iota(const Shared<T[]>& src, const Strides4<i64>& strides, const Shape4<i64>& shape,
                     const Vec4<i64>& tile, Stream& stream) {
        stream.enqueue([=]() {
            iota(src.get(), strides, shape, tile);
        });
    }
}
