#pragma once

#include "noa/algorithms/memory/Iota.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/memory/Arange.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::memory {
    // Returns a tiled sequence [0, elements).
    template<typename T>
    void iota(T* src, i64 elements, i64 tile) {
        if (tile == elements)
            return arange(src, elements);

        NOA_ASSERT(src || !elements);
        for (i64 i = 0; i < elements; ++i, ++src)
            *src = static_cast<T>(i % tile);
    }

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T>
    void iota(T* src, const Strides4<i64>& strides, const Shape4<i64>& shape,
              const Vec4<i64>& tile, i64 threads) {
        if (noa::all(tile == shape.vec()))
            return arange(src, strides, shape);

        NOA_ASSERT(src && noa::all(shape > 0));
        const auto kernel = noa::algorithm::memory::iota_4d<i64, i64>(src, strides, shape, tile);
        noa::cpu::utils::iwise_4d(shape, kernel, threads);
    }
}
