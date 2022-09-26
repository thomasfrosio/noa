#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/memory/Arange.h"

namespace noa::cpu::memory {
    // Returns a tiled sequence [0, elements).
    template<typename T>
    inline void iota(T* src, dim_t elements, dim_t tile) {
        if (tile == elements)
            return arange(src, elements);

        for (dim_t i = 0; i < elements; ++i, ++src)
            *src = static_cast<T>(i % tile);
    }

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T>
    inline void iota(T* src, dim4_t strides, dim4_t shape, dim4_t tile) {
        if (all(tile == shape))
            return arange(src, strides, shape);

        const dim4_t tile_strides = tile.strides();
        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {
                for (dim_t k = 0; k < shape[2]; ++k) {
                    for (dim_t l = 0; l < shape[3]; ++l) {
                        const dim_t value = indexing::at(i % tile[0],
                                                         j % tile[1],
                                                         k % tile[2],
                                                         l % tile[3],
                                                         tile_strides);
                        src[indexing::at(i, j, k, l, strides)] = static_cast<T>(value);
                    }
                }
            }
        }
    }

    // Returns a tiled sequence [0, elements).
    template<typename T>
    inline void iota(const shared_t<T[]>& src, dim_t elements, dim_t tile, Stream& stream) {
        stream.enqueue([=]() {
            iota(src.get(), elements, tile);
        });
    }

    // Returns a tiled sequence [0, elements), in the rightmost order.
    template<typename T>
    inline void iota(const shared_t<T[]>& src, dim4_t strides, dim4_t shape, dim4_t tile, Stream& stream) {
        stream.enqueue([=]() {
            iota(src.get(), strides, shape, tile);
        });
    }
}
