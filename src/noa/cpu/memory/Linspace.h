#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    template<typename T>
    std::tuple<dim_t, T, T> linspaceStep(dim_t elements, T start, T stop, bool endpoint = true) {
        const dim_t count = elements - static_cast<dim_t>(endpoint);
        const T delta = stop - start;
        const T step = delta / static_cast<T>(count);
        return {count, delta, step};
    }

    // Returns evenly spaced values within a given interval.
    template<typename T>
    T linspace(T* src, dim_t elements, T start, T stop, bool endpoint = true) {
        if (elements <= 1) {
            if (elements)
                *src = start;
            return T(0);
        }
        auto[count, delta, step] = linspaceStep(elements, start, stop, endpoint);
        for (dim_t i = 0; i < count; ++i)
            src[i] = start + static_cast<T>(i) * step;
        if (endpoint)
            src[elements - 1] = stop;
        return step;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T>
    T linspace(T* src, dim4_t strides, dim4_t shape,
               T start, T stop, bool endpoint = true) {
        if (indexing::areContiguous(strides, shape))
            return linspace(src, shape.elements(), start, stop, endpoint);

        const dim_t elements = shape.elements();
        if (elements <= 1) {
            if (elements)
                *src = start;
            return T(0);
        }
        auto[count, delta, step] = linspaceStep(elements, start, stop, endpoint);
        dim_t inc = 0;
        for (dim_t i = 0; i < shape[0]; ++i)
            for (dim_t j = 0; j < shape[1]; ++j)
                for (dim_t k = 0; k < shape[2]; ++k)
                    for (dim_t l = 0; l < shape[3]; ++l, ++inc)
                        src[indexing::at(i, j, k, l, strides)] = start + static_cast<T>(inc) * step;
        if (endpoint)
            src[indexing::at(shape - 1, strides)] = stop;
        return step;
    }

    // Returns evenly spaced values within a given interval.
    template<typename T>
    inline T linspace(const shared_t<T[]>& src, dim_t elements,
                      T start, T stop, bool endpoint, Stream& stream) {
        auto[a_, b_, step] = linspaceStep(elements, start, stop, endpoint);
        stream.enqueue([=]() {
            linspace(src.get(), elements, start, stop, endpoint);
        });
        return step;
    }

    // Returns evenly spaced values within a given interval, in the rightmost order.
    template<typename T>
    inline T linspace(const shared_t<T[]>& src, dim4_t strides, dim4_t shape,
                      T start, T stop, bool endpoint, Stream& stream) {
        auto[a_, b_, step] = linspaceStep(shape.elements(), start, stop, endpoint);
        stream.enqueue([=]() {
            linspace(src.get(), strides, shape, start, stop, endpoint);
        });
        return step;
    }
}
