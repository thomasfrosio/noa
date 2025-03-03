#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/Allocators.hpp"
#include "noa/gpu/cuda/Sort.cuh"

namespace noa::cuda {
    template<typename T>
    T median(
        T* input,
        Strides4<i64> strides,
        Shape4<i64> shape,
        bool overwrite,
        Stream& stream
    ) {
        const auto order = ni::order(strides, shape);
        strides = ni::reorder(strides, order);
        shape = ni::reorder(shape, order);

        const auto n_elements = shape.n_elements();
        typename AllocatorDevice<T>::unique_type buffer;
        T* to_sort{};
        if (overwrite and ni::are_contiguous(strides, shape)) {
            to_sort = input;
        } else {
            buffer = AllocatorDevice<T>::allocate_async(n_elements, stream);
            to_sort = buffer.get();
            copy(input, strides, to_sort, shape.strides(), shape, stream);
        }

        // Sort the entire contiguous array.
        const auto shape_1d = Shape4<i64>{1, 1, 1, n_elements};
        sort(to_sort, shape_1d.strides(), shape_1d, true, -1, stream);

        // Retrieve the median.
        const bool is_even = noa::is_even(n_elements);
        T out[2];
        copy(to_sort + (n_elements - is_even) / 2, out, 1 + is_even, stream);
        stream.synchronize();

        if (is_even)
            return (out[0] + out[1]) / T{2};
        return out[0];
    }
}
