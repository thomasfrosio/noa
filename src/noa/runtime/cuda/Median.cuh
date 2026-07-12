#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Utils.hpp"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/Allocators.hpp"
#include "noa/runtime/cuda/Sort.cuh"

namespace noa::cuda {
    template<typename T, usize N>
    auto median(
        T* input,
        Strides<isize, N> strides,
        Shape<isize, N> shape,
        bool overwrite,
        Stream& stream
    ) {
        nd::permute_all_to_rightmost_order(strides, shape, strides, shape);

        const auto n_elements = shape.n_elements();
        using mut_t = std::remove_const_t<T>;
        AllocatorDevice::allocate_type<mut_t> buffer;
        mut_t* to_sort{};
        if constexpr (std::is_const_v<T>) {
            buffer = AllocatorDevice::allocate_async<mut_t>(n_elements, stream);
            to_sort = buffer.get();
            copy(input, strides, to_sort, shape.strides(), shape, stream);
        } else {
            if (overwrite and strides.is_contiguous(shape)) {
                to_sort = input;
            } else {
                buffer = AllocatorDevice::allocate_async<mut_t>(n_elements, stream);
                to_sort = buffer.get();
                copy(input, strides, to_sort, shape.strides(), shape, stream);
            }
        }

        // Sort the entire contiguous array, in-place.
        sort(to_sort, Strides<isize, 1>{1}, Shape<isize, 1>{n_elements}, true, 0, stream);

        // Retrieve the median.
        const bool is_even = noa::is_even(n_elements);
        mut_t out[2];
        copy(to_sort + (n_elements - is_even) / 2, out, 1 + is_even, stream);
        stream.synchronize();

        if (is_even)
            return (out[0] + out[1]) / mut_t{2};
        return out[0];
    }
}
