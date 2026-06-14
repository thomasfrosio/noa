#pragma once

#include "noa/base/Config.hpp"

#include "noa/runtime/cpu/Sort.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/runtime/cuda/Sort.cuh"
#endif

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/Stream.hpp"
#include "noa/runtime/Traits.hpp"

// TODO expose backend allocation

namespace noa {
    struct SortOptions {
        /// Whether to sort in ascending or descending order.
        bool ascending{true};

        /// Axis along which to sort.
        /// The default is -1, which sorts along the first non-empty dimension from the right.
        /// Otherwise, it should be from 0 to N, excluded.
        i32 axis{-1};
    };

    /// Sorts an array, in-place.
    /// \param[in,out] array    Array to sort, in-place.
    /// \param options          Sorting options.
    /// \note The sort algorithms make temporary copies of the data when sorting along any but the last axis.
    ///       Consequently, sorting along the last axis is faster and uses less memory than sorting along any
    ///       other axis.
    template<nt::writable_array_decay_of_scalar Input>
    void sort(Input&& array, SortOptions options = {}) {
        check(not array.is_empty(), "Empty array detected");

        constexpr auto N = nt::array_size_v<Input>;
        if (options.axis == -1) {
            // Set to the first non-empty dimension, starting from the right.
            for (i32 i = N; i >= 0; --i) {
                if (array.shape()[i] > 1) {
                    options.axis = i;
                    break;
                }
            }
        } else {
            check(options.axis >= 0 and options.axis < N, "Invalid axis");
        }
        if (array.shape() == 1 or array.strides()[options.axis] == 0)
            return; // nothing to sort

        using accessor_t = AccessorRestrict<nt::value_type_t<Input>, N>;
        auto shape = array.shape();
        auto accessors = noa::make_tuple(accessor_t(array.data(), array.strides()));

        // Move empty dimensions to the left and try to reorder non-empty dimensions to rightmost.
        const auto optimal_order = nd::optimal_layout_for_accessors(shape, accessors);
        if (optimal_order != Vec<isize, N>::arange()) {
            shape = shape.permute(optimal_order);
            options.axis = optimal_order[options.axis];
            nd::permute_accessors(optimal_order, accessors);
        }

        // Collapse dimensions while making sure to preserve the sorted dimension.
        const auto contiguity = nd::accessors_contiguity(shape, accessors);
        const auto broadcasting = nd::accessors_broadcasting(shape, accessors);
        auto groups = Vec<isize, N>{};
        groups[options.axis] = 1;
        auto collapsed_shape = noa::collapse_contiguous_dimensions(shape, contiguity, broadcasting, groups);

        // Squeeze the newly empty dimensions to the left.
        const auto squeeze_order = noa::squeeze_empty_dimensions_left(collapsed_shape);
        collapsed_shape = collapsed_shape.permute(squeeze_order);
        options.axis = squeeze_order[options.axis];

        // Instead of permuting the accessors, reshape them to the new shape.
        if (nd::reshape_accessors(shape, collapsed_shape, accessors)) {
            shape = collapsed_shape;
        } else {
            panic("Reshape failed, shape={}, contiguity={}, broadcasting={}. Please report this issue",
                  shape, contiguity, broadcasting);
        }
        auto accessor = accessors[Tag<0>{}];

        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.enqueue([=, a = std::forward<Input>(array).share()]{
                noa::cpu::sort(accessor.data(), accessor.strides(), shape, options.ascending, options.axis);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::sort(accessor.data(), accessor.strides(), shape, options.ascending, options.axis, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(array));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    // TODO Add sort by keys.
}
