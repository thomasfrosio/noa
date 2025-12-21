#pragma once

#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Layout.hpp"
#include "noa/runtime/core/Offset.hpp"
#include "noa/runtime/Traits.hpp"

namespace noa {
    /// Broadcasts an array to a given shape.
    template<nt::varray_decay T>
    [[nodiscard]] auto broadcast(T&& input, const Shape4& shape) {
        auto strides = input.strides();
        if (not broadcast(input.shape(), strides, shape))
            panic("Cannot broadcast an array of shape {} into an array of shape {}", input.shape(), shape);
        return std::decay_t<T>(std::forward<T>(input).share(), shape, strides, input.options());
    }
}

namespace noa {
    /// Whether \p lhs and \p rhs overlap in memory.
    [[nodiscard]] bool are_overlapped(const nt::varray auto& lhs, const nt::varray auto& rhs) {
        if (lhs.is_empty() or rhs.is_empty())
            return false;
        return details::are_overlapped(
            reinterpret_cast<uintptr_t>(lhs.get()),
            reinterpret_cast<uintptr_t>(lhs.get() + offset_at(lhs.strides(), (lhs.shape() - 1).vec)),
            reinterpret_cast<uintptr_t>(rhs.get()),
            reinterpret_cast<uintptr_t>(rhs.get() + offset_at(rhs.strides(), (rhs.shape() - 1).vec)));
    }

    /// Returns the multidimensional indexes of \p array corresponding to a memory \p offset.
    /// \note 0 indicates the beginning of the array. The array should not have any broadcast dimension.
    [[nodiscard]] constexpr auto offset2index(isize offset, const nt::varray auto& array) -> Vec<isize, 4> {
        check(array.strides() > 0,
              "Cannot retrieve the 4d index from a broadcast array. Got strides:{}",
              array.strides());
        return offset2index(offset, array.strides(), array.shape());
    }

    /// Whether the input is a contiguous vector.
    [[nodiscard]] constexpr bool is_contiguous_vector(const nt::varray auto& input) {
        return is_vector(input.shape()) and input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous batch of contiguous vectors.
    [[nodiscard]] constexpr bool is_contiguous_vector_batched(const nt::varray auto& input) {
        return is_vector(input.shape(), true) and input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous/strided batch of contiguous vectors.
    /// The batch stride doesn't have to be contiguous.
    [[nodiscard]] constexpr bool is_contiguous_vector_batched_strided(const nt::varray auto& input) {
        return is_vector(input.shape(), true) and input.is_contiguous().pop_front() == true;
    }
}
