#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Indexing.hpp"

namespace noa {
    template<typename T> class View;
    template<typename T> class Array;
}

namespace noa::indexing {
    /// Broadcasts an array to a given shape.
    template<typename Input, typename = std::enable_if_t<nt::is_varray_v<Input>>>
    [[nodiscard]] Input broadcast(const Input& input, const Shape4<i64>& shape) {
        auto strides = input.strides();
        if (!broadcast(input.shape(), strides, shape))
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}", input.shape(), shape);
        return Input(input.share(), shape, strides, input.options());
    }

    /// Whether \p lhs and \p rhs overlap in memory.
    template<typename Lhs, typename Rhs, typename = std::enable_if_t<nt::are_varray_v<Lhs, Rhs>>>
    [[nodiscard]] bool are_overlapped(const Lhs& lhs, const Rhs& rhs) {
        if (lhs.is_empty() || rhs.is_empty())
            return false;
        return are_overlapped(
                reinterpret_cast<uintptr_t>(lhs.get()),
                reinterpret_cast<uintptr_t>(lhs.get() + at((lhs.shape() - 1).vec(), lhs.strides())),
                reinterpret_cast<uintptr_t>(rhs.get()),
                reinterpret_cast<uintptr_t>(rhs.get() + at((rhs.shape() - 1).vec(), rhs.strides())));
    }

    /// Returns the multidimensional indexes of \p array corresponding to a memory \p offset.
    /// \note 0 indicates the beginning of the array. The array should not have any broadcast dimension.
    template<typename Input, typename = std::enable_if_t<nt::is_varray_v<Input>>>
    [[nodiscard]] constexpr Vec4<i64> offset2index(i64 offset, const Input& array) {
        NOA_CHECK(!any(array.strides() == 0),
                  "Cannot retrieve the 4D index from a broadcast array. Got strides:{}",
                  array.strides());
        return offset2index(offset, array.strides(), array.shape());
    }

    /// Whether the input is a contiguous vector.
    template<typename Input, typename = std::enable_if_t<nt::is_varray_v<Input>>>
    [[nodiscard]] constexpr bool is_contiguous_vector(const Input& input) {
        return is_vector(input.shape()) && input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous batch of contiguous vectors.
    template<typename Input, typename = std::enable_if_t<nt::is_varray_v<Input>>>
    [[nodiscard]] constexpr bool is_contiguous_vector_batched(const Input& input) {
        return is_vector(input.shape(), true) && input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous/strided batch of contiguous vectors.
    /// The batch stride doesn't have to be contiguous.
    template<typename Input, typename = std::enable_if_t<nt::is_varray_v<Input>>>
    [[nodiscard]] constexpr bool is_contiguous_vector_batched_strided(const Input& input) {
        return is_vector(input.shape(), true) && noa::all(input.is_contiguous().pop_front() == Vec3<bool>(true));
    }
}
