#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/Indexing.hpp"
#include "noa/unified/Traits.hpp"

namespace noa {
    template<typename T> class View;
    template<typename T> class Array;
}

namespace noa::indexing {
    /// Broadcasts an array to a given shape.
    template<typename Input, nt::enable_if_bool_t<nt::is_varray_v<Input>> = true>
    [[nodiscard]] Input broadcast(const Input& input, const Shape4<i64>& shape) {
        auto strides = input.strides();
        if (!broadcast(input.shape(), strides, shape))
            noa::panic("Cannot broadcast an array of shape {} into an array of shape {}", input.shape(), shape);
        return Input(input.share(), shape, strides, input.options());
    }

    /// Whether \p lhs and \p rhs overlap in memory.
    template<typename Lhs, typename Rhs, nt::enable_if_bool_t<nt::are_varray_v<Lhs, Rhs>> = true>
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
    template<typename Input, nt::enable_if_bool_t<nt::is_varray_v<Input>> = true>
    [[nodiscard]] constexpr Vec4<i64> offset2index(i64 offset, const Input& array) {
        noa::check(!any(array.strides() == 0),
                   "Cannot retrieve the 4D index from a broadcast array. Got strides:{}",
                   array.strides());
        return offset2index(offset, array.strides(), array.shape());
    }

    /// Whether the input is a contiguous vector.
    template<typename Input, nt::enable_if_bool_t<nt::is_varray_v<Input>> = true>
    [[nodiscard]] constexpr bool is_contiguous_vector(const Input& input) {
        return is_vector(input.shape()) && input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous batch of contiguous vectors.
    template<typename Input, nt::enable_if_bool_t<nt::is_varray_v<Input>> = true>
    [[nodiscard]] constexpr bool is_contiguous_vector_batched(const Input& input) {
        return is_vector(input.shape(), true) && input.are_contiguous();
    }

    /// Whether the input is a contiguous vector or a contiguous/strided batch of contiguous vectors.
    /// The batch stride doesn't have to be contiguous.
    template<typename Input, nt::enable_if_bool_t<nt::is_varray_v<Input>> = true>
    [[nodiscard]] constexpr bool is_contiguous_vector_batched_strided(const Input& input) {
        return is_vector(input.shape(), true) &&
               noa::all(input.is_contiguous().pop_front() == Vec3<bool>::filled_with(true));
    }

    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr bool are_all_same_shape(const Tuple<Varrays&&...>& varrays) {
        if constexpr (sizeof...(Varrays) > 1) {
            return varrays.all([&](const auto& varray) {
                if constexpr (nt::is_varray_v<decltype(varray)>)
                    return all(varrays[Tag<0>{}].shape() == varray.shape());
                return true;
            });
        } else {
            return true;
        }
    }

    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr bool are_all_same_order(const Tuple<Varrays&&...>& varrays) {
        if constexpr (sizeof...(Varrays) > 1) {
            auto order = ni::order(varrays[Tag<0>{}].strides(), varrays[Tag<0>{}].shape());
            return varrays.all([&order](const auto& varray) {
                if constexpr (nt::is_varray_v<decltype(varray)>)
                    return all(order == ni::order(varray.strides(), varray.shape()));
                return true;
            });
        } else {
            return true;
        }
    }

    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr bool are_all_same_device(const Tuple<Varrays&&...>& varrays) {
        if constexpr (sizeof...(Varrays) > 1) {
            return varrays.all([&](const auto& varray) {
                if constexpr (nt::is_varray_v<decltype(varray)>)
                    return all(varrays[Tag<0>{}].device() == varray.device());
                return true;
            });
        } else {
            return true;
        }
    }

    /// Permutes the dimensions of the view.
    /// \param permutation  Permutation with the axes numbered from 0 to 3.
    template<typename VArray, nt::enable_if_bool_t<nt::is_varray_v<VArray>> = true>
    [[nodiscard]] constexpr VArray reorder(const VArray& varray, const Vec4<i64>& permutation) {
        return varray.reorder();
    }

    /// Find the "best" order for index-wise access:
    /// \tparam Varrays
    /// \param varrays
    /// \param reorder
    /// \return
    template<typename... Accessors, typename Index, nt::enable_if_bool_t<nt::are_accessor_v<Accessors...>> = true>
    [[nodiscard]] constexpr auto reorder(
            const Tuple<Accessors...>& varrays,
            const Vec4<Index>& order
    ) {
        if (all(order == Vec4<Index>{0, 1, 2, 3}))
            return varrays;
        varrays.for_each([&order](auto& accessor) {
            accessor.reorder(order);
        });

    }


    // find_best_order()
    // if they don't have the same order: no reorder
    // otherwise, get order.



    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto forward_as_tuple_of_shapes(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([](const auto& varray) {
            if constexpr (nt::is_varray_v<decltype(varray)>)
                return varray.shape();
            else
                return Shape4<i64>::filled_with(1);
        }); // FIXME decay()?
    }
    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto make_tuple_of_shapes(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_shapes(varrays).decay();
    }

    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto forward_as_tuple_of_strides(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([](const auto& varray) {
            if constexpr (nt::is_varray_v<decltype(varray)>)
                return varray.strides();
            else
                return Strides4<i64>{};
        });
    }
    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto make_tuple_of_strides(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_strides(varrays).decay();
    }

    template<typename... Varrays>
    [[nodiscard]] constexpr auto forward_as_tuple_of_devices(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([](const auto& varray) {
            if constexpr (nt::is_varray_v<decltype(varray)>)
                return varray.devices();
            else
                return Empty{}; // FIXME
        });
    }
    template<typename... Varrays>
    [[nodiscard]] constexpr auto make_tuple_of_devices(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_devices(varrays).decay();
    }

    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto forward_as_tuple_of_accessors(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([](auto&& v) {
            using v_t = decltype(v);
            if constexpr (nt::is_varray_v<v_t>) {
                return v.accessor();
            } else {
                return AccessorValue(std::forward<v_t>(v));
            }
        });
    }
    template<typename... Varrays, nt::enable_if_bool_t<nt::are_varray_v<Varrays...>> = true>
    [[nodiscard]] constexpr auto make_tuple_of_accessors(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_accessors(varrays).decay();
    }

    template<typename... T>
    inline constexpr auto wrap(T&& ... a) noexcept {
        return Tuple<const T&...>{std::forward<T>(a)...};
    }
}
