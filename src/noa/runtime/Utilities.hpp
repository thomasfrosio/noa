#pragma once

#include "noa/base/Tuple.hpp"
#include "noa/base/Vec.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shareable.hpp"
#include "noa/runtime/core/Batch.hpp"
#include "noa/runtime/Traits.hpp"

namespace noa::details {
    /// Extracts the accessors from the varrays in the tuple.
    /// For types other than varrays, forward the object into an AccessorValue.
    template<bool EnforceConst = false, nt::tuple_decay T>
    [[nodiscard]] constexpr auto to_tuple_of_accessors(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& v) {
            if constexpr (nt::varray_decay<U>) {
                return to_accessor<AccessorConfig{.enforce_const = EnforceConst}>(v);
            } else {
                return to_accessor_value<EnforceConst>(std::forward<U>(v));
            }
        });
    }

    /// Reorders the tuple(s) of accessors (in-place).
    template<typename Index, nt::tuple_of_accessor_or_empty... T>
    constexpr void reorder_accessors(const Vec<Index, 4>& order, T&... accessors) {
        (accessors.for_each([&order]<typename U>(U& accessor) {
            if constexpr (nt::accessor_pure<U>)
                accessor.reorder(order);
        }), ...);
    }

    template<typename Tup>
    [[nodiscard]] constexpr bool are_all_varrays() {
        return []<typename... T>(nt::TypeList<T...>) {
            return nt::varray_decay<T...>;
        }(nt::type_list_t<Tup>{});
    }

    template<typename Tup>
    [[nodiscard]] constexpr bool are_all_value_types_trivially_copyable() {
        return []<typename... T>(nt::TypeList<T...>) {
            return (std::is_trivially_copyable_v<nt::value_type_t<T>> and ...);
        }(nt::type_list_t<Tup>{});
    }

    /// Returns the index of the first varray in the tuple.
    /// Returns -1 if there's no varray in the tuple.
    template<nt::tuple_decay T>
    [[nodiscard]] constexpr isize index_of_first_varray() {
        isize index{-1};
        auto is_varray = [&]<usize I>() {
            if constexpr (nt::varray_decay<decltype(std::declval<T>()[Tag<I>{}])>) {
                index = I;
                return true;
            }
            return false;
        };
        [&is_varray]<usize... I>(std::index_sequence<I...>) {
            (is_varray.template operator()<I>() or ...);
        }(nt::index_list_t<T>{});
        return index;
    }

    /// Extracts the shareable handles of various objects (mostly intended for View and Array).
    template<nt::tuple_decay T>
    [[nodiscard]] auto extract_shared_handle(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& value) {
            if constexpr (nt::shareable<U>)
                return std::forward<U>(value);
            else if constexpr (nt::shareable_using_share<U>)
                return std::forward<U>(value).share();
            else
                return Empty{};
        });
    }

    /// Extracts the shared handles from Arrays.
    template<nt::tuple_decay T>
    [[nodiscard]] auto extract_shared_handle_from_arrays(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& value) {
            if constexpr (nt::array_decay<U>) {
                return std::forward<U>(value).share();
            } else {
                return Empty{};
            }
        });
    }

    /// Whether accessor indexing is safe using a certain offset precision.
    /// \note Accessors increment the pointer on dimension at a time! So the offset of each dimension
    ///       is converted to ptrdiff_t (by the compiler) and added to the pointer. This makes it less
    ///       likely to reach the integer upper limit.
    template<typename Int, typename T, typename I, usize N>
    [[nodiscard]] constexpr bool is_accessor_access_safe(const Strides<I, N>& input, const Shape<I, N>& shape) {
        for (usize i{}; i < N; ++i) {
            const auto end = static_cast<isize>(shape[i] - 1) * static_cast<isize>(input[i]);
            if (not is_safe_cast<Int>(end))
                return false;
        }
        return true;
    }
    template<typename Int, typename T, typename I, usize N>
        requires (nt::accessor_nd<T, N> or (nt::varray<T> and N == 4 and nt::same_as<I, isize>))
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T& input, const Shape<I, N>& shape) {
        return is_accessor_access_safe<Int>(input.strides_full(), shape);
    }
    template<typename _, typename T, typename I, usize N>
        requires (nt::empty<T> or nt::numeric<T>)
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T&, const Shape<I, N>&) {
        return true;
    }

    template<nt::varray_decay Input, nt::varray_decay Output>
    [[nodiscard]] auto broadcast_strides(
        const Input& input,
        const Output& output,
        std::source_location location = std::source_location::current()
    ) -> Strides<isize, 4> {
        auto input_strides = input.strides();
        if (not broadcast(input.shape(), input_strides, output.shape())) {
            panic_at_location(location, "Cannot broadcast an array of shape {} into an array of shape {}",
                              input.shape(), output.shape());
        }
        return input_strides;
    }

    template<nt::varray_decay Input, nt::varray_decay Output>
    [[nodiscard]] auto broadcast_strides_optional(
        const Input& input,
        const Output& output,
        std::source_location location = std::source_location::current()
    ) -> Strides<isize, 4> {
        auto input_strides = input.strides();
        if (not input.is_empty() and not broadcast(input.shape(), input_strides, output.shape())) {
            panic_at_location(location, "Cannot broadcast an array of shape {} into an array of shape {}",
                              input.shape(), output.shape());
        }
        return input_strides;
    }

    template<bool ALLOW_EMPTY = false, typename T>
    constexpr auto to_batch(const T& value) {
        using value_t = nt::const_value_type_t<T>;
        if constexpr (nt::empty<T>) {
            if constexpr (ALLOW_EMPTY)
                return nd::Batch<Empty>{};
            else
                static_assert(nt::always_false<T>);
        } else if constexpr (nt::varray<T>) {
            NOA_ASSERT(value.are_contiguous());
            return nd::Batch<value_t*>{value.get()};
        } else {
            return nd::Batch{value};
        }
    }

    template<nt::varray_decay... T>
    constexpr bool are_arrays_valid(const T&... inputs) {
        return (not inputs.is_empty() and ...);
    }
}
