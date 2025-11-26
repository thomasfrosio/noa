#pragma once

#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/utils/ShareHandles.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/unified/Traits.hpp"

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
    constexpr void reorder_accessors(const Vec4<Index>& order, T&... accessors) {
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
    [[nodiscard]] constexpr i64 index_of_first_varray() {
        i64 index{-1};
        auto is_varray = [&]<size_t I>() {
            if constexpr (nt::varray_decay<decltype(std::declval<T>()[Tag<I>{}])>) {
                index = I;
                return true;
            }
            return false;
        };
        [&is_varray]<size_t... I>(std::index_sequence<I...>) {
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
    template<typename Int, typename T, typename I, size_t N>
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T& input, const Shape<I, N>& shape) {
        if constexpr (nt::same_as<Strides<I, N>, T>) {
            for (size_t i{}; i < N; ++i) {
                const auto end = static_cast<i64>(shape[i] - 1) * static_cast<i64>(input[i]);
                if (not is_safe_cast<Int>(end))
                    return false;
            }
            return true;
        } else if constexpr (nt::accessor_nd<T, N> or (nt::varray_or_texture<T> and N == 4 and nt::same_as<I, i64>)) {
            return is_accessor_access_safe<Int>(input.strides_full(), shape);
        } else if constexpr (nt::empty<T> or nt::numeric<T>) {
            return true;
        } else {
            static_assert(nt::always_false<T>);
        }
    }

    template<nt::varray_decay Input, nt::varray_decay Output>
    [[nodiscard]] auto broadcast_strides(
        const Input& input,
        const Output& output,
        std::source_location location = std::source_location::current()
    ) -> Strides<i64, 4> {
        auto input_strides = input.strides();
        if (not ni::broadcast(input.shape(), input_strides, output.shape())) {
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
    ) -> Strides<i64, 4> {
        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic_at_location(location, "Cannot broadcast an array of shape {} into an array of shape {}",
                              input.shape(), output.shape());
        }
        return input_strides;
    }

    template<bool ALLOW_EMPTY = false, bool ENFORCE_EMPTY = false, typename Xform>
    auto to_batched_transform(const Xform& xform) {
        using value_t = nt::const_value_type_t<Xform>;
        if constexpr (nt::mat<Xform> or nt::vec<Xform> or nt::quaternion<Xform> or (ALLOW_EMPTY and nt::empty<Xform>)) {
            if constexpr (ENFORCE_EMPTY)
                return BatchedParameter<Empty>{};
            else
                return BatchedParameter{xform};
        } else if constexpr (nt::varray<Xform> and (nt::mat<value_t> or nt::vec<value_t> or nt::quaternion<value_t>)) {
            if constexpr (ENFORCE_EMPTY)
                return BatchedParameter<Empty>{};
            else {
                NOA_ASSERT(xform.are_contiguous());
                return BatchedParameter<value_t*>{xform.get()};
            }
        } else {
            static_assert(nt::always_false<Xform>);
        }
    }

    template<bool ALLOW_EMPTY = false, typename T>
    constexpr auto to_batched_parameter(const T& value) {
        using value_t = nt::const_value_type_t<T>;
        if constexpr (nt::empty<T>) {
            if constexpr (ALLOW_EMPTY)
                return BatchedParameter<Empty>{};
            else
                static_assert(nt::always_false<T>);
        } else if constexpr (nt::varray<T>) {
            NOA_ASSERT(value.are_contiguous());
            return BatchedParameter<value_t*>{value.get()};
        } else {
            return BatchedParameter{value};
        }
    }
}

// /// Filters the input tuple by removing non-varrays and
// /// forwards the varrays (i.e. store references) into the new tuple.
// template<typename... Ts>
// [[nodiscard]] constexpr auto filter_and_forward_varrays(const Tuple<Ts&&...>& tuple) {
//     constexpr auto predicate = []<typename T>() { return nt::varray_decay<T>; };
//     constexpr auto transform = []<typename T>(T&& arg) { return forward_as_tuple(std::forward<T>(arg)); };
//     return tuple_filter<decltype(predicate), decltype(transform)>(tuple);
// }
//
// template<typename... Ts, typename Op>
// [[nodiscard]] constexpr bool are_all_equals(Tuple<Ts&&...> const& varrays, Op&& op) {
//     return varrays.apply([&]<typename... Args>(const Args&... args) {
//         if constexpr (sizeof...(Args) <= 1) {
//             return true;
//         } else {
//             return [&](auto const& first, auto const&... rest) {
//                 return (op(first, rest) and ...);
//             }(args...);
//         }
//     });
// }
