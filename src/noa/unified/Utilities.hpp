#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/utils/ShareHandles.hpp"
#include "noa/unified/Traits.hpp"

namespace noa::guts {
    /// Extracts the accessors from the varrays in the tuple.
    /// For types other than varrays, forward the object into an AccessorValue.
    template<bool EnforceConst = false, typename T> requires nt::is_tuple_v<T>
    [[nodiscard]] constexpr auto to_tuple_of_accessors(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& v) {
            if constexpr (nt::is_varray_v<U>) {
                if constexpr (EnforceConst) {
                    using value_type = nt::mutable_value_type_t<U>;
                    return std::forward<U>(v).template accessor<const value_type>();
                } else {
                    return std::forward<U>(v).accessor();
                }
            } else {
                if constexpr (EnforceConst) {
                    return AccessorValue<const std::decay_t<U>>(std::forward<U>(v));
                } else {
                    using value_type = std::remove_reference_t<U>; // preserve the constness
                    return AccessorValue<value_type>(std::forward<U>(v));
                }
            }
        });
    }

    /// Reorders the tuple(s) of accessors (in-place).
    template<typename Index, typename... T> requires nt::are_tuple_of_accessor_or_empty_v<T...>
    constexpr void reorder_accessors(const Vec4<Index>& order, T&... accessors) {
        (accessors.for_each([&order](auto& accessor) {
            if constexpr (nt::is_accessor_pure_v<decltype(accessor)>)
                accessor.reorder(order);
        }), ...);
    }

    template<typename Tup>
    [[nodiscard]] constexpr bool are_all_varrays() {
        return []<typename... T>(nt::TypeList<T...>) {
            return nt::are_varray_v<T...>;
        }(nt::type_list_t<Tup>{});
    };

    template<typename Tup>
    [[nodiscard]] constexpr bool are_all_value_types_trivially_copyable() {
        return []<typename... T>(nt::TypeList<T...>) {
            return (std::is_trivially_copyable_v<nt::value_type_t<T>> and ...);
        }(nt::type_list_t<Tup>{});
    };

    /// Returns the index of the first varray in the tuple.
    /// Returns -1 if there's no varray in the tuple.
    template<typename T> requires nt::is_tuple_v<T>
    [[nodiscard]] constexpr i64 index_of_first_varray() {
        i64 index{-1};
        auto is_varray = [&]<size_t I>() {
            if constexpr (nt::is_varray_v<decltype(std::declval<T>()[Tag<I>{}])>) {
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
    template<typename T>
    [[nodiscard]] auto extract_shared_handle(T&& tuple) requires nt::is_tuple_v<T> {
        return std::forward<T>(tuple).map([]<typename U>(U&& value) {
            if constexpr (nt::is_shareable_v<U>)
                return std::forward<U>(value);
            else if constexpr (nt::has_share_v<U>)
                return std::forward<U>(value).share();
            else
                return Empty{};
        });
    }

    /// Extracts the shared handles from Arrays.
    template<typename T>
    [[nodiscard]] auto extract_shared_handle_from_arrays(T&& tuple) requires nt::is_tuple_v<T> {
        return std::forward<T>(tuple).map([]<typename U>(U&& value) {
            if constexpr (nt::is_array_v<U>) {
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
    template<typename Int, typename I, size_t N>
    [[nodiscard]] constexpr bool is_accessor_access_safe(const Strides<I, N>& strides, const Shape<I, N>& shape) {
        for (size_t i = 0; i < N; ++i) {
            const auto end = static_cast<i64>(shape[i] - 1) * static_cast<i64>(strides[i]);
            if (not is_safe_cast<Int>(end))
                return false;
        }
        return true;
    }
    template<typename Int, typename A, typename I, size_t N> requires nt::is_accessor_nd_v<A, N>
    [[nodiscard]] constexpr bool is_accessor_access_safe(const A& accessor, const Shape<I, N>& shape) {
        for (size_t i = 0; i < N; ++i) {
            const auto end = static_cast<i64>(shape[i] - 1) * static_cast<i64>(accessor.stride(i));
            if (not is_safe_cast<Int>(end))
                return false;
        }
        return true;
    }
    template<typename Int, typename T> requires (nt::is_varray_v<T> or nt::is_texture_v<T>)
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T& array, const Shape4<i64>& shape) {
        return ng::is_accessor_access_safe<i32>(array.strides(), shape);
    }
    template<typename Int, typename T, typename I, size_t N> requires (std::is_empty_v<T> or nt::is_numeric_v<T>)
    [[nodiscard]] constexpr bool is_accessor_access_safe(const T&, const Shape<I, N>&) {
        return true;
    }

    /// Filters the input tuple by remove non-varrays and forwards
    /// the varrays (i.e. store references) into the new tuple.
    template<typename... Ts>
    [[nodiscard]] constexpr auto filter_and_forward_varrays(const Tuple<Ts&&...>& tuple) {
        constexpr auto predicate = []<typename T>() { return nt::is_varray_v<T>; };
        constexpr auto transform = []<typename T>(T&& arg) { return forward_as_tuple(std::forward<T>(arg)); };
        return tuple_filter<decltype(predicate), decltype(transform)>(tuple);
    }

    template<typename... Ts, typename Op>
    [[nodiscard]] constexpr bool are_all_equals(Tuple<Ts&&...> const& varrays, Op&& op) {
        return varrays.apply([&]<typename... Args>(const Args&... args) {
            if constexpr (sizeof...(Args) <= 1) {
                return true;
            } else {
                return [&](auto const& first, auto const& ... rest) {
                    return (op(first, rest) && ...);
                }(args...);
            }
        });
    }
}
#endif
