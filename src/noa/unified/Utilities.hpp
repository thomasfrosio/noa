#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/utils/ShareHandles.hpp"
#include "noa/unified/Traits.hpp"

namespace noa::guts {
    /// Extracts the accessors from the varrays in the tuple.
    /// For types other than varrays, forward the object into an AccessorValue.
    template<typename T> requires nt::is_tuple_v<T>
    [[nodiscard]] constexpr auto to_tuple_of_accessors(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& v) {
            if constexpr (nt::is_varray_v<U>) {
                return std::forward<U>(v).accessor();
            } else {
                using value_type = std::remove_reference_t<U>; // preserve the constness
                return AccessorValue<value_type>(std::forward<U>(v));
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

    /// Returns the index of the first varray in the tuple.
    /// Returns -1 if there's no varray in the tuple.
    [[nodiscard]] constexpr i64 index_of_first_varray(auto&& tuple) requires nt::is_tuple_v<decltype(tuple)> {
        i64 index{-1};
        tuple.any_enumerate([&]<size_t I>(auto& arg) {
            if constexpr (nt::is_varray_v<decltype(arg)>) {
                index = I;
                return true;
            }
            return false;
        });
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
    [[nodiscard]] auto extract_shared_handle_from_arrays(auto&& tuple) requires nt::is_tuple_v<decltype(tuple)> {
        return tuple.map([](auto&& value) {
            if constexpr (nt::is_array_v<decltype(value)>) {
                return value.share();
            } else {
                return Empty{};
            }
        });
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
