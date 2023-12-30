#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/types/Tuple.hpp"
#include "noa/unified/Traits.hpp"
#include "noa/unified/Device.hpp"

namespace noa::guts {
    /// Filters the input tuple by remove non-varrays and forwards
    /// the varrays (i.e. store references) into the new tuple.
    template<typename... Ts>
    [[nodiscard]] constexpr auto filter_and_forward_varrays(const Tuple<Ts&&...>& tuple) {
        constexpr auto predicate = []<typename T>() { return nt::is_varray_v<T>; };
        constexpr auto transform = []<typename T>(T&& arg) { return forward_as_tuple(std::forward<T>(arg)); };
        return tuple_filter<decltype(predicate), decltype(transform)>(tuple);
    }

    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr auto forward_as_tuple_of_shapes(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([]<typename T>(const T& varray) -> decltype(auto) {
            return varray.shape();
        });
    }

    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr auto make_tuple_of_shapes(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_shapes(varrays).decay();
    }

    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr auto forward_as_tuple_of_strides(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([]<typename T>(const T& varray) -> decltype(auto) {
            return varray.strides();
        });
    }

    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr auto make_tuple_of_strides(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_strides(varrays).decay();
    }

    template<typename... Varrays>
    [[nodiscard]] constexpr auto forward_as_tuple_of_devices(const Tuple<Varrays&&...>& varrays) {
        return varrays.map([]<typename T>(const T& varray) -> decltype(auto) {
            return varray.device();
        });
    }
    template<typename... Varrays>
    [[nodiscard]] constexpr auto make_tuple_of_devices(const Tuple<Varrays&&...>& varrays) {
        return forward_as_tuple_of_devices(varrays).decay();
    }

    /// Extracts the accessors from the arguments in the tuple.
    /// If the argument does not have a .accessor() member function, forward the object into an AccessorValue.
    template<typename T> requires nt::is_tuple_v<T>
    [[nodiscard]] constexpr auto to_tuple_of_accessors(T&& tuple) {
        return std::forward<T>(tuple).map([]<typename U>(U&& v) {
            if constexpr (requires { v.accessor(); }) {
                return v.accessor();
            } else {
                return AccessorValue(std::forward<U>(v));
            }
        });
    }

    /// Extracts the std::shared_ptr from Arrays
    template<typename... Ts>
    auto extract_shared_handles(const Tuple<Ts&&...> tuple) {
        return tuple.map([](const auto& value) {
            if constexpr (requires { value.share(); }) {
                return value.share();
            } else {
                return Empty{};
            }
        });
        // constexpr auto predicate = []<typename T>() { return requires(T x) { x.share(); }; };
        // constexpr auto transform = []<typename T>(T&& v) { return Tuple{std::forward<T>(v).share()}; };
        // return tuple_filter<decltype(predicate), decltype(transform)>(forward_as_tuple(std::forward<Ts>(s)...));
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

    /// Whether the varrays are all on the same device.
    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr bool are_all_same_device(Tuple<Varrays&&...> const& varrays) {
        return are_all_equals(varrays, [](const auto& lhs, const auto& rhs) {
            return lhs.device() == rhs.device();
        });
    }

    /// Whether the varrays have all the same shape.
    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr bool are_all_same_shape(Tuple<Varrays&&...> const& varrays) {
        return are_all_equals(varrays, [](const auto& lhs, const auto& rhs) {
            return all(lhs.shape() == rhs.shape());
        });
    }

    /// Whether the varrays have all the same dimension order.
    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr bool are_all_same_order(Tuple<Varrays&&...> const& varrays) {
        return are_all_equals(varrays, [](const auto& lhs, const auto& rhs) {
            return all(ni::order(lhs.strides(), lhs.shape()) == ni::order(rhs.strides(), rhs.shape()));
        });
    }
    template<typename... Varrays> requires nt::are_varray_v<Varrays...>
    [[nodiscard]] constexpr bool are_all_same_order(Tuple<Varrays&&...> const& varrays, const Vec4<i64>& order) {
        return varrays.all([&](const auto& varray) {
            return all(order == ni::order(varray.strides(), varray.shape()));
        });
    }

    /// Reorders the tuple(s) of accessors (in-place).
    template<typename Index, typename... T> requires nt::are_tuple_of_accessor_v<T...>
    constexpr void reorder_accessors(const Vec4<Index>& order, T&... accessors) {
        (accessors.for_each([&order](auto& accessor) {
            if constexpr (!nt::is_accessor_reference_v<decltype(accessor)>)
                accessor.reorder(order);
        }), ...);
    }

    /// Returns the index of the first varray in the tuple.
    /// Returns -1 if there's no varray in the tuple.
    template<typename T> requires nt::is_tuple_v<T>
    [[nodiscard]] constexpr auto index_of_first_varray(const T& tuple) -> i64 {
        i64 index{-1};
        tuple.any_enumerate([&]<size_t I>(auto&& arg) {
            if (nt::is_varray_v<decltype(arg)>) {
                index = I;
                return true;
            }
            return false;
        });
        return index;
    }
}
#endif