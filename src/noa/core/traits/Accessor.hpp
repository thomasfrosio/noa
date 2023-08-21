#pragma once

#include <type_traits>
#include "noa/core/traits/Utilities.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_accessor : std::false_type {};
    template<typename T> using is_accessor = std::bool_constant<proclaim_is_accessor<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_accessor_v = is_accessor<T>::value;
    template<typename... Ts> constexpr bool are_accessor_v = bool_and<is_accessor_v<Ts>...>::value;

    template<typename T> struct proclaim_is_accessor_restrict : std::false_type {};
    template<typename T> using is_accessor_restrict = std::bool_constant<proclaim_is_accessor_restrict<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_accessor_restrict_v = is_accessor_restrict<T>::value;
    template<typename... Ts> constexpr bool are_accessor_restrict_v = bool_and<is_accessor_restrict_v<Ts>...>::value;

    template<typename T> struct proclaim_is_accessor_contiguous : std::false_type {};
    template<typename T> using is_accessor_contiguous = std::bool_constant<proclaim_is_accessor_contiguous<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_accessor_contiguous_v = is_accessor_contiguous<T>::value;
    template<typename... Ts> constexpr bool are_accessor_contiguous_v = bool_and<is_accessor_contiguous_v<Ts>...>::value;

    template<typename T, size_t N> struct proclaim_is_accessor_nd : std::false_type {};
    template<typename T, size_t N> using is_accessor_nd = std::bool_constant<proclaim_is_accessor_nd<remove_ref_cv_t<T>, N>::value>;
    template<typename T, size_t N> constexpr bool is_accessor_nd_v = is_accessor_nd<T, N>::value;
    template<typename... Ts> constexpr bool are_accessor_nd_v = bool_and<is_accessor_contiguous_v<Ts>...>::value;

    template<typename T> constexpr bool is_accessor_1d_v = is_accessor_nd_v<T, 1>;
    template<typename T> constexpr bool is_accessor_2d_v = is_accessor_nd_v<T, 2>;
    template<typename T> constexpr bool is_accessor_3d_v = is_accessor_nd_v<T, 3>;
    template<typename T> constexpr bool is_accessor_4d_v = is_accessor_nd_v<T, 4>;

    template<typename T> constexpr bool is_accessor_1d_restrict_contiguous_v =
            is_accessor_nd_v<T, 1> && is_accessor_restrict_v<T> && is_accessor_contiguous_v<T>;
}
