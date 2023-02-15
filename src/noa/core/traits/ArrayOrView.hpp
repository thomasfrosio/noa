#pragma once

#include "noa/core/traits/Utilities.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_array : std::false_type {};
    template<typename T> using is_array = std::bool_constant<proclaim_is_array<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_array_v = is_array<T>::value;
    template<typename... Ts> constexpr bool are_array_v = bool_and<is_array_v<Ts>...>::value;

    template<typename T> struct proclaim_is_view : std::false_type {};
    template<typename T> using is_view = std::bool_constant<proclaim_is_view<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_view_v = is_view<T>::value;
    template<typename... Ts> constexpr bool are_view_v = bool_and<is_view_v<Ts>...>::value;

    template<typename T> using is_array_or_view = std::bool_constant<is_array_v<T> || is_view_v<T>>;
    template<typename T> constexpr bool is_array_or_view_v = is_array_or_view<T>::value;
    template<typename... Ts> constexpr bool are_array_or_view_v = bool_and<is_array_or_view_v<Ts>...>::value;

    template<typename T> constexpr bool is_array_or_view_of_real_v = is_array_or_view<T>::value && is_real_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_complex_v = is_array_or_view<T>::value && is_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_real_or_complex_v = is_array_or_view<T>::value && is_real_or_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_numeric_v = is_array_or_view<T>::value && is_numeric_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_scalar_v = is_array_or_view<T>::value && is_scalar_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_restricted_int_v = is_array_or_view<T>::value && is_restricted_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_restricted_scalar_v = is_array_or_view<T>::value && is_restricted_scalar_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_restricted_numeric_v = is_array_or_view<T>::value && is_restricted_numeric_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_int_v = is_array_or_view<T>::value && is_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_sint_v = is_array_or_view<T>::value && is_sint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_array_or_view_of_uint_v = is_array_or_view<T>::value && is_uint_v<value_type_t<T>>;

    template<typename... Ts> constexpr bool are_array_or_view_of_real_v = bool_and<is_array_or_view_of_real_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_complex_v = bool_and<is_array_or_view_of_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_real_or_complex_v = bool_and<is_array_or_view_of_real_or_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_numeric_v = bool_and<is_array_or_view_of_numeric_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_scalar_v = bool_and<is_array_or_view_of_scalar_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_restricted_int_v = bool_and<is_array_or_view_of_restricted_int_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_restricted_scalar_v = bool_and<is_array_or_view_of_restricted_scalar_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_restricted_numeric_v = bool_and<is_array_or_view_of_restricted_numeric_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_int_v = bool_and<is_array_or_view_of_int_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_sint_v = bool_and<is_array_or_view_of_sint_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_array_or_view_of_uint_v = bool_and<is_array_or_view_of_uint_v<Ts>...>::value;

    template<typename T, typename... Ts> struct have_almost_same_value_type : std::bool_constant<(is_almost_same_v<value_type_t<T>, value_type_t<Ts>> && ...)> {};
    template<typename T, typename... Ts> constexpr bool have_almost_same_value_type_v = have_almost_same_value_type<T, Ts...>::value;
}
