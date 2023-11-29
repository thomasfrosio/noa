#pragma once

#include "noa/core/Traits.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_array : std::false_type {};
    template<typename T> using is_array = std::bool_constant<proclaim_is_array<T>::value>;
    template<typename T> constexpr bool is_array_v = is_array<std::decay_t<T>>::value;
    template<typename... Ts> constexpr bool are_array_v = bool_and<is_array_v<Ts>...>::value;

    template<typename T> struct proclaim_is_view : std::false_type {};
    template<typename T> using is_view = std::bool_constant<proclaim_is_view<T>::value>;
    template<typename T> constexpr bool is_view_v = is_view<std::decay_t<T>>::value;
    template<typename... Ts> constexpr bool are_view_v = bool_and<is_view_v<Ts>...>::value;

    template<typename T> using is_varray = std::bool_constant<is_array<T>::value || is_view<T>::value>;
    template<typename T> constexpr bool is_varray_v = is_varray<std::decay_t<T>>::value;
    template<typename... Ts> constexpr bool are_varray_v = bool_and<is_varray_v<Ts>...>::value;

    template<typename T> constexpr bool is_varray_of_real_v = is_varray_v<T> && is_real_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_complex_v = is_varray_v<T> && is_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_real_or_complex_v = is_varray_v<T> && is_real_or_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_numeric_v = is_varray_v<T> && is_numeric_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_scalar_v = is_varray_v<T> && is_scalar_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_int_v = is_varray_v<T> && is_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_sint_v = is_varray_v<T> && is_sint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_varray_of_uint_v = is_varray_v<T> && is_uint_v<value_type_t<T>>;

    template<typename... Ts> constexpr bool are_varray_of_real_v = bool_and<is_varray_of_real_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_complex_v = bool_and<is_varray_of_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_real_or_complex_v = bool_and<is_varray_of_real_or_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_numeric_v = bool_and<is_varray_of_numeric_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_scalar_v = bool_and<is_varray_of_scalar_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_int_v = bool_and<is_varray_of_int_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_sint_v = bool_and<is_varray_of_sint_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_varray_of_uint_v = bool_and<is_varray_of_uint_v<Ts>...>::value;

    template<typename T, typename... Ts> constexpr bool is_varray_of_any_v = is_varray_v<T> && is_any_v<value_type_t<T>, Ts...>;
    template<typename T, typename... Ts> constexpr bool is_varray_of_almost_any_v = is_varray_v<T> && is_almost_any_v<value_type_t<T>, Ts...>;
}

namespace noa::traits {
    template<typename T> struct proclaim_is_texture : std::false_type {};
    template<typename T> using is_texture = std::bool_constant<proclaim_is_texture<T>::value>;
    template<typename T> constexpr bool is_texture_v = is_texture<std::decay_t<T>>::value;
    template<typename... Ts> constexpr bool are_texture_v = bool_and<is_texture_v<Ts>...>::value;

    template<typename T> constexpr bool is_texture_of_real_v = is_texture_v<T> && is_real_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_complex_v = is_texture_v<T> && is_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_real_or_complex_v = is_texture_v<T> && is_real_or_complex_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_numeric_v = is_texture_v<T> && is_numeric_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_scalar_v = is_texture_v<T> && is_scalar_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_int_v = is_texture_v<T> && is_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_sint_v = is_texture_v<T> && is_sint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_texture_of_uint_v = is_texture_v<T> && is_uint_v<value_type_t<T>>;

    template<typename... Ts> constexpr bool are_texture_of_real_v = bool_and<is_texture_of_real_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_complex_v = bool_and<is_texture_of_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_real_or_complex_v = bool_and<is_texture_of_real_or_complex_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_numeric_v = bool_and<is_texture_of_numeric_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_scalar_v = bool_and<is_texture_of_scalar_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_int_v = bool_and<is_texture_of_int_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_sint_v = bool_and<is_texture_of_sint_v<Ts>...>::value;
    template<typename... Ts> constexpr bool are_texture_of_uint_v = bool_and<is_texture_of_uint_v<Ts>...>::value;

    template<typename T, typename... Ts> constexpr bool is_texture_of_any_v = is_texture_v<T> && is_any_v<value_type_t<T>, Ts...>;
    template<typename T, typename... Ts> constexpr bool is_texture_of_almost_any_v = is_texture_v<T> && is_almost_any_v<value_type_t<T>, Ts...>;
}
