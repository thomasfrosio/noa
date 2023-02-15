#pragma once

#include <type_traits>
#include "noa/core/traits/Utilities.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_vec : std::false_type {};
    template<typename T> using is_vec = std::bool_constant<proclaim_is_vec<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_vec_v = is_vec<T>::value;
    template<typename T> constexpr bool is_vecX_v = is_vec<T>::value;

    template<typename, typename> struct proclaim_is_vec_of_type : std::false_type {};
    template<typename T, typename V> using is_vec_of_type = std::bool_constant<proclaim_is_vec_of_type<remove_ref_cv_t<T>, V>::value>;
    template<typename T, typename V> constexpr bool is_vec_of_type_v = is_vec_of_type<T, V>::value;
    template<typename T, typename V> constexpr bool is_vecT_v = is_vec_of_type<T, V>::value;

    template<typename, size_t> struct proclaim_is_vec_of_size : std::false_type {};
    template<typename T, size_t N> using is_vec_of_size = std::bool_constant<proclaim_is_vec_of_size<remove_ref_cv_t<T>, N>::value>;
    template<typename T, size_t N> constexpr bool is_vec_of_size_v = is_vec_of_size<T, N>::value;
    template<typename T, size_t N> constexpr bool is_vecN_v = is_vec_of_size<T, N>::value;

    template<typename T> constexpr bool is_vec1_v = is_vecN_v<T, 1>;
    template<typename T> constexpr bool is_vec2_v = is_vecN_v<T, 2>;
    template<typename T> constexpr bool is_vec3_v = is_vecN_v<T, 3>;
    template<typename T> constexpr bool is_vec4_v = is_vecN_v<T, 4>;

    // Vector of integers:
    template<typename T> constexpr bool is_intX_v = is_vecX_v<T> && is_int_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_intN_v = is_vecN_v<T, N> && is_int_v<value_type_t<T>>;
    template<typename T> constexpr bool is_int1_v = is_intN_v<T, 1>;
    template<typename T> constexpr bool is_int2_v = is_intN_v<T, 2>;
    template<typename T> constexpr bool is_int3_v = is_intN_v<T, 3>;
    template<typename T> constexpr bool is_int4_v = is_intN_v<T, 4>;

    // Vector of bool:
    template<typename T> constexpr bool is_boolX_v = is_vecX_v<T> && is_bool_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_boolN_v = is_vecN_v<T, N> && is_bool_v<value_type_t<T>>;
    template<typename T> constexpr bool is_bool1_v = is_boolN_v<T, 1>;
    template<typename T> constexpr bool is_bool2_v = is_boolN_v<T, 2>;
    template<typename T> constexpr bool is_bool3_v = is_boolN_v<T, 3>;
    template<typename T> constexpr bool is_bool4_v = is_boolN_v<T, 4>;

    // Vector of signed integers:
    template<typename T> constexpr bool is_sintX_v = is_vecX_v<T> && is_sint_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_sintN_v = is_vecN_v<T, N> && is_sint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_sint1_v = is_sintN_v<T, 1>;
    template<typename T> constexpr bool is_sint2_v = is_sintN_v<T, 2>;
    template<typename T> constexpr bool is_sint3_v = is_sintN_v<T, 3>;
    template<typename T> constexpr bool is_sint4_v = is_sintN_v<T, 4>;

    // Vector of unsigned integers:
    template<typename T> constexpr bool is_uintX_v = is_vecX_v<T> && is_uint_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_uintN_v = is_vecN_v<T, N> && is_uint_v<value_type_t<T>>;
    template<typename T> constexpr bool is_uint1_v = is_uintN_v<T, 1>;
    template<typename T> constexpr bool is_uint2_v = is_uintN_v<T, 2>;
    template<typename T> constexpr bool is_uint3_v = is_uintN_v<T, 3>;
    template<typename T> constexpr bool is_uint4_v = is_uintN_v<T, 4>;

    // Vector of real (= floating-points):
    template<typename T> constexpr bool is_realX_v = is_vecX_v<T> && is_real_v<value_type_t<T>>;
    template<typename T, size_t N> constexpr bool is_realN_v = is_vecN_v<T, N> && is_real_v<value_type_t<T>>;
    template<typename T> constexpr bool is_real1_v = is_realN_v<T, 1>;
    template<typename T> constexpr bool is_real2_v = is_realN_v<T, 2>;
    template<typename T> constexpr bool is_real3_v = is_realN_v<T, 3>;
    template<typename T> constexpr bool is_real4_v = is_realN_v<T, 4>;
}
