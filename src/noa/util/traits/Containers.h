/**
 * @file Containers.h
 * @brief Some type traits about std containers.
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * Type traits:
 *  -# @c is_std_vector_v               : std::vector
 *  -# @c is_std_vector_bool_v          : std::vector<is_bool_v, A>
 *  -# @c is_std_vector_string_v        : std::vector<is_string_v, A>
 *  -# @c is_std_vector_unsigned_v      : std::vector<is_uint_v, A>
 *  -# @c is_std_vector_int_v           : std::vector<is_int_v, A>
 *  -# @c is_std_vector_float_v         : std::vector<is_float_v, A>
 *  -# @c is_std_vector_complex_v       : std::vector<is_complex_v, A>
 *  -# @c is_std_vector_std_complex_v   : std::vector<is_std_complex_v, A>

 *  -# @c is_std_vector_scalar_v        : std::vector<(is_float_v|is_int_v), A>
 *  -# @c is_std_vector_data_v          : std::vector<(is_float_v|is_int_v|is_complex_v), A>
 *
 *  -# @c is_std_array_v                : std::array
 *  -# @c is_std_array_bool_v           : std::array<is_bool_v, N>
 *  -# @c is_std_array_string_v         : std::array<is_string_v, N>
 *  -# @c is_std_array_unsigned_v       : std::array<is_uint_v, N>
 *  -# @c is_std_array_int_v            : std::array<is_int_v, N>
 *  -# @c is_std_array_float_v          : std::array<is_float_v, N>
 *  -# @c is_std_array_complex_v        : std::array<is_complex_v, N>
 *  -# @c is_std_array_std_complex_v    : std::array<is_std_complex_v, N>
 *  -# @c is_std_array_scalar_v         : std::array<(is_float_v|is_int_v), N>
 *  -# @c is_std_array_data_v           : std::array<(is_float_v|is_int_v|is_complex_v), N>
 *
 *  -# @c is_std_sequence_v             : std::(vector|array)
 *  -# @c is_std_sequence_bool_v        : std::(vector|array)<is_bool_v, X>
 *  -# @c is_std_sequence_string_v      : std::(vector|array)<is_string_v, X>
 *  -# @c is_std_sequence_unsigned_v    : std::(vector|array)<is_uint_v, A>
 *  -# @c is_std_sequence_int_v         : std::(vector|array)<is_int_v, X>
 *  -# @c is_std_sequence_float_v       : std::(vector|array)<is_float_v, X>
 *  -# @c is_std_sequence_complex_v     : std::(vector|array)<is_complex_v, X>
 *  -# @c is_std_sequence_std_complex_v : std::(vector|array)<is_std_complex_v, X>
 *  -# @c is_std_sequence_scalar_v      : std::(vector|array)<(is_float_v|is_int_v), X>
 *  -# @c is_std_sequence_data_v        : std::(vector|array)<(is_float_v|is_int_v|is_complex_v), X>
 *
 *  -# @c is_std_sequence_of_type_v<T1, V2>         T1 = std::(vector|array)<V1>; check if V1 == V2
 *  -# @c are_std_sequence_of_same_type_v<T1, T2>   T1|T2 = std::(vector|array)<V1|V2>; check if V1 == V2
 */
#pragma once

#include <vector>
#include <array>

#include "noa/util/traits/BaseTypes.h"

//@CLION-formatter:off

/** Gathers a bunch of type traits. */
namespace Noa::Traits {
    template<typename T> struct p_is_std_vector : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector<std::vector<T, A>> : std::true_type {};
    template<typename T> using is_std_vector = std::bool_constant<p_is_std_vector<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_v = is_std_vector<T>::value;


    template<typename T> struct p_is_std_vector_uint : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_uint<std::vector<T, A>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_std_vector_uint = std::bool_constant<p_is_std_vector_uint<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_uint_v = is_std_vector_uint<T>::value;


    template<typename T> struct p_is_std_vector_int : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_int<std::vector<T, A>> : std::bool_constant<is_int_v<T>> {};
    template<typename T> using is_std_vector_int = std::bool_constant<p_is_std_vector_int<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_int_v = is_std_vector_int<T>::value;


    template<typename T> struct p_is_std_vector_float : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_float<std::vector<T, A>> : std::bool_constant<is_float_v<T>> {};
    template<typename T> using is_std_vector_float = std::bool_constant<p_is_std_vector_float<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_float_v = is_std_vector_float<T>::value;


    template<typename T> struct p_is_std_vector_complex : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_complex<std::vector<T, A>> : std::bool_constant<is_complex_v<T>> {};
    template<typename T> using is_std_vector_complex = std::bool_constant<p_is_std_vector_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_complex_v = is_std_vector_complex<T>::value;


    template<typename T> struct p_is_std_vector_std_complex : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_std_complex<std::vector<T, A>> : std::bool_constant<is_std_complex_v<T>> {};
    template<typename T> using is_std_vector_std_complex = std::bool_constant<p_is_std_vector_std_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_std_complex_v = is_std_vector_std_complex<T>::value;


    template<typename T> struct p_is_std_vector_scalar : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_scalar<std::vector<T, A>> : std::bool_constant<is_int_v<T> || is_float_v<T>> {};
    template<typename T> using is_std_vector_scalar = std::bool_constant<p_is_std_vector_scalar<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_scalar_v = is_std_vector_scalar<T>::value;


    template<typename T> struct p_is_std_vector_data : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_data<std::vector<T, A>> : std::bool_constant<is_complex_v<T> || is_float_v<T> || is_int_v<T>> {};
    template<typename T> using is_std_vector_data = std::bool_constant<p_is_std_vector_data<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_data_v = is_std_vector_data<T>::value;


    template<typename> struct p_is_std_vector_bool : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_bool<std::vector<T, A>> : std::bool_constant<is_bool_v<T>> {};
    template<typename T> using is_std_vector_bool = std::bool_constant<p_is_std_vector_bool<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_bool_v = is_std_vector_bool<T>::value;


    template<typename T> struct p_is_std_vector_string : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_string<std::vector<T, A>> : std::bool_constant<is_string_v<T>> {};
    template<typename T> using is_std_vector_string = std::bool_constant<p_is_std_vector_string<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_vector_string_v = is_std_vector_string<T>::value;


    template<typename T> struct p_is_std_array : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array<std::array<T, N>> : std::true_type {};
    template<typename T> using is_std_array = std::bool_constant<p_is_std_array<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_v = is_std_array<T>::value;


    template<typename T> struct p_is_std_array_uint : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_uint<std::array<T, N>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_std_array_uint = std::bool_constant<p_is_std_array_uint<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_uint_v = is_std_array_uint<T>::value;


    template<typename T> struct p_is_std_array_int : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_int<std::array<T, N>> : std::bool_constant<is_int_v<T>> {};
    template<typename T> using is_std_array_int = std::bool_constant<p_is_std_array_int<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_int_v = is_std_array_int<T>::value;


    template<typename T> struct p_is_std_array_float : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_float<std::array<T, N>> : std::bool_constant<is_float_v<T>> {};
    template<typename T> using is_std_array_float = std::bool_constant<p_is_std_array_float<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_float_v = is_std_array_float<T>::value;


    template<typename T> struct p_is_std_array_complex : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_complex<std::array<T, N>> : std::bool_constant<is_complex_v<T>> {};
    template<typename T> using is_std_array_complex = std::bool_constant<p_is_std_array_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_complex_v = is_std_array_complex<T>::value;


    template<typename T> struct p_is_std_array_std_complex : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_std_complex<std::array<T, N>> : std::bool_constant<is_std_complex_v<T>> {};
    template<typename T> using is_std_array_std_complex = std::bool_constant<p_is_std_array_std_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_std_complex_v = is_std_array_std_complex<T>::value;


    template<typename T> struct p_is_std_array_scalar : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_scalar<std::array<T, N>> : std::bool_constant<is_float_v<T> || is_int_v<T>> {};
    template<typename T> using is_std_array_scalar = std::bool_constant<p_is_std_array_scalar<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_scalar_v = is_std_array_scalar<T>::value;


    template<typename T> struct p_is_std_array_data : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_data<std::array<T, N>> : std::bool_constant<is_float_v<T> || is_int_v<T> || is_complex_v<T>> {};
    template<typename T> using is_std_array_data = std::bool_constant<p_is_std_array_data<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_data_v = is_std_array_data<T>::value;


    template<typename T> struct p_is_std_array_bool : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_bool<std::array<T, N>> : std::bool_constant<is_bool_v<T>> {};
    template<typename T> using is_std_array_bool = std::bool_constant<p_is_std_array_bool<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_bool_v = is_std_array_bool<T>::value;


    template<typename T> struct p_is_std_array_string : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_string<std::array<T, N>> : std::bool_constant<is_string_v<T>> {};
    template<typename T> using is_std_array_string = std::bool_constant<p_is_std_array_string<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_std_array_string_v = is_std_array_string<T>::value;


    template<typename T> using is_std_sequence = std::bool_constant<is_std_array_v<T> || is_std_vector_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_v = is_std_sequence<T>::value;

    template<typename T> using is_std_sequence_uint = std::bool_constant<is_std_array_uint_v<T> || is_std_vector_uint_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_uint_v = is_std_sequence_uint<T>::value;

    template<typename T> using is_std_sequence_int = std::bool_constant<is_std_array_int_v<T> || is_std_vector_int_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_int_v = is_std_sequence_int<T>::value;

    template<typename T> using is_std_sequence_float = std::bool_constant<is_std_array_float_v<T> || is_std_vector_float_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_float_v = is_std_sequence_float<T>::value;

    template<typename T> using is_std_sequence_std_complex = std::bool_constant<is_std_array_std_complex_v<T> || is_std_vector_std_complex_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_std_complex_v = is_std_sequence_std_complex<T>::value;

    template<typename T> using is_std_sequence_complex = std::bool_constant<is_std_array_complex_v<T> || is_std_vector_complex_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_complex_v = is_std_sequence_complex<T>::value;

    template<typename T> using is_std_sequence_scalar = std::bool_constant<is_std_array_scalar_v<T> || is_std_vector_scalar_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_scalar_v = is_std_sequence_scalar<T>::value;

    template<typename T> using is_std_sequence_data = std::bool_constant<is_std_array_data_v<T> || is_std_vector_data_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_data_v = is_std_sequence_data<T>::value;

    template<typename T> using is_std_sequence_bool = std::bool_constant<is_std_array_bool_v<T> || is_std_vector_bool_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_bool_v = is_std_sequence_bool<T>::value;

    template<typename T> using is_std_sequence_string = std::bool_constant<is_std_array_string_v<T> || is_std_vector_string_v<T>>;
    template<typename T> inline constexpr bool is_std_sequence_string_v = is_std_sequence_string<T>::value;


    //
    template<typename, typename>
    struct p_is_std_sequence_of_type : std::false_type {};

    template<typename V1, typename A, typename V2>
    struct p_is_std_sequence_of_type<std::vector<V1, A>, V2>
            : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename V1, size_t N, typename V2>
    struct p_is_std_sequence_of_type<std::array<V1, N>, V2>
            : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename T, typename V>
    using is_std_sequence_of_type = std::bool_constant<p_is_std_sequence_of_type<remove_ref_cv_t < T>, V>::value>;

    template<typename T, typename V> inline constexpr bool is_std_sequence_of_type_v = is_std_sequence_of_type<T, V>::value;


    //
    template<typename, typename>
    struct p_are_std_sequence_of_same_type : std::false_type {};

    template<typename V1, typename V2, typename X>
    struct p_are_std_sequence_of_same_type<std::vector<V1, X>, std::vector<V2, X>> : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename V1, typename V2, size_t X>
    struct p_are_std_sequence_of_same_type<std::array<V1, X>, std::array<V2, X>> : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename V1, typename V2, typename X1, size_t X2>
    struct p_are_std_sequence_of_same_type<std::vector<V1, X1>, std::array<V2, X2>> : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename V1, typename V2, size_t X1, typename X2>
    struct p_are_std_sequence_of_same_type<std::array<V1, X1>, std::vector<V2, X2>> : std::bool_constant<std::is_same_v<V1, V2>> {};

    template<typename T1, typename T2>
    using are_std_sequence_of_same_type = std::bool_constant<p_are_std_sequence_of_same_type<remove_ref_cv_t < T1>, remove_ref_cv_t<T2>>::value>;

    template<typename T1, typename T2>
    inline constexpr bool are_std_sequence_of_same_type_v = are_std_sequence_of_same_type<T1, T2>::value;
}

//@CLION-formatter:on
