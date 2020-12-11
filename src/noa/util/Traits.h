/**
 * @file Traits.h
 * @brief Some type traits.
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * Available type traits:
 *  -# @c is_uint_v                     : (cv qualifiers) uint8_t, uint16_t, uint32_t, uint64_t
 *  -# @c is_int_v                      : (cv qualifiers) (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t
 *  -# @c is_float_v                    : (cv qualifiers) float|double|long double
 *  -# @c is_complex_v                  : (cv qualifiers) std::complex<float|double|long double>
 *  -# @c is_scalar_v                   : is_float_v || is_int_v
 *  -# @c is_data_v                     : is_float_v || is_complex_v
 *  -# @c is_arith_v                    : is_float_v || is_int_v || is_complex_v
 *
 *  -# @c is_bool_v                     : (cv qualifiers) bool
 *  -# @c is_string_v                   : (cv qualifiers) std::string(_view)
 *
 *  -# @c is_std_vector_v               : std::vector
 *  -# @c is_std_vector_bool_v          : std::vector<is_bool_v, A>
 *  -# @c is_std_vector_string_v        : std::vector<is_string_v, A>
 *  -# @c is_std_vector_unsigned_v      : std::vector<is_uint_v, A>
 *  -# @c is_std_vector_int_v           : std::vector<is_int_v, A>
 *  -# @c is_std_vector_float_v         : std::vector<is_float_v, A>
 *  -# @c is_std_vector_complex_v       : std::vector<is_complex_v, A>
 *  -# @c is_std_vector_scalar_v        : std::vector<(is_float_v|is_int_v), A>
 *  -# @c is_std_vector_data_v          : std::vector<(is_float_v|is_complex_v), A>
 *  -# @c is_std_vector_arith_v         : std::vector<(is_float_v|is_complex_v|is_int_v), A>
 *
 *  -# @c is_std_array_v                : std::array
 *  -# @c is_std_array_bool_v           : std::array<is_bool_v, N>
 *  -# @c is_std_array_string_v         : std::array<is_string_v, N>
 *  -# @c is_std_array_unsigned_v       : std::array<is_uint_v, N>
 *  -# @c is_std_array_int_v            : std::array<is_int_v, N>
 *  -# @c is_std_array_float_v          : std::array<is_float_v, N>
 *  -# @c is_std_array_complex_v        : std::array<is_complex_v, N>
 *  -# @c is_std_array_scalar_v         : std::array<(is_float_v|is_int_v), N>
 *  -# @c is_std_array_data_v           : std::array<(is_float_v|is_complex_v), N>
 *  -# @c is_std_array_arith_v          : std::array<(is_float_v|is_complex_v|is_int_v), N>
 *
 *  -# @c is_std_sequence_v             : std::(vector|array)
 *  -# @c is_std_sequence_bool_v        : std::(vector|array)<is_bool_v, X>
 *  -# @c is_std_sequence_string_v      : std::(vector|array)<is_string_v, X>
 *  -# @c is_std_sequence_unsigned_v    : std::(vector|array)<is_uint_v, A>
 *  -# @c is_std_sequence_int_v         : std::(vector|array)<is_int_v, X>
 *  -# @c is_std_sequence_float_v       : std::(vector|array)<is_float_v, X>
 *  -# @c is_std_sequence_complex_v     : std::(vector|array)<is_complex_v, X>
 *  -# @c is_std_sequence_scalar_v      : std::(vector|array)<(is_float_v|is_int_v), X>
 *  -# @c is_std_sequence_data_v        : std::(vector|array)<(is_float_v|is_complex_v), X>
 *  -# @c is_std_sequence_arith_v       : std::(vector|array)<(is_float_v|is_complex_v|is_int_v), X>
 *
 *  -# @c is_std_sequence_of_type_v<T1, V2>         T1 = std::(vector|array)<V1>; check if V1 == V2
 *  -# @c are_std_sequence_of_same_type_v<T1, T2>   T1|T2 = std::(vector|array)<V1|V2>; check if V1 == V2
 *  -# @c is_same_v<T1, T2>                         T1|T2 = (cv) V1|V2(&); check if V1 == V2
 *  -# @c remove_ref_cv<T>                          std::remove_cv_t<std::remove_reference_t<T>>
 */
#pragma once

#include "noa/Base.h"

//@CLION-formatter:off

/** Gathers a bunch of type traits. */
namespace Noa::Traits {
    template<typename T> struct NOA_API remove_ref_cv { using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>; };
    template<typename T> NOA_API using remove_ref_cv_t = typename remove_ref_cv<T>::type;


    template<typename> struct p_is_uint : public std::false_type {};
    template<> struct p_is_uint<uint8_t> : public std::true_type {};
    template<> struct p_is_uint<uint16_t> : public std::true_type {};
    template<> struct p_is_uint<uint32_t> : public std::true_type {};
    template<> struct p_is_uint<uint64_t> : public std::true_type {};
    template<typename T> struct NOA_API is_uint : p_is_uint<remove_ref_cv_t<T>>::type {};

    // One of: uint8_t, uint16_t, uint32_t, uint64_t
    template<typename T> NOA_API inline constexpr bool is_uint_v = is_uint<T>::value;


    template<typename> struct p_is_int : public std::false_type {};
    template<> struct p_is_int<int8_t> : public std::true_type {};
    template<> struct p_is_int<uint8_t> : public std::true_type {};
    template<> struct p_is_int<int16_t> : public std::true_type {};
    template<> struct p_is_int<uint16_t> : public std::true_type {};
    template<> struct p_is_int<int32_t> : public std::true_type {};
    template<> struct p_is_int<uint32_t> : public std::true_type {};
    template<> struct p_is_int<int64_t> : public std::true_type {};
    template<> struct p_is_int<uint64_t> : public std::true_type {};
    template<typename T> struct NOA_API is_int : p_is_int<remove_ref_cv_t<T>>::type {};

    // One of: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t
    template<typename T> NOA_API inline constexpr bool is_int_v = is_int<T>::value;


    template<typename> struct p_is_float : std::false_type {};
    template<> struct p_is_float<float> : std::true_type {};
    template<> struct p_is_float<double> : std::true_type {};
    template<> struct p_is_float<long double> : std::true_type {};
    template<typename T> struct NOA_API is_float : p_is_float<remove_ref_cv_t<T>>::type {};

    // One of: float, double, long double
    template<typename T> NOA_API inline constexpr bool is_float_v = is_float<T>::value;


    template<typename> struct p_is_complex : std::false_type {};
    template<> struct p_is_complex<std::complex<float>> : std::true_type {};
    template<> struct p_is_complex<std::complex<double>> : std::true_type {};
    template<> struct p_is_complex<std::complex<long double>> : std::true_type {};
    template<typename T> struct NOA_API is_complex : p_is_complex<remove_ref_cv_t<T>>::type {};
    template<typename T> NOA_API inline constexpr bool is_complex_v = is_complex<T>::value; // One of: std::complex<float|double|long double>


    template<typename T> struct NOA_API is_scalar { static constexpr const bool value = is_float<T>::value || is_int<T>::value; };
    template<typename T> NOA_API inline constexpr bool is_scalar_v = is_scalar<T>::value; // One of: is_float_v, is_int_v


    template<typename T> struct NOA_API is_data { static constexpr bool value = is_float<T>::value || is_complex<T>::value; };
    template<typename T> NOA_API inline constexpr bool is_data_v = is_data<T>::value; // One of: is_float_v, is_complex_v


    template<typename T> struct NOA_API is_arith { static constexpr bool value = (is_float<T>::value || is_int<T>::value || is_complex<T>::value); };
    template<typename T> NOA_API inline constexpr bool is_arith_v = is_arith<T>::value; // One of: is_int_v, is_float_v, is_complex_v


    template<typename> struct p_is_bool : std::false_type {};
    template<> struct p_is_bool<bool> : std::true_type {};
    template<typename T> struct NOA_API is_bool : p_is_bool<remove_ref_cv_t<T>>::type {};
    template<typename T> NOA_API inline constexpr bool is_bool_v = is_bool<T>::value;


    template<typename> struct p_is_string : std::false_type {};
    template<> struct p_is_string<std::string> : std::true_type {};
    template<> struct p_is_string<std::string_view> : std::true_type {};
    template<typename T> struct NOA_API is_string : p_is_string<remove_ref_cv_t<T>>::type {};
    template<typename T> NOA_API inline constexpr bool is_string_v = is_string<T>::value; // One of: std::string(_view)


    template<typename T> struct p_is_std_vector : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector<std::vector<T, A>> : std::true_type {};
    template<typename T> struct NOA_API is_std_vector { static constexpr bool value = p_is_std_vector<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_v = is_std_vector<T>::value;


    template<typename T> struct p_is_std_vector_uint : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_uint<std::vector<T, A>> { static constexpr bool value = is_uint_v<T>; };
    template<typename T> struct NOA_API is_std_vector_uint { static constexpr bool value = p_is_std_vector_uint<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_uint_v = is_std_vector_uint<T>::value;


    template<typename T> struct p_is_std_vector_int : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_int<std::vector<T, A>> { static constexpr bool value = is_int_v<T>; };
    template<typename T> struct NOA_API is_std_vector_int { static constexpr bool value = p_is_std_vector_int<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_int_v = is_std_vector_int<T>::value;


    template<typename T> struct p_is_std_vector_float : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_float<std::vector<T, A>> { static constexpr bool value = is_float_v<T>; };
    template<typename T> struct NOA_API is_std_vector_float { static constexpr bool value = p_is_std_vector_float<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_float_v = is_std_vector_float<T>::value;


    template<typename T> struct p_is_std_vector_complex : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_complex<std::vector<T, A>> { static constexpr bool value = is_complex_v<T>; };
    template<typename T> struct NOA_API is_std_vector_complex { static constexpr bool value = p_is_std_vector_complex<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_complex_v = is_std_vector_complex<T>::value;


    template<typename T> struct p_is_std_vector_scalar : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_scalar<std::vector<T, A>> { static constexpr bool value = is_int_v<T> || is_float_v<T>; };
    template<typename T> struct NOA_API is_std_vector_scalar { static constexpr bool value = p_is_std_vector_scalar<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_scalar_v = is_std_vector_scalar<T>::value;


    template<typename T> struct p_is_std_vector_data : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_data<std::vector<T, A>> { static constexpr bool value = is_complex_v<T> || is_float_v<T>; };
    template<typename T> struct NOA_API is_std_vector_data { static constexpr bool value = p_is_std_vector_data<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_data_v = is_std_vector_data<T>::value;


    template<typename T> struct p_is_std_vector_arith : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_arith<std::vector<T, A>> { static constexpr bool value = is_int_v<T> || is_float_v<T> || is_complex_v<T>; };
    template<typename T> struct NOA_API is_std_vector_arith { static constexpr bool value = p_is_std_vector_arith<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_arith_v = is_std_vector_arith<T>::value;


    template<typename> struct p_is_std_vector_bool : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_bool<std::vector<T, A>> { static constexpr bool value = is_bool_v<T>; };
    template<typename T> struct NOA_API is_std_vector_bool { static constexpr bool value = p_is_std_vector_bool<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_bool_v = is_std_vector_bool<T>::value;


    template<typename T> struct p_is_std_vector_string : std::false_type {};
    template<typename T, typename A> struct p_is_std_vector_string<std::vector<T, A>> { static constexpr bool value = is_string_v<T>; };
    template<typename T> struct NOA_API is_std_vector_string { static constexpr bool value = p_is_std_vector_string<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_vector_string_v = is_std_vector_string<T>::value;


    template<typename T> struct p_is_std_array : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array<std::array<T, N>> : std::true_type {};
    template<typename T> struct NOA_API is_std_array : p_is_std_array<remove_ref_cv_t<T>>::type {};
    template<typename T> NOA_API inline constexpr bool is_std_array_v = is_std_array<T>::value;


    template<typename T> struct p_is_std_array_uint : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_uint<std::array<T, N>> { static constexpr bool value = is_uint_v<T>; };
    template<typename T> struct NOA_API is_std_array_uint { static constexpr bool value = p_is_std_array_uint<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_uint_v = is_std_array_uint<T>::value;


    template<typename T> struct p_is_std_array_int : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_int<std::array<T, N>> { static constexpr bool value = is_int_v<T>;  };
    template<typename T> struct NOA_API is_std_array_int { static constexpr bool value = p_is_std_array_int<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_int_v = is_std_array_int<T>::value;


    template<typename T> struct p_is_std_array_float : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_float<std::array<T, N>> { static constexpr bool value = is_float_v<T>; };
    template<typename T> struct NOA_API is_std_array_float { static constexpr bool value = p_is_std_array_float<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_float_v = is_std_array_float<T>::value;


    template<typename T> struct p_is_std_array_complex : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_complex<std::array<T, N>> { static constexpr bool value = is_complex_v<T>; };
    template<typename T> struct NOA_API is_std_array_complex { static constexpr bool value = p_is_std_array_complex<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_complex_v = is_std_array_complex<T>::value;


    template<typename T> struct p_is_std_array_scalar : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_scalar<std::array<T, N>> { static constexpr bool value = is_int_v<T> || is_float_v<T>; };
    template<typename T> struct NOA_API is_std_array_scalar { static constexpr bool value = p_is_std_array_scalar<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_scalar_v = is_std_array_scalar<T>::value;


    template<typename T> struct p_is_std_array_data : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_data<std::array<T, N>> { static constexpr bool value = is_complex_v<T> || is_float_v<T>; };
    template<typename T> struct NOA_API is_std_array_data { static constexpr bool value = p_is_std_array_data<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_data_v = is_std_array_data<T>::value;


    template<typename T> struct p_is_std_array_arith : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_arith<std::array<T, N>> { static constexpr bool value = is_complex_v<T> || is_int_v<T> || is_float_v<T>; };
    template<typename T> struct NOA_API is_std_array_arith { static constexpr bool value = p_is_std_array_arith<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_arith_v = is_std_array_arith<T>::value;


    template<typename T> struct p_is_std_array_bool : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_bool<std::array<T, N>> { static constexpr bool value = is_bool_v<T>; };
    template<typename T> struct NOA_API is_std_array_bool { static constexpr bool value = p_is_std_array_bool<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_bool_v = is_std_array_bool<T>::value;


    template<typename T> struct p_is_std_array_string : std::false_type {};
    template<typename T, std::size_t N> struct p_is_std_array_string<std::array<T, N>> { static constexpr bool value = is_string_v<T>; };
    template<typename T> struct NOA_API is_std_array_string { static constexpr bool value = p_is_std_array_string<remove_ref_cv_t<T>>::value; };
    template<typename T> NOA_API inline constexpr bool is_std_array_string_v = is_std_array_string<T>::value;


    template<typename T> struct NOA_API is_std_sequence { static constexpr bool value = (is_std_array_v<T> || is_std_vector_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_v = is_std_sequence<T>::value;


    template<typename T> struct NOA_API is_std_sequence_uint { static constexpr bool value = (is_std_array_uint_v<T> || is_std_vector_uint_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_uint_v = is_std_sequence_uint<T>::value;


    template<typename T> struct NOA_API is_std_sequence_int { static constexpr bool value = (is_std_array_int_v<T> || is_std_vector_int_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_int_v = is_std_sequence_int<T>::value;


    template<typename T> struct NOA_API is_std_sequence_float { static constexpr bool value = (is_std_array_float_v<T> || is_std_vector_float_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_float_v = is_std_sequence_float<T>::value;


    template<typename T> struct NOA_API is_std_sequence_complex { static constexpr bool value = (is_std_array_complex_v<T> || is_std_vector_complex_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_complex_v = is_std_sequence_complex<T>::value;


    template<typename T> struct NOA_API is_std_sequence_scalar { static constexpr bool value = (is_std_array_scalar_v<T> || is_std_vector_scalar_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_scalar_v = is_std_sequence_scalar<T>::value;


    template<typename T> struct NOA_API is_std_sequence_data { static constexpr bool value = (is_std_array_data_v<T> || is_std_vector_data_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_data_v = is_std_sequence_data<T>::value;


    template<typename T> struct NOA_API is_std_sequence_arith { static constexpr bool value = (is_std_array_arith_v<T> || is_std_vector_arith_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_arith_v = is_std_sequence_arith<T>::value;


    template<typename T> struct NOA_API is_std_sequence_bool { static constexpr bool value = (is_std_array_bool_v<T> || is_std_vector_bool_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_bool_v = is_std_sequence_bool<T>::value;


    template<typename T> struct NOA_API is_std_sequence_string { static constexpr bool value = (is_std_array_string_v<T> || is_std_vector_string_v<T>); };
    template<typename T> NOA_API inline constexpr bool is_std_sequence_string_v = is_std_sequence_string<T>::value;
}


//@CLION-formatter:on
namespace Noa::Traits {
    template<typename, typename>
    struct p_is_std_sequence_of_type : std::false_type {
    };
    template<typename V1, typename A, typename V2>
    struct p_is_std_sequence_of_type<std::vector<V1, A>, V2> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, size_t N, typename V2>
    struct p_is_std_sequence_of_type<std::array<V1, N>, V2> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename T, typename V>
    struct NOA_API is_std_sequence_of_type {
        static constexpr bool value = p_is_std_sequence_of_type<remove_ref_cv_t<T>, V>::value;
    };
    template<typename T, typename V> NOA_API inline constexpr bool is_std_sequence_of_type_v = is_std_sequence_of_type<T, V>::value;
}


namespace Noa::Traits {
    template<typename, typename>
    struct p_are_std_sequence_of_same_type : std::false_type {
    };
    template<typename V1, typename V2, typename X>
    struct p_are_std_sequence_of_same_type<std::vector<V1, X>, std::vector<V2, X>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, size_t X>
    struct p_are_std_sequence_of_same_type<std::array<V1, X>, std::array<V2, X>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, typename X1, size_t X2>
    struct p_are_std_sequence_of_same_type<std::vector<V1, X1>, std::array<V2, X2>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename V1, typename V2, size_t X1, typename X2>
    struct p_are_std_sequence_of_same_type<std::array<V1, X1>, std::vector<V2, X2>> {
        static constexpr bool value = std::is_same_v<V1, V2>;
    };
    template<typename T1, typename T2>
    struct NOA_API are_std_sequence_of_same_type {
        static constexpr bool value = p_are_std_sequence_of_same_type<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>::value;
    };
    template<typename T1, typename T2>
    NOA_API inline constexpr bool are_std_sequence_of_same_type_v = are_std_sequence_of_same_type<T1, T2>::value;
}


namespace Noa::Traits {
    template<typename T1, typename T2>
    struct NOA_API is_same {
        static constexpr bool value = std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>;
    };
    template<typename T1, typename T2>
    NOA_API inline constexpr bool is_same_v = is_same<T1, T2>::value;
}


namespace Noa::Traits {
    template<typename T>
    struct NOA_API always_false {
        static constexpr bool value = false;
    };
    template<typename T>
    NOA_API inline constexpr bool always_false_v = always_false<T>::value;
}
