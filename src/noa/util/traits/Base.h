/**
 * @file traits/Base.h
 * @brief Some type traits.
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * Type traits:
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
 *  -# @c remove_ref_cv<T>              : std::remove_cv_t<std::remove_reference_t<T>>
 *  -# @c is_same_v<T1, T2>             : T1|T2 = (cv) V1|V2(&); check if V1 == V2
 *  -# @c is_scoped_enum_v              : enum class|struct
 *  -# @c is_always_false_v             : false
 */
#pragma once

#include <type_traits>
#include <string>
#include <string_view>
#include <complex>

#include "noa/API.h"

//@CLION-formatter:off

namespace Noa::Traits {

    template<typename T> using remove_ref_cv = std::integral_constant<typename std::remove_cv_t<typename std::remove_reference_t<T>>, true>;
    template<typename T> NOA_API using remove_ref_cv_t = typename remove_ref_cv<T>::type;


    template<typename> struct p_is_uint : std::false_type {};
    template<> struct p_is_uint<uint8_t> : std::true_type {};
    template<> struct p_is_uint<uint16_t> : std::true_type {};
    template<> struct p_is_uint<uint32_t> : std::true_type {};
    template<> struct p_is_uint<uint64_t> : std::true_type {};
    template<typename T> using is_uint = std::bool_constant<p_is_uint<remove_ref_cv_t<T>>::value>;

    // One of: uint8_t, uint16_t, uint32_t, uint64_t
    template<typename T> NOA_API inline constexpr bool is_uint_v = is_uint<T>::value;


    template<typename> struct p_is_int : std::false_type {};
    template<> struct p_is_int<int8_t> : std::true_type {};
    template<> struct p_is_int<uint8_t> : std::true_type {};
    template<> struct p_is_int<int16_t> : std::true_type {};
    template<> struct p_is_int<uint16_t> : std::true_type {};
    template<> struct p_is_int<int32_t> : std::true_type {};
    template<> struct p_is_int<uint32_t> : std::true_type {};
    template<> struct p_is_int<int64_t> : std::true_type {};
    template<> struct p_is_int<uint64_t> : std::true_type {};
    template<typename T> using is_int = std::bool_constant<p_is_int<remove_ref_cv_t<T>>::value>;

    // One of: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t
    template<typename T> NOA_API constexpr bool is_int_v = is_int<T>::value;


    template<typename> struct p_is_float : std::false_type {};
    template<> struct p_is_float<float> : std::true_type {};
    template<> struct p_is_float<double> : std::true_type {};
    template<> struct p_is_float<long double> : std::true_type {};
    template<typename T> using is_float = std::bool_constant<p_is_float<remove_ref_cv_t<T>>::value>;

    // One of: float, double, long double
    template<typename T> NOA_API constexpr bool is_float_v = is_float<T>::value;


    template<typename> struct p_is_complex : std::false_type {};
    template<> struct p_is_complex<std::complex<float>> : std::true_type {};
    template<> struct p_is_complex<std::complex<double>> : std::true_type {};
    template<> struct p_is_complex<std::complex<long double>> : std::true_type {};
    template<typename T> using is_complex = std::bool_constant<p_is_complex<remove_ref_cv_t<T>>::value>;

    // One of: std::complex<float|double|long double>
    template<typename T> NOA_API constexpr bool is_complex_v = is_complex<T>::value;


    template<typename T> using is_scalar = std::bool_constant<is_float<T>::value || is_int<T>::value>;
    template<typename T> NOA_API constexpr bool is_scalar_v = is_scalar<T>::value; // One of: is_float_v, is_int_v


    template<typename T> using is_data = std::bool_constant<is_float<T>::value || is_complex<T>::value>;
    template<typename T> NOA_API constexpr bool is_data_v = is_data<T>::value; // One of: is_float_v, is_complex_v


    template<typename T> using is_arith = std::bool_constant<is_int<T>::value || is_float<T>::value || is_complex<T>::value>;
    template<typename T> NOA_API constexpr bool is_arith_v = is_arith<T>::value; // One of: is_int_v, is_float_v, is_complex_v


    template<typename> struct p_is_bool : std::false_type {};
    template<> struct p_is_bool<bool> : std::true_type {};
    template<typename T> using is_bool = std::bool_constant<p_is_bool<remove_ref_cv_t<T>>::value>;
    template<typename T> NOA_API constexpr bool is_bool_v = is_bool<T>::value;


    template<typename> struct p_is_string : std::false_type {};
    template<> struct p_is_string<std::string> : std::true_type {};
    template<> struct p_is_string<std::string_view> : std::true_type {};
    template<typename T> using is_string = std::bool_constant<p_is_string<remove_ref_cv_t<T>>::value>;
    template<typename T> NOA_API constexpr bool is_string_v = is_string<T>::value; // One of: std::string(_view)


    template<typename E> using is_scoped_enum = std::bool_constant<std::is_enum_v<E> && !std::is_convertible_v<E, int>>;
    template<typename E> constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value; // An enum class


    template<typename T1, typename T2> using is_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;
    template<typename T1, typename T2> NOA_API inline constexpr bool is_same_v = is_same<T1, T2>::value;


    template<typename T> using always_false = std::false_type;
    template<typename T> NOA_API inline constexpr bool always_false_v = always_false<T>::value;
}

//@CLION-formatter:on
