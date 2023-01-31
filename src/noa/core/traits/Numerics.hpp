#pragma once

#include <cstdint>
#include <type_traits>
#include <complex>

#include "noa/core/traits/Utilities.hpp"

namespace noa::traits {
    // boolean
    template<typename> struct proclaim_is_bool : std::false_type {};
    template<> struct proclaim_is_bool<bool> : std::true_type {};
    template<typename T> using is_bool = std::bool_constant<proclaim_is_bool<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_bool_v = is_bool<T>::value;
    template<typename... Ts> using are_bool = bool_and<is_bool_v<Ts>...>;
    template<typename... Ts> constexpr bool are_bool_v = are_bool<Ts...>::value;

    // any unsigned integer
    template<typename> struct proclaim_is_uint : std::false_type {};
    template<> struct proclaim_is_uint<bool> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned char> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned short> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned int> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned long> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned long long> : std::true_type {};
    template<> struct proclaim_is_uint<char> : std::conditional_t<std::is_unsigned_v<char>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_uint<wchar_t> : std::conditional_t<std::is_unsigned_v<wchar_t>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_uint<char16_t> : std::conditional_t<std::is_unsigned_v<char16_t>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_uint<char32_t> : std::conditional_t<std::is_unsigned_v<char32_t>, std::true_type, std::false_type> {};
    template<typename T> using is_uint = std::bool_constant<proclaim_is_uint<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint_v = is_uint<T>::value;
    template<typename... Ts> using are_uint = bool_and<is_uint_v<Ts>...>;
    template<typename... Ts> constexpr bool are_uint_v = are_uint<Ts...>::value;

    // any signed integer
    template<typename> struct proclaim_is_sint : std::false_type {};
    template<> struct proclaim_is_sint<signed char> : std::true_type {};
    template<> struct proclaim_is_sint<short> : std::true_type {};
    template<> struct proclaim_is_sint<int> : std::true_type {};
    template<> struct proclaim_is_sint<long> : std::true_type {};
    template<> struct proclaim_is_sint<long long> : std::true_type {};
    template<> struct proclaim_is_sint<char> : std::conditional_t<std::is_signed_v<char>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_sint<wchar_t> : std::conditional_t<std::is_signed_v<wchar_t>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_sint<char16_t> : std::conditional_t<std::is_signed_v<char16_t>, std::true_type, std::false_type> {};
    template<> struct proclaim_is_sint<char32_t> : std::conditional_t<std::is_signed_v<char32_t>, std::true_type, std::false_type> {};
    template<typename T> using is_sint = std::bool_constant<proclaim_is_sint<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_sint_v = is_sint<T>::value;
    template<typename... Ts> using are_sint = bool_and<is_sint_v<Ts>...>;
    template<typename... Ts> constexpr bool are_sint_v = are_sint<Ts...>::value;

    // any integer
    template<typename T> using is_int = std::bool_constant<is_uint_v<T> || is_sint_v<T>>;
    template<typename T> constexpr bool is_int_v = is_int<T>::value;
    template<typename... Ts> using are_int = bool_and<is_int_v<Ts>...>;
    template<typename... Ts> constexpr bool are_int_v = are_int<Ts...>::value;

    // any fixed-size integer (including bool)
    template<typename T> using is_restricted_int = std::bool_constant<is_any_v<T, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>>;
    template<typename T> constexpr bool is_restricted_int_v = is_restricted_int<T>::value;
    template<typename... Ts> using are_restricted_int = bool_and<is_restricted_int_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_int_v = are_restricted_int<Ts...>::value;

    // float or double
    template<typename> struct proclaim_is_real : std::false_type {}; // Half is proclaimed in Half.h
    template<> struct proclaim_is_real<float> : std::true_type {};
    template<> struct proclaim_is_real<double> : std::true_type {};
    template<typename T> using is_real = std::bool_constant<proclaim_is_real<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_real_v = is_real<T>::value;
    template<typename... Ts> using are_real = bool_and<is_real_v<Ts>...>;
    template<typename... Ts> constexpr bool are_real_v = are_real<Ts...>::value;

    // std::complex<float|double>
    template<typename> struct proclaim_is_std_complex : std::false_type {};
    template<> struct proclaim_is_std_complex<std::complex<float>> : std::true_type {};
    template<> struct proclaim_is_std_complex<std::complex<double>> : std::true_type {};
    template<typename T> using is_std_complex = std::bool_constant<proclaim_is_std_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_std_complex_v = is_std_complex<T>::value;
    template<typename... Ts> using are_std_complex = bool_and<is_std_complex_v<Ts>...>;
    template<typename... Ts> constexpr bool are_std_complex_v = are_std_complex<Ts...>::value;

    // Complex<>
    template<typename> struct proclaim_is_complex : std::false_type {}; // Complex<> is proclaimed in Complex.h
    template<typename T> using is_complex = std::bool_constant<proclaim_is_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_complex_v = is_complex<T>::value;
    template<typename... Ts> using are_complex = bool_and<is_complex_v<Ts>...>;
    template<typename... Ts> constexpr bool are_complex_v = are_complex<Ts...>::value;

    // (complex) floating-point
    template<typename T> using is_real_or_complex = std::bool_constant<is_real_v<T> || is_complex_v<T>>;
    template<typename T> constexpr bool is_real_or_complex_v = is_real_or_complex<T>::value;
    template<typename... Ts> using are_real_or_complex = bool_and<is_real_or_complex_v<Ts>...>;
    template<typename... Ts> constexpr bool are_real_or_complex_v = are_real_or_complex<Ts...>::value;

    // any integer or real
    template<typename T> using is_scalar = std::bool_constant<is_real_v<T> || is_int_v<T>>;
    template<typename T> constexpr bool is_scalar_v = is_scalar<T>::value;
    template<typename... Ts> using are_scalar = bool_and<is_scalar_v<Ts>...>;
    template<typename... Ts> constexpr bool are_scalar_v = are_scalar<Ts...>::value;

    // fixed-sized integers or floating-points
    template<typename T> using is_restricted_scalar = std::bool_constant<is_restricted_int_v<T> || is_real_v<T>>;
    template<typename T> constexpr bool is_restricted_scalar_v = is_restricted_scalar<T>::value;
    template<typename... Ts> using are_restricted_scalar = bool_and<is_restricted_scalar_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_scalar_v = are_restricted_scalar<Ts...>::value;

    // any integer, floating-point or complex floating-point
    template<typename T> using is_numeric = std::bool_constant<is_int_v<T> || is_real_or_complex_v<T>>;
    template<typename T> constexpr bool is_numeric_v = is_numeric<T>::value;
    template<typename... Ts> using are_numeric = bool_and<is_numeric_v<Ts>...>;
    template<typename... Ts> constexpr bool are_numeric_v = are_numeric<Ts...>::value;

    // fixed-sized integers, floating-points or complex floating-points
    template<typename T> using is_restricted_numeric = std::bool_constant<is_restricted_scalar_v<T> || is_complex_v<T>>;
    template<typename T> constexpr bool is_restricted_numeric_v = is_restricted_numeric<T>::value;
    template<typename... Ts> using are_restricted_numeric = bool_and<is_restricted_numeric_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_numeric_v = are_restricted_numeric<Ts...>::value;
}
