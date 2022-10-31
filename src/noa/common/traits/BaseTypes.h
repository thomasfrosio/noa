/// \file noa/common/traits/BaseTypes.h
/// \brief Some type traits.
/// \author Thomas - ffyr2w
/// \date 23 Jul 2020
#pragma once

#include <cstdint>
#include <type_traits>
#include <string>
#include <string_view>
#include <complex>

#include "noa/common/traits/Utilities.h"

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
    template<typename T> using is_restricted_int = std::bool_constant<is_any_v<T, bool, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>>;
    template<typename T> constexpr bool is_restricted_int_v = is_restricted_int<T>::value;
    template<typename... Ts> using are_restricted_int = bool_and<is_restricted_int_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_int_v = are_restricted_int<Ts...>::value;

    // float or double
    template<typename> struct proclaim_is_float : std::false_type {}; // Half is proclaimed in Half.h
    template<> struct proclaim_is_float<float> : std::true_type {};
    template<> struct proclaim_is_float<double> : std::true_type {};
    template<typename T> using is_float = std::bool_constant<proclaim_is_float<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float_v = is_float<T>::value;
    template<typename... Ts> using are_float = bool_and<is_float_v<Ts>...>;
    template<typename... Ts> constexpr bool are_float_v = are_float<Ts...>::value;

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
    template<typename T> using is_float_or_complex = std::bool_constant<is_float_v<T> || is_complex_v<T>>;
    template<typename T> constexpr bool is_float_or_complex_v = is_float_or_complex<T>::value;
    template<typename... Ts> using are_float_or_complex = bool_and<is_float_or_complex_v<Ts>...>;
    template<typename... Ts> constexpr bool are_float_or_complex_v = are_float_or_complex<Ts...>::value;

    // any integer or floating-point
    template<typename T> using is_scalar = std::bool_constant<is_float<T>::value || is_int<T>::value>;
    template<typename T> constexpr bool is_scalar_v = is_scalar<T>::value;
    template<typename... Ts> using are_scalar = bool_and<is_scalar_v<Ts>...>;
    template<typename... Ts> constexpr bool are_scalar_v = are_scalar<Ts...>::value;

    // fixed-sized integers or floating-points
    template<typename T> using is_restricted_scalar = std::bool_constant<is_restricted_int_v<T> || is_float_v<T>>;
    template<typename T> constexpr bool is_restricted_scalar_v = is_restricted_scalar<T>::value;
    template<typename... Ts> using are_restricted_scalar = bool_and<is_restricted_scalar_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_scalar_v = are_restricted_scalar<Ts...>::value;

    // any integer, floating-point or complex floating-point
    template<typename T> using is_data = std::bool_constant<is_int_v<T> || is_float_or_complex_v<T>>;
    template<typename T> constexpr bool is_data_v = is_data<T>::value;
    template<typename... Ts> using are_data = bool_and<is_data_v<Ts>...>;
    template<typename... Ts> constexpr bool are_data_v = are_data<Ts...>::value;

    // fixed-sized integers, floating-points or complex floating-points
    template<typename T> using is_restricted_data = std::bool_constant<is_restricted_scalar_v<T> || is_complex_v<T>>;
    template<typename T> constexpr bool is_restricted_data_v = is_restricted_data<T>::value;
    template<typename... Ts> using are_restricted_data = bool_and<is_restricted_data_v<Ts>...>;
    template<typename... Ts> constexpr bool are_restricted_data_v = are_restricted_data<Ts...>::value;

    // std::string(_view)
    template<typename> struct proclaim_is_string : std::false_type {};
    template<> struct proclaim_is_string<std::string> : std::true_type {};
    template<> struct proclaim_is_string<std::string_view> : std::true_type {};
    template<typename T> using is_string = std::bool_constant<proclaim_is_string<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_string_v = is_string<T>::value;

    template<typename E> using is_scoped_enum = std::bool_constant<std::is_enum_v<E> && !std::is_convertible_v<E, int>>;
    template<typename E> constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value;

    template<typename T> using is_function_ptr = std::bool_constant<std::is_pointer_v<T> && std::is_function_v<std::remove_pointer_t<T>>>;
    template<typename T> constexpr bool is_function_ptr_v = is_function_ptr<T>::value;

    template<typename T> using is_function = std::bool_constant<std::is_function_v<T>>;
    template<typename T> constexpr bool is_function_v = is_function<T>::value;

    template<typename T> using is_valid_ptr_type = std::bool_constant<!std::is_reference_v<T> && !std::is_array_v<T> && !std::is_const_v<T>>;
    template<typename T> constexpr bool is_valid_ptr_type_v = is_valid_ptr_type<T>::value;
}
