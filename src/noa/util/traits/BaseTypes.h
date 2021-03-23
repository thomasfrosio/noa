/**
 * @file traits/Base.h
 * @brief Some type traits.
 * @author Thomas - ffyr2w
 * @date 23 Jul 2020
 *
 * Type traits:
 *  -# @c is_uint_v                     : (cv qualifiers) uint8_t, uint16_t, uint32_t, uint64_t
 *  -# @c is_int_v                      : (cv qualifiers) (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t
 *  -# @c is_float_v                    : (cv qualifiers) float|double
 *  -# @c is_std_complex_v              : (cv qualifiers) std::complex<float|double>
 *  -# @c is_complex_v                  : (cv qualifiers) cfloat_t|cdouble_t
 *  -# @c is_scalar_v                   : is_int_v || is_float_v
 *  -# @c is_data_v                     : is_int_v || is_float_v || is_complex_v
 *
 *  -# @c is_bool_v                     : (cv qualifiers) bool
 *  -# @c is_string_v                   : (cv qualifiers) std::string(_view)
 *
 *  -# @c remove_ref_cv<T>              : std::remove_cv_t<std::remove_reference_t<T>>
 *  -# @c value_type_t<T>               : typename T::value_type if it exists, T otherwise;
 *  -# @c is_same_v<T1, T2>             : T1|T2 = (cv) V1|V2(&); check if V1 == V2
 *  -# @c is_scoped_enum_v              : enum class|struct
 *  -# @c is_always_false_v             : false
 *
 * -# @c is_intX_v
 * -# @c is_uintX_v
 * -# @c is_floatX_v
 *
 * -# @c is_valid_ptr_type
 */
#pragma once

#include <cstdint>
#include <type_traits>
#include <string>
#include <string_view>
#include <complex>

//@CLION-formatter:off

namespace Noa::Traits {
    template<typename T> struct remove_ref_cv { using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>; };
    template<typename T> using remove_ref_cv_t = typename remove_ref_cv<T>::type;

    template<typename T, typename = void> struct private_value_type { using type = T; };
    template<typename T> struct private_value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
    template<typename T> struct value_type { using type = typename private_value_type<T>::type ; };
    template<typename T> using value_type_t = typename value_type<T>::type;

    template<typename> struct proclaim_is_uint : std::false_type {};
    template<> struct proclaim_is_uint<uint8_t> : std::true_type {};
    template<> struct proclaim_is_uint<uint16_t> : std::true_type {};
    template<> struct proclaim_is_uint<uint32_t> : std::true_type {};
    template<> struct proclaim_is_uint<uint64_t> : std::true_type {};
    template<typename T> using is_uint = std::bool_constant<proclaim_is_uint<remove_ref_cv_t<T>>::value>;
    template<typename T> inline constexpr bool is_uint_v = is_uint<T>::value; // One of: uint8_t, uint16_t, uint32_t, uint64_t


    template<typename> struct proclaim_is_int : std::false_type {};
    template<> struct proclaim_is_int<int8_t> : std::true_type {};
    template<> struct proclaim_is_int<uint8_t> : std::true_type {};
    template<> struct proclaim_is_int<int16_t> : std::true_type {};
    template<> struct proclaim_is_int<uint16_t> : std::true_type {};
    template<> struct proclaim_is_int<int32_t> : std::true_type {};
    template<> struct proclaim_is_int<uint32_t> : std::true_type {};
    template<> struct proclaim_is_int<int64_t> : std::true_type {};
    template<> struct proclaim_is_int<uint64_t> : std::true_type {};
    template<typename T> using is_int = std::bool_constant<proclaim_is_int<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int_v = is_int<T>::value;  // One of: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t


    template<typename> struct proclaim_is_float : std::false_type {};
    template<> struct proclaim_is_float<float> : std::true_type {};
    template<> struct proclaim_is_float<double> : std::true_type {};
    template<typename T> using is_float = std::bool_constant<proclaim_is_float<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float_v = is_float<T>::value; // One of: float, double, long double


    template<typename> struct proclaim_is_std_complex : std::false_type {};
    template<> struct proclaim_is_std_complex<std::complex<float>> : std::true_type {};
    template<> struct proclaim_is_std_complex<std::complex<double>> : std::true_type {};
    template<typename T> using is_std_complex = std::bool_constant<proclaim_is_std_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_std_complex_v = is_std_complex<T>::value; // One of: std::complex<float|double>


    template<typename> struct proclaim_is_complex : std::false_type {}; // Noa complex is proclaimed in noa/Types.h
    template<typename T> using is_complex = std::bool_constant<proclaim_is_complex<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_complex_v = is_complex<T>::value; // One of: cfloat_t, cdouble_t


    template<typename T> using is_scalar = std::bool_constant<is_float<T>::value || is_int<T>::value>;
    template<typename T> constexpr bool is_scalar_v = is_scalar<T>::value; // One of: is_float_v, is_int_v


    template<typename T> using is_data = std::bool_constant<is_int<T>::value || is_float<T>::value || is_complex<T>::value>;
    template<typename T> constexpr bool is_data_v = is_data<T>::value; // One of: is_int_v, is_float_v, is_complex_v


    template<typename> struct proclaim_is_bool : std::false_type {};
    template<> struct proclaim_is_bool<bool> : std::true_type {};
    template<typename T> using is_bool = std::bool_constant<proclaim_is_bool<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_bool_v = is_bool<T>::value;


    template<typename> struct proclaim_is_string : std::false_type {};
    template<> struct proclaim_is_string<std::string> : std::true_type {};
    template<> struct proclaim_is_string<std::string_view> : std::true_type {};
    template<typename T> using is_string = std::bool_constant<proclaim_is_string<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_string_v = is_string<T>::value; // One of: std::string(_view)


    template<typename E> using is_scoped_enum = std::bool_constant<std::is_enum_v<E> && !std::is_convertible_v<E, int>>;
    template<typename E> constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value; // An enum class


    template<typename T1, typename T2> using is_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;
    template<typename T1, typename T2> inline constexpr bool is_same_v = is_same<T1, T2>::value;


    template<typename T> using always_false = std::false_type;
    template<typename T> inline constexpr bool always_false_v = always_false<T>::value;

    // IntX and FloatX
    template<typename> struct proclaim_is_intX : std::false_type {};
    template<typename T> using is_intX = std::bool_constant<proclaim_is_intX<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_intX_v = is_intX<T>::value;

    template<typename> struct proclaim_is_uintX : std::false_type {};
    template<typename T> using is_uintX = std::bool_constant<proclaim_is_uintX<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uintX_v = is_uintX<T>::value;

    template<typename> struct proclaim_is_floatX : std::false_type {};
    template<typename T> using is_floatX = std::bool_constant<proclaim_is_floatX<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_floatX_v = is_floatX<T>::value;

    template<typename T> using is_function_ptr = std::bool_constant<std::is_pointer_v<T> &&
                                                                     std::is_function_v<std::remove_pointer_t<T>>>;
    template<typename T> constexpr bool is_function_ptr_v = is_function_ptr<T>::value;

    template<typename T> using is_function = std::bool_constant<std::is_function_v<T>>;
    template<typename T> constexpr bool is_function_v = is_function<T>::value;

    template<typename T>
    using is_valid_ptr_type = std::bool_constant<
            (std::is_arithmetic_v<T> || Noa::Traits::is_complex_v<T> || Noa::Traits::is_same_v<std::byte, T>)
            && !std::is_reference_v<T> && !std::is_array_v<T> && !std::is_const_v<T>>;
    template<typename T> constexpr bool is_valid_ptr_type_v = is_valid_ptr_type<T>::value;
}

//@CLION-formatter:on
