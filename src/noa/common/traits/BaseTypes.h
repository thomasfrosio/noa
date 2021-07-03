/// @file noa/common/traits/BaseTypes.h
/// @brief Some type traits.
/// @author Thomas - ffyr2w
/// @date 23 Jul 2020

#pragma once

#include <cstdint>
#include <type_traits>
#include <string>
#include <string_view>
#include <complex>

//@CLION-formatter:off

namespace noa::traits {
    template<typename T> struct remove_ref_cv { using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>; };
    /// Removes the const/volatile and reference from T.
    template<typename T> using remove_ref_cv_t = typename remove_ref_cv<T>::type;

    template<typename T, typename = void> struct private_value_type { using type = T; };
    template<typename T> struct private_value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
    template<typename T> struct value_type { using type = typename private_value_type<T>::type ; };
    /// Extracts the typedef value_type from T if it exists, returns T otherwise.
    template<typename T> using value_type_t = typename value_type<T>::type;

    template<typename> struct proclaim_is_uint : std::false_type {};
    template<> struct proclaim_is_uint<uint8_t> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned short> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned int> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned long> : std::true_type {};
    template<> struct proclaim_is_uint<unsigned long long> : std::true_type {};
    template<typename T> using is_uint = std::bool_constant<proclaim_is_uint<remove_ref_cv_t<T>>::value>;
    /// One of: uint8_t, unsigned short, unsigned int, unsigned long, unsigned long long. \c remove_ref_cv_t is applied to T.
    template<typename T> inline constexpr bool is_uint_v = is_uint<T>::value;

    template<typename> struct proclaim_is_int : std::false_type {};
    template<> struct proclaim_is_int<int8_t> : std::true_type {};
    template<> struct proclaim_is_int<uint8_t> : std::true_type {};
    template<> struct proclaim_is_int<short> : std::true_type {};
    template<> struct proclaim_is_int<unsigned short> : std::true_type {};
    template<> struct proclaim_is_int<int> : std::true_type {};
    template<> struct proclaim_is_int<unsigned int> : std::true_type {};
    template<> struct proclaim_is_int<long> : std::true_type {};
    template<> struct proclaim_is_int<unsigned long> : std::true_type {};
    template<> struct proclaim_is_int<long long> : std::true_type {};
    template<> struct proclaim_is_int<unsigned long long> : std::true_type {};
    template<typename T> using is_int = std::bool_constant<proclaim_is_int<remove_ref_cv_t<T>>::value>;
    /// One of: (u)int8_t, (u)short, (u)int, (u)long, (u)long long. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_int_v = is_int<T>::value;

    template<typename> struct proclaim_is_float : std::false_type {};
    template<> struct proclaim_is_float<float> : std::true_type {};
    template<> struct proclaim_is_float<double> : std::true_type {};
    template<typename T> using is_float = std::bool_constant<proclaim_is_float<remove_ref_cv_t<T>>::value>;
    /// One of: float, double. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_float_v = is_float<T>::value;

    template<typename> struct proclaim_is_std_complex : std::false_type {};
    template<> struct proclaim_is_std_complex<std::complex<float>> : std::true_type {};
    template<> struct proclaim_is_std_complex<std::complex<double>> : std::true_type {};
    template<typename T> using is_std_complex = std::bool_constant<proclaim_is_std_complex<remove_ref_cv_t<T>>::value>;
    /// One of: std::complex<float|double>. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_std_complex_v = is_std_complex<T>::value;

    template<typename> struct proclaim_is_complex : std::false_type {}; // noa complex is proclaimed in noa/Types.h
    template<typename T> using is_complex = std::bool_constant<proclaim_is_complex<remove_ref_cv_t<T>>::value>;
    /// // One of: cfloat_t, cdouble_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_complex_v = is_complex<T>::value;

    template<typename T> using is_scalar = std::bool_constant<is_float<T>::value || is_int<T>::value>;
    /// One of: \c is_float_v, \c is_int_v.
    template<typename T> constexpr bool is_scalar_v = is_scalar<T>::value;

    template<typename T> using is_data = std::bool_constant<is_int<T>::value || is_float<T>::value || is_complex<T>::value>;
    /// One of: \c is_int_v, \c is_float_v, \c is_complex_v.
    template<typename T> constexpr bool is_data_v = is_data<T>::value;

    template<typename> struct proclaim_is_bool : std::false_type {};
    template<> struct proclaim_is_bool<bool> : std::true_type {};
    template<typename T> using is_bool = std::bool_constant<proclaim_is_bool<remove_ref_cv_t<T>>::value>;
    /// One of: bool. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_bool_v = is_bool<T>::value;

    template<typename> struct proclaim_is_string : std::false_type {};
    template<> struct proclaim_is_string<std::string> : std::true_type {};
    template<> struct proclaim_is_string<std::string_view> : std::true_type {};
    template<typename T> using is_string = std::bool_constant<proclaim_is_string<remove_ref_cv_t<T>>::value>;
    /// One of: \c std::string(_view). \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_string_v = is_string<T>::value;

    template<typename E> using is_scoped_enum = std::bool_constant<std::is_enum_v<E> && !std::is_convertible_v<E, int>>;
    /// Whether \a E is an enum class.
    template<typename E> constexpr bool is_scoped_enum_v = is_scoped_enum<E>::value;

    template<typename T1, typename T2> using is_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;
    /// Whether \a T1 and \a T2 are the same types, ignoring const/volatile and reference.
    template<typename T1, typename T2> inline constexpr bool is_same_v = is_same<T1, T2>::value;

    template<typename T> using always_false = std::false_type;
    /// Always false. Used to invalidate some code paths at compile time.
    template<typename T> inline constexpr bool always_false_v = always_false<T>::value;

    template<typename> struct proclaim_is_boolX : std::false_type {}; // added by BoolX.h
    template<typename T> using is_boolX = std::bool_constant<proclaim_is_boolX<remove_ref_cv_t<T>>::value>;
    /// One of: bool2_t, bool3_t, bool4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_boolX_v = is_boolX<T>::value;

    template<typename> struct proclaim_is_intX : std::false_type {}; // added by IntX.h
    template<typename T> using is_intX = std::bool_constant<proclaim_is_intX<remove_ref_cv_t<T>>::value>;
    /// One of: int2_t, int3_t, int4_t, long2_t, long3_t, long4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_intX_v = is_intX<T>::value;

    template<typename> struct proclaim_is_uintX : std::false_type {}; // added by IntX.h
    template<typename T> using is_uintX = std::bool_constant<proclaim_is_uintX<remove_ref_cv_t<T>>::value>;
    /// One of: uint2_t, uint3_t, uint4_t, ulong2_t, ulong3_t, ulong4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_uintX_v = is_uintX<T>::value;

    template<typename> struct proclaim_is_floatX : std::false_type {}; // added by FloatX.h
    template<typename T> using is_floatX = std::bool_constant<proclaim_is_floatX<remove_ref_cv_t<T>>::value>;
    /// One of: float2_t, float3_t, float4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_floatX_v = is_floatX<T>::value;

    template<typename T> using is_function_ptr = std::bool_constant<std::is_pointer_v<T> && std::is_function_v<std::remove_pointer_t<T>>>;
    template<typename T> constexpr bool is_function_ptr_v = is_function_ptr<T>::value;

    template<typename T> using is_function = std::bool_constant<std::is_function_v<T>>;
    template<typename T> constexpr bool is_function_v = is_function<T>::value;

    template<typename T>
    using is_valid_ptr_type = std::bool_constant<!std::is_reference_v<T> && !std::is_array_v<T> && !std::is_const_v<T>>;
    /// Whether T can be used by the Ptr* classes, i.e. any type that is not a reference, an array or const qualified.
    template<typename T> constexpr bool is_valid_ptr_type_v = is_valid_ptr_type<T>::value;
}

//@CLION-formatter:on
