#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Namespace.hpp"

#if defined(NOA_IS_OFFLINE)
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#else
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#endif

// Assume POSIX and/or Windows, both of which guarantee CHAR_BIT == 8.
// The rest should fine for all modern hardware.
static_assert(CHAR_BIT == 8);
static_assert(sizeof(short) == 2);
static_assert(sizeof(int) == 4);
static_assert(sizeof(float) == 4);
static_assert(std::is_same_v<int8_t, signed char>);
static_assert(std::is_same_v<uint8_t, unsigned char>);
static_assert(std::is_same_v<int16_t, signed short>);
static_assert(std::is_same_v<uint16_t, unsigned short>);
static_assert(std::is_same_v<int32_t, signed int>);
static_assert(std::is_same_v<uint32_t, unsigned int>);

namespace noa::inline types {
    struct Empty {};
    using Byte = std::byte;

    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;

    using f32 = float;
    using f64 = double;
    static_assert(sizeof(f32) == 4);
    static_assert(sizeof(f64) == 8);
}

namespace noa::traits {
    template<typename T> struct remove_ref_cv { using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>; };
    template<typename T> using remove_ref_cv_t = typename remove_ref_cv<T>::type;

    template<typename T> struct remove_pointer_cv { using type = typename std::remove_cv_t<typename std::remove_pointer_t<T>>; };
    template<typename T> using remove_pointer_cv_t = typename remove_pointer_cv<T>::type;

    template <bool... Bs> using bool_sequence_t = std::integer_sequence<bool, Bs...>;
    template <bool... Bs> using bool_and = std::is_same<bool_sequence_t<Bs...>, bool_sequence_t<(Bs || true)...>>;
    template <bool... Bs> using bool_or = std::integral_constant<bool, !bool_and<!Bs...>::value>;

    template<typename T> using always_false = std::false_type;
    template<typename T> constexpr bool always_false_v = always_false<T>::value;

    template<bool B>
    using enable_if_bool_t = std::enable_if_t<B, bool>;

    template<typename T>
    using identity_t = T;

    template<typename First, typename...>
    using first_t = First;
}

namespace noa::traits {
    namespace guts {
        template<typename T, typename = void> struct type_type { using type = T; };
        template<typename T, typename = void> struct value_type { using type = T; };
        template<typename T, typename = void> struct mutable_value_type { using type = T; };
        template<typename T, typename = void> struct element_type { using type = T; };
        template<typename T, typename = void> struct index_type { using type = T; };
        template<typename T, typename = void> struct shared_type { using type = T; };
        template<typename T, typename = void> struct pointer_type { using type = T; };

        template<typename T> struct type_type<T, std::void_t<typename T::type_type>> { using type = typename T::type; };
        template<typename T> struct value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
        template<typename T> struct mutable_value_type<T, std::void_t<typename T::mutable_value_type>> { using type = typename T::mutable_value_type; };
        template<typename T> struct element_type<T, std::void_t<typename T::element_type>> { using type = typename T::element_type; };
        template<typename T> struct index_type<T, std::void_t<typename T::index_type>> { using type = typename T::index_type; };
        template<typename T> struct pointer_type<T, std::void_t<typename T::pointer_type>> { using type = typename T::pointer_type; };
        template<typename T> struct shared_type<T, std::void_t<typename T::shared_type>> { using type = typename T::shared_type; };
    }

    template<typename T> struct value_type { using type = typename guts::value_type<T>::type; };
    template<typename T> struct mutable_value_type { using type = typename guts::mutable_value_type<T>::type; };
    template<typename T> struct element_type { using type = typename guts::element_type<T>::type; };
    template<typename T> struct index_type { using type = typename guts::index_type<T>::type; };
    template<typename T> struct pointer_type { using type = typename guts::pointer_type<T>::type; };
    template<typename T> struct shared_type { using type = typename guts::shared_type<T>::type; };

    template<typename T> using type_t = typename guts::type_type<T>::type;
    template<typename T> using value_type_t = typename value_type<std::decay_t<T>>::type;
    template<typename T> using value_type_twice_t = value_type_t<value_type_t<T>>;
    template<typename T> using mutable_value_type_t = typename mutable_value_type<std::decay_t<T>>::type;
    template<typename T> using element_type_t = typename element_type<std::decay_t<T>>::type;
    template<typename T> using index_type_t = typename index_type<std::decay_t<T>>::type;
    template<typename T> using pointer_type_t = typename pointer_type<std::decay_t<T>>::type;
    template<typename T> using shared_type_t = typename shared_type<std::decay_t<T>>::type;

    template<typename InputValue, typename OutputValue>
    constexpr bool is_mutable_value_type_v =
            std::is_const_v<OutputValue> &&
            std::is_same_v<InputValue, std::remove_const_t<OutputValue>>;
}

namespace noa::traits {
    template<typename T1, typename T2> using is_almost_same = std::bool_constant<std::is_same_v<std::decay_t<T1>, std::decay_t<T2>>>;
    template<typename T1, typename T2> constexpr bool is_almost_same_v = is_almost_same<T1, T2>::value;

    template<typename T, typename... Ts> struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
    template<typename T, typename... Ts> constexpr bool is_any_v = is_any<T, Ts...>::value;

    template<typename T, typename... Ts> struct is_almost_any : std::bool_constant<(is_almost_same_v<T, Ts> || ...)> {};
    template<typename T, typename... Ts> constexpr bool is_almost_any_v = is_almost_any<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_all_same : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_all_same_v = are_all_same<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_almost_all_same : std::bool_constant<(is_almost_same_v<T, Ts> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_almost_all_same_v = are_almost_all_same<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_almost_same_value_type : std::bool_constant<(is_almost_same_v<value_type_t<T>, value_type_t<Ts>> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_almost_same_value_type_v = are_almost_same_value_type<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_same_value_type : std::bool_constant<(std::is_same_v<value_type_t<T>, value_type_t<Ts>> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_same_value_type_v = are_same_value_type<T, Ts...>::value;
}

// From https://en.cppreference.com/w/cpp/experimental/is_detected
// And https://stackoverflow.com/a/41936999
namespace noa::traits {
    namespace guts {
        struct nonesuch {
            nonesuch() = delete;
            ~nonesuch() = delete;
            nonesuch(nonesuch const&) = delete;
            void operator=(nonesuch const&) = delete;
        };

        template<typename Default, typename AlwaysVoid, template<typename...> typename Op, typename... Args>
        struct detector {
            using value_t = std::false_type;
            using type = Default;
        };
        template<typename Default, template<typename...> typename Op, typename... Args>
        struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };
    }

    template<template<typename...> typename Op, typename... Args>
    using is_detected = typename guts::detector<guts::nonesuch, void, Op, Args...>::value_t;

    template<template<typename...> typename Op, typename... Args>
    using detected_t = typename guts::detector<guts::nonesuch, void, Op, Args...>::type;

    template<typename Default, template<typename...> typename Op, typename... Args>
    using detected_or = guts::detector<Default, void, Op, Args...>;

    template< template<typename...> typename Op, typename... Args>
    constexpr inline bool is_detected_v = is_detected<Op, Args...>::value;

    template< typename Default, template<typename...> typename Op, typename... Args>
    using detected_or_t = typename detected_or<Default, Op, Args...>::type;

    template <typename Expected, template<typename...> typename Op, typename... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

    template <typename Expected, template<typename...> typename Op, typename... Args>
    constexpr inline bool is_detected_exact_v =
            is_detected_exact<Expected, Op, Args...>::value;

    template <typename To, template<typename...> typename Op, typename... Args>
    using is_detected_convertible =
            std::is_convertible<detected_t<Op, Args...>, To>;

    template <typename To, template<typename...> typename Op, typename... Args>
    constexpr inline bool is_detected_convertible_v =
            is_detected_convertible<To, Op, Args...>::value;

    // Predefined detection traits.
    template<class T> using has_name = decltype(T::name());
    template<class T> using has_initialize = decltype(std::declval<T&>().initialize(std::declval<int64_t>()));
    template<class T> using has_closure = decltype(std::declval<T&>().closure(std::declval<int64_t>()));

    template<typename Op, typename Lhs>
    using has_unary_operator = decltype(std::declval<Op&>().operator()(std::declval<Lhs>()));

    template<typename Op, typename Lhs, typename Rhs>
    using has_binary_operator = decltype(std::declval<Op&>().operator()(std::declval<Lhs>(), std::declval<Rhs>()));

    template<typename Op, typename Lhs, typename Mhs, typename Rhs>
    using has_trinary_operator = decltype(std::declval<Op&>().operator()(std::declval<Lhs>(), std::declval<Mhs>(), std::declval<Rhs>()));

    template<typename T> using has_greater_operator = decltype(operator>(std::declval<const T&>(), std::declval<const T&>()));
    template<typename T> using has_less_operator = decltype(operator<(std::declval<const T&>(), std::declval<const T&>()));
}

namespace noa::traits {
    // boolean
    template<typename> struct proclaim_is_bool : std::false_type {};
    template<> struct proclaim_is_bool<bool> : std::true_type {};
    template<typename T> using is_bool = std::bool_constant<proclaim_is_bool<T>::value>;
    template<typename T> constexpr bool is_bool_v = is_bool<std::decay_t<T>>::value;
    template<typename... Ts> using are_bool = bool_and<is_bool<Ts>::value...>;
    template<typename... Ts> constexpr bool are_bool_v = are_bool<std::decay_t<Ts>...>::value;

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
    template<typename T> using is_uint = std::bool_constant<proclaim_is_uint<T>::value>;
    template<typename T> constexpr bool is_uint_v = is_uint<std::decay_t<T>>::value;
    template<typename... Ts> using are_uint = bool_and<is_uint<Ts>::value...>;
    template<typename... Ts> constexpr bool are_uint_v = are_uint<std::decay_t<Ts>...>::value;

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
    template<typename T> using is_sint = std::bool_constant<proclaim_is_sint<T>::value>;
    template<typename T> constexpr bool is_sint_v = is_sint<std::decay_t<T>>::value;
    template<typename... Ts> using are_sint = bool_and<is_sint<Ts>::value...>;
    template<typename... Ts> constexpr bool are_sint_v = are_sint<std::decay_t<Ts>...>::value;

    // any integer
    template<typename T> using is_int = std::bool_constant<is_uint<T>::value || is_sint<T>::value>;
    template<typename T> constexpr bool is_int_v = is_int<std::decay_t<T>>::value;
    template<typename... Ts> using are_int = bool_and<is_int<Ts>::value...>;
    template<typename... Ts> constexpr bool are_int_v = are_int<std::decay_t<Ts>...>::value;

    // float or double
    template<typename> struct proclaim_is_real : std::false_type {}; // Half is proclaimed in core/types/Half.hpp
    template<> struct proclaim_is_real<float> : std::true_type {};
    template<> struct proclaim_is_real<double> : std::true_type {};
    template<typename T> using is_real = std::bool_constant<proclaim_is_real<T>::value>;
    template<typename T> constexpr bool is_real_v = is_real<std::decay_t<T>>::value;
    template<typename... Ts> using are_real = bool_and<is_real<Ts>::value...>;
    template<typename... Ts> constexpr bool are_real_v = are_real<std::decay_t<Ts>...>::value;

    // std::complex<float|double>
    template<typename> struct proclaim_is_std_complex : std::false_type {};
    template<typename T> using is_std_complex = std::bool_constant<proclaim_is_std_complex<T>::value>;
    template<typename T> constexpr bool is_std_complex_v = is_std_complex<std::decay_t<T>>::value;
    template<typename... Ts> using are_std_complex = bool_and<is_std_complex<Ts>::value...>;
    template<typename... Ts> constexpr bool are_std_complex_v = are_std_complex<std::decay_t<Ts>...>::value;

    // Complex<>
    template<typename> struct proclaim_is_complex : std::false_type {}; // Complex<> is proclaimed in core/types/Complex.hpp
    template<typename T> using is_complex = std::bool_constant<proclaim_is_complex<T>::value>;
    template<typename T> constexpr bool is_complex_v = is_complex<std::decay_t<T>>::value;
    template<typename... Ts> using are_complex = bool_and<is_complex<Ts>::value...>;
    template<typename... Ts> constexpr bool are_complex_v = are_complex<std::decay_t<Ts>...>::value;

    // (complex) floating-point
    template<typename T> using is_real_or_complex = std::bool_constant<is_real<T>::value || is_complex<T>::value>;
    template<typename T> constexpr bool is_real_or_complex_v = is_real_or_complex<std::decay_t<T>>::value;
    template<typename... Ts> using are_real_or_complex = bool_and<is_real_or_complex<Ts>::value...>;
    template<typename... Ts> constexpr bool are_real_or_complex_v = are_real_or_complex<std::decay_t<Ts>...>::value;

    // any integer or real
    template<typename T> using is_scalar = std::bool_constant<is_real<T>::value || is_int<T>::value>;
    template<typename T> constexpr bool is_scalar_v = is_scalar<std::decay_t<T>>::value;
    template<typename... Ts> using are_scalar = bool_and<is_scalar<Ts>::value...>;
    template<typename... Ts> constexpr bool are_scalar_v = are_scalar<std::decay_t<Ts>...>::value;

    // any integer, floating-point or complex floating-point
    template<typename T> using is_numeric = std::bool_constant<is_int<T>::value || is_real_or_complex<T>::value>;
    template<typename T> constexpr bool is_numeric_v = is_numeric<std::decay_t<T>>::value;
    template<typename... Ts> using are_numeric = bool_and<is_numeric<Ts>::value...>;
    template<typename... Ts> constexpr bool are_numeric_v = are_numeric<std::decay_t<Ts>...>::value;
}
