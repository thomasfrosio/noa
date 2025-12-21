#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <concepts>
#include <utility>

#include "noa/base/Config.hpp"

namespace noa {
    inline namespace types {}

    namespace details {}
    namespace traits {}
    namespace access {}
    namespace nt = ::noa::traits;
    namespace nd = ::noa::details;
    namespace na = ::noa::access;

    namespace xform {}
    namespace signal {}
    namespace fft {}
    namespace nf = ::noa::fft;
    namespace ns = ::noa::signal;
    namespace nx = ::noa::xform;
}

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
    struct Unchecked {};
    using Byte = std::byte;

    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    using i8 = int8_t;
    using i16 = int16_t;
    using i32 = int32_t;
    using i64 = int64_t;

    using usize = size_t;
    using isize = std::make_signed_t<size_t>; // ssize_t

    using f32 = float;
    using f64 = double;
    static_assert(sizeof(f32) == 4);
    static_assert(sizeof(f64) == 8);
}

namespace noa::traits {
    template<typename T = void> concept always_false = false;
    template<typename T = void> constexpr bool always_false_t = false;
    template<typename T> concept empty = std::is_empty_v<T>;

    template<typename T, typename... U>
    struct first { using type = T; };
    template<typename... T>
    using first_t = first<T...>::type;

    template<typename... T>
    struct TypeList {};

    template<typename... Ls, typename... Rs>
    constexpr auto operator+(TypeList<Ls...>, TypeList<Rs...>) {
        return TypeList<Ls..., Rs...>{};
    }

    namespace details {
        template<typename, typename>
        struct append_to_type_seq {};

        template<typename T, typename... Ts, template<typename...> class TT>
        struct append_to_type_seq<T, TT<Ts...>> {
            using type = TT<Ts..., T>;
        };
    }

    template<typename T, usize N, template<typename...> class TT>
    struct repeat {
        using type = details::append_to_type_seq<T, typename repeat<T, N - 1, TT>::type>::type;
    };

    template<typename T, template<typename...> class TT>
    struct repeat<T, 0, TT> {
        using type = TT<>;
    };

    template<typename T, usize N, template<typename...> class L>
    using repeat_t = repeat<T, N, L>::type;

    // std::conjunction equivalent that default to false_type if no types are specified.
    template<typename... T> struct conjunction_or_false : std::conjunction<T...> {};
    template<> struct conjunction_or_false<> : std::false_type {};

    template<typename... Ts>
    struct largest_type;

    template<typename T>
    struct largest_type<T> {
        using type = T;
    };

    template<typename T, typename U, typename... Ts>
    struct largest_type<T, U, Ts...> {
        using type = typename largest_type<typename std::conditional<(sizeof(U) <= sizeof(T)), T, U>::type, Ts...>::type;
    };

    template<typename... Ts>
    using largest_type_t = typename largest_type<Ts...>::type;
}

namespace noa::traits {
    #define NOA_TRAITS_(name, default_type, special_type)       \
    namespace details {                                         \
        template<typename T, typename = void>                   \
        struct name {                                           \
            using type = default_type;                          \
            static constexpr bool HAS_ALIAS = false;            \
        };                                                      \
        template<typename T>                                    \
        struct name<T, std::void_t<typename special_type>> {    \
            using type = special_type;                          \
            static constexpr bool HAS_ALIAS = true;             \
        };                                                      \
    }                                                           \
    template<typename T>                                        \
    struct name {                                               \
        using type = details::name<std::decay_t<T>>::type;      \
    };                                                          \
    template<typename T>                                        \
    using name##_t = name<T>::type;                             \
                                                                \
    template<typename T>                                        \
    constexpr bool has_##name##_v = details::name<std::decay_t<T>>::HAS_ALIAS;

    NOA_TRAITS_(type_type, T, T::type);
    NOA_TRAITS_(index_type, T, T::index_type);
    NOA_TRAITS_(shared_type, T, T::shared_type);
    NOA_TRAITS_(value_type, std::remove_pointer_t<T>, T::value_type);
    NOA_TRAITS_(mutable_value_type, std::remove_const_t<value_type_t<T>>, T::mutable_value_type);
    NOA_TRAITS_(const_value_type, std::add_const_t<mutable_value_type_t<T>>, T::const_value_type);
    NOA_TRAITS_(reference_type, std::add_lvalue_reference_t<value_type_t<T>>, T::reference_type);
    NOA_TRAITS_(pointer_type, std::add_pointer_t<value_type_t<T>>, T::pointer_type);
    #undef NOA_TRAITS_
    template<typename T> using value_type_twice_t         = value_type_t<value_type_t<T>>;
    template<typename T> using mutable_value_type_twice_t = mutable_value_type_t<mutable_value_type_t<T>>;

    template<typename T, typename U> struct is_same : std::is_same<T, U> {};
    template<typename T, typename U> constexpr bool is_same_v = is_same<T, U>::value;
    template<typename T, typename... U> struct are_same : conjunction_or_false<std::is_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool are_same_v = are_same<T, U...>::value;
    template<typename T, typename... U> concept same_as = are_same_v<T, U...>;
    // std::same_as can be used to disambiguate overloads when the T/U types are entered in the reverse order.
    // nt::same_as cannot, but we never use that feature (if we end up relying on it, it won't compile anyway,
    // it's not a silent killer) and prefer being able to compare multiple types with T in a simple syntax.

    template<typename T, typename U> struct is_almost_same : std::is_same<std::decay_t<T>, std::decay_t<U>> {};
    template<typename T, typename U> constexpr bool is_almost_same_v = is_almost_same<T, U>::value;
    template<typename T, typename... U> struct are_almost_same : conjunction_or_false<is_almost_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool are_almost_same_v = are_almost_same<T, U...>::value;
    template<typename T, typename... U> concept almost_same_as = are_almost_same_v<T, U...>;

    template<typename T, typename... U> struct is_any : std::disjunction<std::is_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool is_any_v = is_any<T, U...>::value;
    template<typename T, typename... U> concept any_of = (same_as<T, U> or ...); // "fold or" defaults to false

    template<typename T, typename... U> struct is_almost_any : std::disjunction<is_almost_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool is_almost_any_v = is_almost_any<T, U...>::value;
    template<typename T, typename... U> concept almost_any_of = (almost_same_as<T, U> or ...); // "fold or" defaults to false

    template<typename T, typename U> concept same_as_value_type_of = same_as<T, value_type_t<U>>;
    template<typename T, typename U> concept same_as_mutable_value_type_of = same_as<T, mutable_value_type_t<U>>;
    template<typename T, typename U> concept almost_same_as_value_type_of = almost_same_as<T, value_type_t<U>>;
    template<typename T, typename U> concept almost_same_as_mutable_value_type_of = almost_same_as<T, mutable_value_type_t<U>>;

    template<typename T, typename... U> concept same_value_type = (same_as<value_type_t<T>, value_type_t<U>> and ...);
    template<typename T, typename... U> concept same_mutable_value_type = (same_as<mutable_value_type_t<T>, mutable_value_type_t<U>> and ...);

    template<typename Input, typename Output>
    concept mutable_of = std::same_as<Input, std::remove_const_t<Output>> and std::is_const_v<Output>;
}

namespace noa::traits {
    template<typename, usize S, typename = void>
    struct size_or {
        static constexpr usize value = S;
    };
    template<typename T, usize S>
    struct size_or<T, S, std::void_t<decltype(std::remove_reference_t<T>::SIZE)>> {
        static constexpr usize value = std::remove_reference_t<T>::SIZE;
    };
    template<typename T, usize S>
    constexpr usize size_or_v = size_or<T, S>::value;

    template<typename... Ts>
    struct have_same_size {
        static constexpr usize N = first_t<std::remove_reference_t<Ts>...>::SIZE;
        static constexpr bool value = ((N == std::remove_reference_t<Ts>::SIZE) and ...);
    };
    template<typename T>
    struct have_same_size<T> {
        static constexpr usize N = first_t<std::remove_reference_t<T>>::SIZE;
        static constexpr bool value = true;
    };
    template<>
    struct have_same_size<> {
        static constexpr usize N = 0;
        static constexpr bool value = true;
    };
    template<typename... Ts>
    constexpr bool have_same_size_v = have_same_size<Ts...>::value;
}

#define NOA_GENERATE_PROCLAIM_UTILS(name)                                                           \
    template<typename T> constexpr bool is_##name##_v = is_##name<T>::value;                        \
    template<typename... T> using are_##name = noa::traits::conjunction_or_false<is_##name<T>...>;  \
    template<typename... T> constexpr bool are_##name##_v = are_##name<T...>::value

#define NOA_GENERATE_PROCLAIM(name)                                                 \
    template<typename> struct proclaim_is_##name : std::false_type {};              \
    template<typename T> using is_##name = proclaim_is_##name<std::remove_cv_t<T>>; \
    NOA_GENERATE_PROCLAIM_UTILS(name)

#define NOA_GENERATE_PROCLAIM_FULL(name)    \
    NOA_GENERATE_PROCLAIM(name);            \
    template<typename... T> concept name = are_##name##_v<T...>

#define NOA_GENERATE_PROCLAIM_FULL_ND(name)                                                                              \
    template<typename T, usize N> struct proclaim_is_##name : std::false_type {};                                        \
    template<typename T, usize... N> using is_##name = std::disjunction<proclaim_is_##name<std::remove_cv_t<T>, N>...>;  \
    template<typename T, usize... N> constexpr bool is_##name##_v = is_##name<T, N...>::value;                           \
    template<typename T, usize... N> concept name = is_##name##_v<T, N...>;                                              \
    template<usize N, typename... T> using are_##name = noa::traits::conjunction_or_false<is_##name<T, N>...>;           \
    template<usize N, typename... T> constexpr bool are_##name##_v = are_##name<N, T...>::value;

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(integer);
    template<> struct proclaim_is_integer<bool> : std::true_type {};
    template<> struct proclaim_is_integer<uint8_t> : std::true_type {};
    template<> struct proclaim_is_integer<unsigned short> : std::true_type {};
    template<> struct proclaim_is_integer<unsigned int> : std::true_type {};
    template<> struct proclaim_is_integer<unsigned long> : std::true_type {};
    template<> struct proclaim_is_integer<unsigned long long> : std::true_type {};
    template<> struct proclaim_is_integer<int8_t> : std::true_type {};
    template<> struct proclaim_is_integer<signed short> : std::true_type {};
    template<> struct proclaim_is_integer<signed int> : std::true_type {};
    template<> struct proclaim_is_integer<signed long> : std::true_type {};
    template<> struct proclaim_is_integer<signed long long> : std::true_type {};

    template<typename T, bool = is_integer_v<T>> struct proclaim_is_sinteger : std::false_type {};
    template<typename T> struct proclaim_is_sinteger<T, true> : std::bool_constant<T(-1) < T(0)> {};
    template<typename T> struct is_sinteger : proclaim_is_sinteger<T>::type {};
    NOA_GENERATE_PROCLAIM_UTILS(sinteger);
    template<typename... T> concept sinteger = integer<T...> and are_sinteger_v<T...>;

    template<typename T> struct is_uinteger : std::conjunction<is_integer<T>, std::negation<is_sinteger<T>>>::type{};
    NOA_GENERATE_PROCLAIM_UTILS(uinteger);
    template<typename... T> concept uinteger = integer<T...> and are_uinteger_v<T...>;

    NOA_GENERATE_PROCLAIM(boolean);
    template<> struct proclaim_is_boolean<bool> : std::true_type {};
    template<typename... T> concept boolean = uinteger<T...> and are_boolean_v<T...>;

    NOA_GENERATE_PROCLAIM(complex);
    template<typename... T> concept complex = are_complex_v<T...>;

    // floating-points
    template<typename> struct proclaim_is_real : std::false_type {}; // add more floating-point types, e.g. f16
    template<typename T> using is_real = std::disjunction<std::is_floating_point<T>, proclaim_is_real<std::remove_cv_t<T>>>;
    NOA_GENERATE_PROCLAIM_UTILS(real);
    template<typename... T> concept real = are_real_v<T...>;

    // (complex) floating-point
    template<typename T> using is_real_or_complex = std::disjunction<is_real<T>, is_complex<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(real_or_complex);
    template<typename... T> concept real_or_complex = real<T...> or complex<T...>;

    // any integer or real
    template<typename T> using is_scalar = std::disjunction<is_real<T>, is_integer<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(scalar);
    template<typename... T> concept scalar = real<T...> or integer<T...>;

    // any integer, floating-point or complex floating-point
    template<typename T> using is_numeric = std::disjunction<is_integer<T>, is_real_or_complex<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(numeric);
    template<typename... T> concept numeric = integer<T...> or real<T...> or complex<T...>;

    // pointer
    template<typename T> using is_pointer = std::is_pointer<T>;
    NOA_GENERATE_PROCLAIM_UTILS(pointer);
    template<typename... T> concept pointer = are_pointer_v<T...>;

    #define NOA_TRAITS_POINTER(name)                                                                                    \
    template<typename T> using is_pointer_##name = std::conjunction<is_pointer<T>, is_##name<std::remove_pointer_t<T>>>;\
    template<typename T> constexpr bool is_pointer_##name##_v = is_pointer_##name<T>::value;                            \
    template<typename... T> using are_pointer_##name = conjunction_or_false<is_pointer_##name<T>...>;                   \
    template<typename... T> constexpr bool are_pointer_##name##_v = are_pointer_##name<T...>::value;                    \
    template<typename... T> concept pointer_##name = pointer<T...> and are_pointer_##name##_v<T...>
    NOA_TRAITS_POINTER(integer);
    NOA_TRAITS_POINTER(sinteger);
    NOA_TRAITS_POINTER(uinteger);
    NOA_TRAITS_POINTER(boolean);
    NOA_TRAITS_POINTER(real);
    NOA_TRAITS_POINTER(scalar);
    NOA_TRAITS_POINTER(numeric);

    // Convert floating-points to the double-precision version. Otherwise, return the same type.
    template<typename T> struct double_precision { using type = T; };
    template<> struct double_precision<f32> { using type = f64; };
    template<> struct double_precision<f64> { using type = f64; };
    template<typename T> using double_precision_t = typename double_precision<std::remove_cv_t<T>>::type;
}

namespace noa::traits {
    // Vec
    NOA_GENERATE_PROCLAIM_FULL(vec);
    NOA_GENERATE_PROCLAIM_FULL_ND(vec_of_size);

    template<typename, typename> struct proclaim_is_vec_of_type : std::false_type {};
    template<typename T, typename U> using is_vec_of_type = proclaim_is_vec_of_type<std::remove_cv_t<T>, U>;
    template<typename T, typename U> constexpr bool is_vec_of_type_v = is_vec_of_type<T, U>::value;
    template<typename T, typename U> concept vec_of_type = vec<T> and is_vec_of_type_v<T, U>;
    template<typename T, typename... U> using are_vec_of_type = conjunction_or_false<is_vec_of_type<T, U>...>;
    template<typename T, typename... U> constexpr bool are_vec_of_type_v = are_vec_of_type<T, U...>::value;

    #define NOA_TRAITS_VEC_(name)                                                                                                           \
    template<typename T> using is_vec_##name = std::conjunction<is_vec<T>, is_##name<value_type_t<T>>>;                                     \
    template<typename T> constexpr bool is_vec_##name##_v = is_vec_##name<T>::value;                                                        \
    template<typename... T> concept vec_##name = vec<T...> and name<value_type_t<T>...>;                                                    \
    template<typename... T> constexpr bool are_vec_##name##_v = conjunction_or_false<is_vec_##name<T>...>::value;                           \
    template<typename T, size_t... N> using is_vec_##name##_size = std::conjunction<is_vec_of_size<T, N...>, is_##name<value_type_t<T>>>;   \
    template<typename T, size_t... N> constexpr bool is_vec_##name##_size_v = is_vec_##name##_size<T, N...>::value;                         \
    template<typename T, size_t... N> concept vec_##name##_size = vec_of_size<T, N...> and name<value_type_t<T>>;                           \
    template<size_t N, typename... T> constexpr bool are_vec_##name##_size_v = conjunction_or_false<is_vec_##name##_size<T, N>...>::value

    NOA_TRAITS_VEC_(integer);
    NOA_TRAITS_VEC_(sinteger);
    NOA_TRAITS_VEC_(uinteger);
    NOA_TRAITS_VEC_(boolean);
    NOA_TRAITS_VEC_(real);
    NOA_TRAITS_VEC_(scalar);
    NOA_TRAITS_VEC_(numeric);
    #undef NOA_TRAITS_VEC_

    // Tuple and Pair
    NOA_GENERATE_PROCLAIM_FULL(pair);
    NOA_GENERATE_PROCLAIM_FULL(tuple);
    NOA_GENERATE_PROCLAIM_FULL(empty_tuple);

    template<typename T> using is_pair_decay = is_pair<std::decay_t<T>>;
    template<typename T> using is_tuple_decay = is_tuple<std::decay_t<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(pair_decay);
    NOA_GENERATE_PROCLAIM_UTILS(tuple_decay);
    template<typename... T> concept pair_decay = are_pair_decay_v<T...>;
    template<typename... T> concept tuple_decay = are_tuple_decay_v<T...>;

    NOA_GENERATE_PROCLAIM_FULL(trivial_zero);
    template<numeric T> struct proclaim_is_trivial_zero<T> : std::true_type {};
    template<vec T> struct proclaim_is_trivial_zero<T> : std::true_type {};
}

namespace noa::inline types {
    template<typename, size_t, size_t>
    class Vec;
}

namespace noa::traits {
    template<typename From, typename To>
    concept static_castable_to = requires(From v) { static_cast<To>(v); };

    template<typename From, typename To>
    concept compatible_types = requires(From v) { static_cast<To>(v); };
}
