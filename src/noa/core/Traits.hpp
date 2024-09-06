#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Namespace.hpp"

#ifdef NOA_IS_OFFLINE
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
    inline constexpr auto operator+(TypeList<Ls...>, TypeList<Rs...>) {
        return TypeList<Ls..., Rs...>{};
    }

    namespace guts {
        template<typename, typename>
        struct append_to_type_seq {};

        template<typename T, typename... Ts, template<typename...> class TT>
        struct append_to_type_seq<T, TT<Ts...>> {
            using type = TT<Ts..., T>;
        };
    }

    template<typename T, size_t N, template<typename...> class TT>
    struct repeat {
        using type = guts::append_to_type_seq<T, typename repeat<T, N - 1, TT>::type>::type;
    };

    template<typename T, template<typename...> class TT>
    struct repeat<T, 0, TT> {
        using type = TT<>;
    };

    template<typename T, size_t N, template<typename...> class L>
    using repeat_t = repeat<T, N, L>::type;
}

namespace noa::traits {
    #define NOA_TRAITS_(name, default_type, special_type)       \
    namespace guts {                                            \
        template<typename T, typename = void>                   \
        struct name {                                           \
            using type = default_type;                          \
        };                                                      \
        template<typename T>                                    \
        struct name<T, std::void_t<typename special_type>> {    \
            using type = special_type;                          \
        };                                                      \
    }                                                           \
    template<typename T>                                        \
    struct name {                                               \
        using type = guts::name<std::decay_t<T>>::type;         \
    };                                                          \
    template<typename T>                                        \
    using name##_t = name<T>::type

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
    template<typename T, typename... U> struct are_same : std::conjunction<std::is_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool are_same_v = are_same<T, U...>::value;
    template<typename T, typename... U> concept same_as = (std::same_as<T, U> and ...);

    template<typename T, typename U> struct is_almost_same : std::is_same<std::decay_t<T>, std::decay_t<U>> {};
    template<typename T, typename U> constexpr bool is_almost_same_v = is_almost_same<T, U>::value;
    template<typename T, typename... U> struct are_almost_same : std::conjunction<is_almost_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool are_almost_same_v = are_almost_same<T, U...>::value;
    template<typename T, typename... U> concept almost_same_as = (std::same_as<std::decay_t<T>, std::decay_t<U>> and ...);

    template<typename T, typename... U> struct is_any : std::disjunction<std::is_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool is_any_v = is_any<T, U...>::value;
    template<typename T, typename... U> concept any_of = (same_as<T, U> or ...);

    template<typename T, typename... U> struct is_almost_any : std::disjunction<is_almost_same<T, U>...> {};
    template<typename T, typename... U> constexpr bool is_almost_any_v = is_almost_any<T, U...>::value;
    template<typename T, typename... U> concept almost_any_of = (almost_same_as<T, U> or ...);

    template<typename T, typename U> concept same_as_value_type_of = same_as<T, nt::value_type_t<U>>;
    template<typename T, typename U> concept same_as_mutable_value_type_of = same_as<T, nt::mutable_value_type_t<U>>;
    template<typename T, typename U> concept almost_same_as_value_type_of = almost_same_as<T, nt::value_type_t<U>>;
    template<typename T, typename U> concept almost_same_as_mutable_value_type_of = almost_same_as<T, nt::mutable_value_type_t<U>>;

    template<typename T, typename... U> concept same_value_type = (same_as<nt::value_type_t<T>, nt::value_type_t<U>> and ...);
    template<typename T, typename... U> concept same_mutable_value_type = (same_as<nt::mutable_value_type_t<T>, nt::mutable_value_type_t<U>> and ...);

    template<typename Input, typename Output>
    concept mutable_of = std::same_as<Input, std::remove_const_t<Output>> and std::is_const_v<Output>;
}

namespace noa::traits {
    template<typename, size_t S, typename = void>
    struct size_or {
        static constexpr size_t value = S;
    };
    template<typename T, size_t S>
    struct size_or<T, S, std::void_t<decltype(std::remove_reference_t<T>::SIZE)>> {
        static constexpr size_t value = std::remove_reference_t<T>::SIZE;
    };
    template<typename T, size_t S>
    constexpr size_t size_or_v = size_or<T, S>::value;

    template<typename... Ts>
    struct have_same_size {
        static constexpr size_t N = nt::first_t<std::remove_reference_t<Ts>...>::SIZE;
        static constexpr bool value = ((N == std::remove_reference_t<Ts>::SIZE) and ...);
    };
    template<typename T>
    struct have_same_size<T> {
        static constexpr size_t N = nt::first_t<std::remove_reference_t<T>>::SIZE;
        static constexpr bool value = true;
    };
    template<>
    struct have_same_size<> {
        static constexpr size_t N = 0;
        static constexpr bool value = true;
    };
    template<typename... Ts>
    constexpr bool have_same_size_v = have_same_size<Ts...>::value;
}

#define NOA_GENERATE_PROCLAIM_UTILS(name)                                           \
    template<typename T> constexpr bool is_##name##_v = is_##name<T>::value;        \
    template<typename... T> using are_##name = std::conjunction<is_##name<T>...>;   \
    template<typename... T> constexpr bool are_##name##_v = are_##name<T...>::value

#define NOA_GENERATE_PROCLAIM(name)                                                 \
    template<typename> struct proclaim_is_##name : std::false_type {};              \
    template<typename T> using is_##name = proclaim_is_##name<std::remove_cv_t<T>>; \
    NOA_GENERATE_PROCLAIM_UTILS(name)

#define NOA_GENERATE_PROCLAIM_FULL(name)    \
    NOA_GENERATE_PROCLAIM(name);            \
    template<typename... T> concept name = are_##name##_v<T...>

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

    template<typename T, bool = is_integer_v<T>> struct proclaim_is_sinteger : public std::false_type {};
    template<typename T> struct proclaim_is_sinteger<T, true> : public std::bool_constant<T(-1) < T(0)> {};
    template<typename T> struct is_sinteger : public proclaim_is_sinteger<T>::type {};
    NOA_GENERATE_PROCLAIM_UTILS(sinteger);
    template<typename... T> concept sinteger = integer<T...> and are_sinteger_v<T...>;

    template<typename T> struct is_uinteger : public std::conjunction<is_integer<T>, std::negation<is_sinteger<T>>>::type{};
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
    template<typename... T> using are_pointer_##name = std::conjunction<is_pointer_##name<T>...>;                       \
    template<typename... T> constexpr bool are_pointer_##name##_v = are_pointer_##name<T...>::value;                    \
    template<typename... T> concept pointer_##name = pointer<T...> and are_pointer_##name##_v<T...>
    NOA_TRAITS_POINTER(integer);
    NOA_TRAITS_POINTER(sinteger);
    NOA_TRAITS_POINTER(uinteger);
    NOA_TRAITS_POINTER(boolean);
    NOA_TRAITS_POINTER(real);
    NOA_TRAITS_POINTER(scalar);
    NOA_TRAITS_POINTER(numeric);
}

namespace noa::traits {
    // Accessor
    NOA_GENERATE_PROCLAIM_FULL(accessor);

    NOA_GENERATE_PROCLAIM(accessor_restrict);
    NOA_GENERATE_PROCLAIM(accessor_contiguous);
    NOA_GENERATE_PROCLAIM(accessor_pure);
    NOA_GENERATE_PROCLAIM(accessor_reference);
    NOA_GENERATE_PROCLAIM(accessor_value);
    template<typename... T> concept accessor_restrict = accessor<T...> and are_accessor_restrict_v<T...>;
    template<typename... T> concept accessor_contiguous = accessor<T...> and are_accessor_contiguous_v<T...>;
    template<typename... T> concept accessor_pure = accessor<T...> and are_accessor_pure_v<T...>;
    template<typename... T> concept accessor_reference = accessor<T...> and are_accessor_reference_v<T...>;
    template<typename... T> concept accessor_value = accessor<T...> and are_accessor_value_v<T...>;

    template<typename T, size_t N> struct proclaim_is_accessor_nd : std::false_type {};
    template<typename T, size_t... N> using is_accessor_nd = std::disjunction<proclaim_is_accessor_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_accessor_nd_v = is_accessor_nd<T, N...>::value;
    template<typename T, size_t... N> concept accessor_nd = accessor<T> and is_accessor_nd_v<T, N...>;
    template<size_t N, typename... T> using are_accessor_nd = std::conjunction<is_accessor_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_accessor_nd_v = are_accessor_nd<N, T...>::value;

    template<typename T, size_t... N> using is_accessor_pure_nd = std::conjunction<is_accessor_pure<T>, is_accessor_nd<T, N...>>;
    template<typename T, size_t... N>  constexpr bool is_accessor_pure_nd_v = is_accessor_pure_nd<T, N...>::value;
    template<typename T, size_t... N> concept accessor_pure_nd = accessor_nd<T, N...> and accessor_pure<T>;
    template<size_t N, typename... T> using are_accessor_pure_nd = std::conjunction<is_accessor_pure_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_accessor_pure_nd_v = are_accessor_pure_nd<N, T...>::value;

    // Mat
    NOA_GENERATE_PROCLAIM_FULL(mat);

    template<typename, size_t, size_t> struct proclaim_is_mat_of_shape : std::false_type {};
    template<typename T, size_t R, size_t C> using is_mat_of_shape = proclaim_is_mat_of_shape<std::remove_cv_t<T>, R, C>;
    template<typename T, size_t R, size_t C> constexpr bool is_mat_of_shape_v = is_mat_of_shape<T, R, C>::value;
    template<typename T, size_t R, size_t C> concept mat_of_shape = mat<T> and is_mat_of_shape_v<T, R, C>;
    template<size_t R, size_t C, typename... T> using are_mat_of_shape = std::conjunction<is_mat_of_shape<T, R, C>...>;
    template<size_t R, size_t C, typename... T> constexpr bool are_mat_of_shape_v = are_mat_of_shape<R, C, std::decay_t<T>...>::value;

    #define NOA_TRAITS_MAT_(R, C)                                                   \
    template<typename T> using is_mat##R##C = is_mat_of_shape<T, R, C>::value;      \
    template<typename T> constexpr bool is_mat##R##C##_v = is_mat##R##C<T>::value;  \
    template<typename T> concept mat##R##C = mat_of_shape<T, R, C>
    NOA_TRAITS_MAT_(2, 2);
    NOA_TRAITS_MAT_(2, 3);
    NOA_TRAITS_MAT_(3, 3);
    NOA_TRAITS_MAT_(3, 4);
    NOA_TRAITS_MAT_(4, 4);

    // Vec
    NOA_GENERATE_PROCLAIM_FULL(vec);

    template<typename, typename> struct proclaim_is_vec_of_type : std::false_type {};
    template<typename T, typename U> using is_vec_of_type = proclaim_is_vec_of_type<std::remove_cv_t<T>, U>;
    template<typename T, typename U> constexpr bool is_vec_of_type_v = is_vec_of_type<T, U>::value;
    template<typename T, typename U> concept vec_of_type = vec<T> and is_vec_of_type_v<T, U>;
    template<typename T, typename... U> using are_vec_of_type = std::conjunction<is_vec_of_type<T, U>...>;
    template<typename T, typename... U> constexpr bool are_vec_of_type_v = are_vec_of_type<T, U...>::value;

    template<typename, size_t> struct proclaim_is_vec_of_size : std::false_type {};
    template<typename T, size_t... N> using is_vec_of_size = std::disjunction<proclaim_is_vec_of_size<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_vec_of_size_v = is_vec_of_size<T, N...>::value;
    template<typename T, size_t... N> concept vec_of_size = vec<T> and is_vec_of_size_v<T, N...>;
    template<size_t N, typename... T> using are_vec_of_size = std::conjunction<is_vec_of_size<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_vec_of_size_v = are_vec_of_size<N, T...>::value;

    #define NOA_TRAITS_VEC_(name)                                                                                                           \
    template<typename T> using is_vec_##name = std::conjunction<is_vec<T>, is_##name<value_type_t<T>>>;                                     \
    template<typename T> constexpr bool is_vec_##name##_v = is_vec_##name<T>::value;                                                        \
    template<typename... T> concept vec_##name = vec<T...> and name<value_type_t<T>...>;                                                    \
    template<typename... T> constexpr bool are_vec_##name##_v = std::conjunction<is_vec_##name<T>...>::value;                               \
    template<typename T, size_t... N> using is_vec_##name##_size = std::conjunction<is_vec_of_size<T, N...>, is_##name<value_type_t<T>>>;   \
    template<typename T, size_t... N> constexpr bool is_vec_##name##_size_v = is_vec_##name##_size<T, N...>::value;                         \
    template<typename T, size_t... N> concept vec_##name##_size = vec_of_size<T, N...> and name<value_type_t<T>>;                           \
    template<size_t N, typename... T> constexpr bool are_vec_##name##_size_v = std::conjunction<is_vec_##name##_size<T, N>...>::value

    NOA_TRAITS_VEC_(integer);
    NOA_TRAITS_VEC_(sinteger);
    NOA_TRAITS_VEC_(uinteger);
    NOA_TRAITS_VEC_(boolean);
    NOA_TRAITS_VEC_(real);
    NOA_TRAITS_VEC_(scalar);
    NOA_TRAITS_VEC_(numeric);
    #undef NOA_TRAITS_VEC_

    // Shape and Strides
    #define NOA_TRAITS_SHAPE_(name)                                                                                                         \
    NOA_GENERATE_PROCLAIM_FULL(name);                                                                                                       \
    template<typename, typename> struct proclaim_is_##name##_of_type : std::false_type {};                                                  \
    template<typename T, typename U> using is_##name##_of_type = proclaim_is_##name##_of_type<std::remove_cv_t<T>, U>;                      \
    template<typename T, typename U> constexpr bool is_##name##_of_type_v = is_##name##_of_type<T, U>::value;                               \
    template<typename T, typename U> concept name##_of_type = name<T> and is_##name##_of_type_v<T, U>;                                      \
    template<typename, size_t> struct proclaim_is_##name##_of_size : std::false_type {};                                                    \
    template<typename T, size_t... N> using is_##name##_of_size = std::disjunction<proclaim_is_##name##_of_size<std::remove_cv_t<T>, N>...>;\
    template<typename T, size_t... N> constexpr bool is_##name##_of_size_v = is_##name##_of_size<T, N...>::value;                           \
    template<typename T, size_t... N> concept name##_of_size = name<T> and is_##name##_of_size_v<T, N...>
    NOA_TRAITS_SHAPE_(shape);
    NOA_TRAITS_SHAPE_(strides);

    template<typename T> using is_shape_or_strides = std::disjunction<is_strides<T>, is_shape<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(shape_or_strides);
    template<typename... T> concept shape_or_strides = ((shape<T> or strides<T>) and ...);

    template<typename T> using is_vec_shape_or_strides = std::disjunction<is_strides<T>, is_shape<T>, is_vec<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(vec_shape_or_strides);
    template<typename... T> concept vec_shape_or_strides = ((shape<T> or strides<T> or vec<T>) and ...);

    template<typename T, size_t... N> concept vec_shape_or_strides_of_size =
            vec_of_size<T, N...> or shape_of_size<T, N...> or strides_of_size<T, N...>;

    // Tuple and Pair
    NOA_GENERATE_PROCLAIM_FULL(pair);
    NOA_GENERATE_PROCLAIM_FULL(tuple);
    NOA_GENERATE_PROCLAIM_FULL(empty_tuple);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_pure);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_reference);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_value);
    NOA_GENERATE_PROCLAIM_FULL(tuple_of_accessor_or_empty);

    template<typename T> using is_pair_decay = is_pair<std::decay_t<T>>;
    template<typename T> using is_tuple_decay = is_tuple<std::decay_t<T>>;
    NOA_GENERATE_PROCLAIM_UTILS(pair_decay);
    NOA_GENERATE_PROCLAIM_UTILS(tuple_decay);
    template<typename... T> concept pair_decay = are_pair_decay_v<T...>;
    template<typename... T> concept tuple_decay = are_tuple_decay_v<T...>;

    template<typename T, size_t N> struct proclaim_is_tuple_of_accessor_nd : std::false_type {};
    template<typename T, size_t... N> using is_tuple_of_accessor_nd = std::disjunction<proclaim_is_tuple_of_accessor_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_tuple_of_accessor_nd_v = is_tuple_of_accessor_nd<T, N...>::value;
    template<typename T, size_t... N> concept tuple_of_accessor_nd = is_tuple_of_accessor_nd_v<T, N...>;
    template<size_t N, typename... T> using are_tuple_of_accessor_nd = std::conjunction<is_tuple_of_accessor_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_tuple_of_accessor_nd_v = are_tuple_of_accessor_nd<N, T...>::value;

    // Geometry
    NOA_GENERATE_PROCLAIM_FULL(quaternion);
    template<typename... T> concept quaternion_f32 = quaternion<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept quaternion_f64 = quaternion<T...> and same_as<f64, value_type_t<T>...>;

    NOA_GENERATE_PROCLAIM_FULL(interpolator);
    template<typename T, size_t N> struct proclaim_is_interpolator_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_nd = std::disjunction<proclaim_is_interpolator_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_nd_v = is_interpolator_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_nd = interpolator<T> and is_interpolator_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_nd = std::conjunction<is_interpolator_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_nd_v = are_interpolator_nd<N, T...>::value;

    NOA_GENERATE_PROCLAIM_FULL(interpolator_spectrum);
    template<typename T, size_t N> struct proclaim_is_interpolator_spectrum_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_spectrum_nd = std::disjunction<proclaim_is_interpolator_spectrum_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_spectrum_nd_v = is_interpolator_spectrum_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_spectrum_nd = interpolator_spectrum<T> and is_interpolator_spectrum_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_spectrum_nd = std::conjunction<is_interpolator_spectrum_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_spectrum_nd_v = are_interpolator_spectrum_nd<N, T...>::value;

    template<typename... T> concept interpolator_or_empty = ((interpolator<T> or empty<T>) and ...);
    template<typename... T> concept interpolator_spectrum_or_empty = ((interpolator_spectrum<T> or empty<T>) and ...);
    template<typename T, size_t... N> concept interpolator_nd_or_empty = interpolator_nd<T, N...> or empty<T>;
    template<typename T, size_t... N> concept interpolator_spectrum_nd_or_empty = interpolator_spectrum_nd<T, N...> or empty<T>;

    // CTF
    NOA_GENERATE_PROCLAIM_FULL(ctf);
    NOA_GENERATE_PROCLAIM(ctf_isotropic);
    NOA_GENERATE_PROCLAIM(ctf_anisotropic);
    template<typename... T> concept ctf_isotropic = ctf<T...> and are_ctf_isotropic_v<T...>;
    template<typename... T> concept ctf_anisotropic = ctf<T...> and are_ctf_anisotropic_v<T...>;

    template<typename... T> concept ctf_f32 = ctf<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept ctf_f64 = ctf<T...> and same_as<f64, value_type_t<T>...>;
    template<typename... T> concept ctf_isotropic_f32 = ctf_isotropic<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept ctf_isotropic_f64 = ctf_isotropic<T...> and same_as<f64, value_type_t<T>...>;
    template<typename... T> concept ctf_anisotropic_f32 = ctf_anisotropic<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept ctf_anisotropic_f64 = ctf_anisotropic<T...> and same_as<f64, value_type_t<T>...>;

    NOA_GENERATE_PROCLAIM_FULL(batched_parameter);
}

namespace noa::inline types {
    template<typename T, size_t N, size_t A>
    class Vec;
}

namespace noa::traits {
    template<typename From, typename To>
    concept static_castable_to = requires(From v) { static_cast<To>(v); };

    template<typename From, typename To>
    concept compatible_types = requires(From v) { static_cast<To>(v); };

    template<typename From, typename To>
    concept spectrum_types = nt::complex<From, To> or nt::real<From, To> or (nt::complex<From> and nt::real<To>);

    template<typename From, typename To>
    concept compatible_or_spectrum_types = (nt::complex<From> and nt::real<To>) or static_castable_to<From, To>;

    template<typename T, size_t N>
    concept indexer_compatible =
            nt::integer<typename T::index_type> and
            std::convertible_to<decltype(T::SIZE), size_t> and
            std::convertible_to<decltype(std::declval<const T&>().template stride<N - 1>()), size_t>;

    template<size_t N, typename... T>
    concept offset_indexing =
        ((sizeof...(T) <= N and integer<T...>) or
         (sizeof...(T) == 1 and vec_integer<T...> and nt::size_or_v<T..., 0> <= N));

    template<size_t N, typename... T>
    concept iwise_general_indexing =
        ((sizeof...(T) == N and integer<T...>) or
         (sizeof...(T) == 1 and nt::vec_integer<T...> and nt::are_vec_of_size_v<N, T...>));

    template<size_t N, typename I, typename... T>
    concept iwise_core_indexing =
        ((sizeof...(T) == N and same_as<I, T...>) or
         (sizeof...(T) == 1 and nt::are_vec_of_type_v<I, T...> and nt::are_vec_of_size_v<N, T...>));

    namespace guts {
        template<typename T, size_t... N>
        struct atomic_addable_nd_t {
            template<size_t S, i32... J>
            static consteval bool has_op(std::integer_sequence<i32, J...>) {
                return requires(const T& t) {
                    requires std::is_pointer_v<decltype(t.get())>;
                    requires numeric<std::remove_pointer_t<decltype(t.get())>>;
                    requires same_as<decltype(t.offset_pointer(t.get(), J...)), decltype(t.get())>;
                    requires same_as<decltype(t.offset_pointer(t.get(), t(Vec<i32, S, 0>{}))), decltype(t.get())>;
                };
            }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<i32, N>{}) or ...);
        };

        template<typename T, typename I, size_t... N>
        struct readable_nd_t {
            template<size_t S, I... J>
            static consteval bool has_op(std::integer_sequence<I, J...>) {
                return std::convertible_to<decltype(std::declval<const T&>()(J...)), mutable_value_type_t<T>> and
                       std::convertible_to<decltype(std::declval<const T&>()(Vec<I, S, 0>{})), mutable_value_type_t<T>>;
            }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<I, N>{}) or ...);
        };

        template<typename T, typename I, size_t... N>
        struct writable_nd_t {
            template<size_t S, I... J>
            static consteval bool has_op(std::integer_sequence<I, J...>) {
                return std::same_as<decltype(std::declval<T&>()(J...)), reference_type_t<T>> and
                       std::same_as<decltype(std::declval<T&>()(Vec<I, S, 0>{})), reference_type_t<T>>;
            }
            static constexpr bool value = (has_op<N>(std::make_integer_sequence<I, N>{}) or ...);
        };
    }

    template<typename T, size_t... N>
    concept readable_nd = std::copyable<std::remove_cv_t<T>> and requires {
        typename T::value_type;
        typename T::index_type;
        requires integer<decltype(T::SIZE)>;
    } and guts::readable_nd_t<T, typename T::index_type, N...>::value;

    template<typename T, size_t... N>
    concept readable_nd_or_empty = readable_nd<T, N...> or nt::empty<T>;

    template<typename T, size_t... N>
    concept readable_nd_optional = readable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept writable_nd = readable_nd<T, N...> and
            not std::is_const_v<value_type_t<T>> and
            not std::is_const_v<mutable_value_type_t<T>> and
            guts::writable_nd_t<T, typename T::index_type, N...>::value;

    template<typename T, size_t... N>
    concept writable_nd_optional = writable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept writable_nd_or_empty = writable_nd<T, N...> or nt::empty<T>;

    template<typename T, size_t... N>
    concept atomic_addable_nd =
            std::copyable<std::remove_cv_t<T>> and
            numeric<typename T::value_type> and
            guts::atomic_addable_nd_t<T, N...>::value;

    template<typename T, size_t... N>
    concept atomic_addable_nd_optional = atomic_addable_nd<T, N...> and static_castable_to<decltype(std::declval<const T&>()), bool>;

    template<typename T, size_t... N>
    concept atomic_addable_nd_or_empty = atomic_addable_nd<T, N...> or nt::empty<T>;

    template<typename T>
    concept readable_pointer_like =
            std::copyable<std::remove_cv_t<T>> and
            std::convertible_to<decltype(std::declval<const T&>()[size_t{}]), mutable_value_type_t<T>>;

    template<typename T>
    concept writable_pointer_like =
            readable_pointer_like<T> and
            std::same_as<decltype(std::declval<T&>()[size_t{}]), reference_type_t<T>>;

    template<typename T> concept marked_contiguous = same_as<std::remove_cv_t<decltype(T::IS_CONTIGUOUS)>, bool> and T::IS_CONTIGUOUS == true;
    template<typename T> concept marked_restrict =   same_as<std::remove_cv_t<decltype(T::IS_RESTRICT)>,   bool> and T::IS_RESTRICT   == true;
}
