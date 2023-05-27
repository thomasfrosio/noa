#pragma once

#include <type_traits>
#include <utility>

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

    struct Empty {};
    using empty_t = Empty;

    template<typename T>
    constexpr auto to_underlying(T value) noexcept {
        return static_cast<std::underlying_type_t<T>>(value);
    }
}

namespace noa::traits {
    namespace details {
        template<typename T, typename = void> struct value_type { using type = T; };
        template<typename T, typename = void> struct mutable_value_type { using type = T; };
        template<typename T, typename = void> struct element_type { using type = T; };
        template<typename T, typename = void> struct index_type { using type = T; };
        template<typename T, typename = void> struct shared_type { using type = T; };
        template<typename T, typename = void> struct pointer_type { using type = T; };

        template<typename T> struct value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };
        template<typename T> struct mutable_value_type<T, std::void_t<typename T::mutable_value_type>> { using type = typename T::mutable_value_type; };
        template<typename T> struct element_type<T, std::void_t<typename T::element_type>> { using type = typename T::element_type; };
        template<typename T> struct index_type<T, std::void_t<typename T::index_type>> { using type = typename T::index_type; };
        template<typename T> struct pointer_type<T, std::void_t<typename T::pointer_type>> { using type = typename T::pointer_type; };
        template<typename T> struct shared_type<T, std::void_t<typename T::shared_type>> { using type = typename T::shared_type; };
    }

    template<typename T> struct value_type { using type = typename details::value_type<T>::type; };
    template<typename T> struct mutable_value_type { using type = typename details::mutable_value_type<T>::type; };
    template<typename T> struct element_type { using type = typename details::element_type<T>::type; };
    template<typename T> struct index_type { using type = typename details::index_type<T>::type; };
    template<typename T> struct pointer_type { using type = typename details::pointer_type<T>::type; };
    template<typename T> struct shared_type { using type = typename details::shared_type<T>::type; };

    template<typename T> using value_type_t = typename value_type<remove_ref_cv_t<T>>::type;
    template<typename T> using value_type_twice_t = value_type_t<value_type_t<T>>;
    template<typename T> using mutable_value_type_t = typename mutable_value_type<remove_ref_cv_t<T>>::type;
    template<typename T> using element_type_t = typename element_type<remove_ref_cv_t<T>>::type;
    template<typename T> using index_type_t = typename index_type<remove_ref_cv_t<T>>::type;
    template<typename T> using pointer_type_t = typename pointer_type<remove_ref_cv_t<T>>::type;
    template<typename T> using shared_type_t = typename shared_type<remove_ref_cv_t<T>>::type;
}

namespace noa::traits {
    template<typename T1, typename T2> using is_almost_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;
    template<typename T1, typename T2> constexpr bool is_almost_same_v = is_almost_same<T1, T2>::value;

    template<typename T, typename... Ts> struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
    template<typename T, typename... Ts> constexpr bool is_any_v = is_any<T, Ts...>::value;

    template<typename T, typename... Ts> struct is_almost_any : std::bool_constant<(is_almost_same_v<T, Ts> || ...)> {};
    template<typename T, typename... Ts> constexpr bool is_almost_any_v = is_almost_any<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_all_same : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_all_same_v = are_all_same<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_almost_all_same : std::bool_constant<(is_almost_same_v<T, Ts> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_almost_all_same_v = are_almost_all_same<T, Ts...>::value;
}

// From https://en.cppreference.com/w/cpp/experimental/is_detected
// And https://stackoverflow.com/a/41936999
namespace noa::traits {
    namespace details {
        struct nonesuch {
            nonesuch() = delete;
            ~nonesuch() = delete;
            nonesuch(nonesuch const&) = delete;
            void operator=(nonesuch const&) = delete;
        };

        template<class Default, class AlwaysVoid, template<class...> class Op, class... Args>
        struct detector {
            using value_t = std::false_type;
            using type = Default;
        };
        template<class Default, template<class...> class Op, class... Args>
        struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
            using value_t = std::true_type;
            using type = Op<Args...>;
        };
    }

    template<template<class...> class Op, class... Args>
    using is_detected = typename details::detector<details::nonesuch, void, Op, Args...>::value_t;

    template<template<class...> class Op, class... Args>
    using detected_t = typename details::detector<details::nonesuch, void, Op, Args...>::type;

    template<class Default, template<class...> class Op, class... Args>
    using detected_or = details::detector<Default, void, Op, Args...>;

    template< template<class...> class Op, class... Args >
    constexpr inline bool is_detected_v = is_detected<Op, Args...>::value;

    template< class Default, template<class...> class Op, class... Args >
    using detected_or_t = typename detected_or<Default, Op, Args...>::type;

    template <class Expected, template<class...> class Op, class... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

    template <class Expected, template<class...> class Op, class... Args>
    constexpr inline bool is_detected_exact_v =
            is_detected_exact<Expected, Op, Args...>::value;

    template <class To, template<class...> class Op, class... Args>
    using is_detected_convertible =
            std::is_convertible<detected_t<Op, Args...>, To>;

    template <class To, template<class...> class Op, class... Args>
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
