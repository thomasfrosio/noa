#pragma once

#include <type_traits>

namespace noa::traits {
    template<typename T> struct remove_ref_cv { using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>; };
    template<typename T> using remove_ref_cv_t = typename remove_ref_cv<T>::type;

    template<typename T> struct remove_pointer_cv { using type = typename std::remove_cv_t<typename std::remove_pointer_t<T>>; };
    template<typename T> using remove_pointer_cv_t = typename remove_pointer_cv<T>::type;

    template <bool... Bs> using bool_sequence_t = std::integer_sequence<bool, Bs...>;
    template <bool... Bs> using bool_and = std::is_same<bool_sequence_t<Bs...>, bool_sequence_t<(Bs || true)...>>;
    template <bool... Bs> using bool_or = std::integral_constant<bool, !bool_and<!Bs...>::value>;
}

namespace noa::traits {
    namespace details {
        template<typename T, typename = void> struct value_type { using type = T; };
        template<typename T> struct value_type<T, std::void_t<typename T::value_type>> { using type = typename T::value_type; };

        template<typename T, typename = void> struct element_type { using type = T; };
        template<typename T> struct element_type<T, std::void_t<typename T::element_type>> { using type = typename T::element_type; };

        template<typename T, typename = void> struct index_type { using type = T; };
        template<typename T> struct index_type<T, std::void_t<typename T::index_type>> { using type = typename T::index_type; };

        template<typename T, typename = void> struct ptr_type { using type = T; };
        template<typename T> struct ptr_type<T, std::void_t<typename T::ptr_type>> { using type = typename T::ptr_type; };

        template<typename T, typename = void> struct shared_type { using type = T; };
        template<typename T> struct shared_type<T, std::void_t<typename T::shared_type>> { using type = typename T::shared_type; };
    }

    template<typename T> struct value_type { using type = typename details::value_type<T>::type; };
    template<typename T> struct element_type { using type = typename details::element_type<T>::type; };
    template<typename T> struct index_type { using type = typename details::index_type<T>::type; };
    template<typename T> struct ptr_type { using type = typename details::ptr_type<T>::type; };
    template<typename T> struct shared_type { using type = typename details::shared_type<T>::type; };

    template<typename T> using value_type_t = typename value_type<remove_ref_cv_t<T>>::type;
    template<typename T> using element_type_t = typename element_type<remove_ref_cv_t<T>>::type;
    template<typename T> using index_type_t = typename index_type<remove_ref_cv_t<T>>::type;
    template<typename T> using ptr_type_t = typename ptr_type<remove_ref_cv_t<T>>::type;
    template<typename T> using shared_type_t = typename shared_type<remove_ref_cv_t<T>>::type;
}

namespace noa::traits {
    template<typename T1, typename T2> using is_almost_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;
    template<typename T1, typename T2> constexpr bool is_almost_same_v = is_almost_same<T1, T2>::value;

    template<typename T, typename... Ts> struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
    template<typename T, typename... Ts> constexpr bool is_any_v = is_any<T, Ts...>::value;

    template<typename T, typename... Ts> struct are_all_same : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};
    template<typename T, typename... Ts> constexpr bool are_all_same_v = are_all_same<T, Ts...>::value;

    template<typename T> using always_false = std::false_type;
    template<typename T> constexpr bool always_false_v = always_false<T>::value;
}
