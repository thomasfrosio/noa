#pragma once

#include <type_traits>

namespace noa::traits {
    template<typename T>
    struct remove_ref_cv {
        using type = typename std::remove_cv_t<typename std::remove_reference_t<T>>;
    };

    /// Removes the const/volatile and reference from T.
    template<typename T>
    using remove_ref_cv_t = typename remove_ref_cv<T>::type;
}

namespace noa::traits {
    namespace details {
        template<typename T, typename = void>
        struct value_type {
            using type = T;
        };
        template<typename T>
        struct value_type<T, std::void_t<typename T::value_type>> {
            using type = typename T::value_type; // TODO Add value_t
        };

        template<typename T, typename = void>
        struct element_type {
            using type = T;
        };
        template<typename T>
        struct element_type<T, std::void_t<typename T::element_type>> {
            using type = typename T::element_type; // TODO Add element_t
        };

        template<typename T, typename = void>
        struct ptr_type {
            using type = T;
        };
        template<typename T>
        struct ptr_type<T, std::void_t<typename T::ptr_type>> {
            using type = typename T::ptr_type; // TODO Add ptr_t
        };
    }

    template<typename T>
    struct value_type {
        using type = typename details::value_type<T>::type;
    };
    template<typename T>
    struct element_type {
        using type = typename details::element_type<T>::type;
    };
    template<typename T>
    struct ptr_type {
        using type = typename details::ptr_type<T>::type;
    };

    /// Extracts the typedef "value_type" from T if it exists, returns T otherwise. \c remove_ref_cv_t is applied to T.
    template<typename T> using value_type_t = typename value_type<remove_ref_cv_t<T>>::type;

    /// Extracts the typedef element_type from T if it exists, returns T otherwise. \c remove_ref_cv_t is applied to T.
    template<typename T> using element_type_t = typename element_type<remove_ref_cv_t<T>>::type;

    /// Extracts the typedef ptr_type from T if it exists, returns T otherwise. \c remove_ref_cv_t is applied to T.
    template<typename T> using ptr_type_t = typename ptr_type<remove_ref_cv_t<T>>::type;
}

namespace noa::traits {
    template<typename T1, typename T2>
    using is_almost_same = std::bool_constant<std::is_same_v<remove_ref_cv_t<T1>, remove_ref_cv_t<T2>>>;

    /// Whether \a T1 and \a T2 are the same types, ignoring const/volatile and reference.
    template<typename T1, typename T2>
    inline constexpr bool is_almost_same_v = is_almost_same<T1, T2>::value;
}

namespace noa::traits {
    template<class T, class... Ts>
    struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

    /// Whether \p T1 is the same type as any of the \p Ts types.
    template<typename T1, typename... Ts>
    inline constexpr bool is_any_v = is_any<T1, Ts...>::value;
}

namespace noa::traits {
    template<class T, class... Ts>
    struct are_all_same : std::bool_constant<(std::is_same_v<T, Ts> && ...)> {};

    /// Whether \p T1 and the \p Ts types are all the same type.
    template<typename T1, typename... Ts>
    inline constexpr bool are_all_same_v = are_all_same<T1, Ts...>::value;
}

namespace noa::traits {
    template<typename T> using always_false = std::false_type;

    /// Always false. Used to invalidate some code paths at compile time.
    template<typename T> inline constexpr bool always_false_v = always_false<T>::value;
}
