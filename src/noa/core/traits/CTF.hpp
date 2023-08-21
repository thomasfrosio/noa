#pragma once

#include <type_traits>
#include "noa/core/traits/Utilities.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_ctf : std::false_type {};
    template<typename T> using is_ctf = std::bool_constant<proclaim_is_ctf<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_v = is_ctf<T>::value;
    template<typename... Ts> constexpr bool are_ctf_v = bool_and<is_ctf_v<Ts>...>::value;

    template<typename T> struct proclaim_is_ctf_isotropic : std::false_type {};
    template<typename T> using is_ctf_isotropic = std::bool_constant<proclaim_is_ctf_isotropic<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_isotropic_v = is_ctf_isotropic<T>::value;
    template<typename... Ts> constexpr bool are_ctf_isotropic_v = bool_and<is_ctf_isotropic_v<Ts>...>::value;

    template<typename T> struct proclaim_is_ctf_anisotropic : std::false_type {};
    template<typename T> using is_ctf_anisotropic = std::bool_constant<proclaim_is_ctf_anisotropic<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_anisotropic_v = is_ctf_anisotropic<T>::value;
    template<typename... Ts> constexpr bool are_ctf_anisotropic_v = bool_and<is_ctf_anisotropic_v<Ts>...>::value;

    template<typename T> constexpr bool is_ctf_f32_v = is_ctf_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_f64_v = is_ctf_v<T> && std::is_same_v<value_type_t<T>, double>;
    template<typename T> constexpr bool is_ctf_isotropic_f32_v = is_ctf_isotropic_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_isotropic_f64_v = is_ctf_isotropic_v<T> && std::is_same_v<value_type_t<T>, double>;
    template<typename T> constexpr bool is_ctf_anisotropic_f32_v = is_ctf_anisotropic_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_anisotropic_f64_v = is_ctf_anisotropic_v<T> && std::is_same_v<value_type_t<T>, double>;
}
