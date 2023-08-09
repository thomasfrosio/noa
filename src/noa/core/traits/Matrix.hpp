#pragma once

#include <type_traits>
#include "noa/core/traits/Utilities.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::traits {
    template<typename> struct proclaim_is_mat22 : std::false_type {};
    template<typename T> using is_mat22 = std::bool_constant<proclaim_is_mat22<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat22_v = is_mat22<T>::value;

    template<typename> struct proclaim_is_mat23 : std::false_type {};
    template<typename T> using is_mat23 = std::bool_constant<proclaim_is_mat23<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat23_v = is_mat23<T>::value;

    template<typename> struct proclaim_is_mat33 : std::false_type {};
    template<typename T> using is_mat33 = std::bool_constant<proclaim_is_mat33<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat33_v = is_mat33<T>::value;

    template<typename> struct proclaim_is_mat34 : std::false_type {};
    template<typename T> using is_mat34 = std::bool_constant<proclaim_is_mat34<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat34_v = is_mat34<T>::value;

    template<typename> struct proclaim_is_mat44 : std::false_type {};
    template<typename T> using is_mat44 = std::bool_constant<proclaim_is_mat44<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat44_v = is_mat44<T>::value;

    template<typename T> using is_matXX = std::bool_constant<is_mat22_v<T> || is_mat23_v<T> || is_mat33_v<T> || is_mat34_v<T> || is_mat44_v<T>>;
    template<typename T> constexpr bool is_matXX_v = is_matXX<T>::value;
}
