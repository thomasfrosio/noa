#pragma once

#include "noa/core/traits/Utilities.hpp"

namespace noa::traits {
    template<typename T> struct proclaim_is_interpolator_2d : std::false_type {};
    template<typename T> struct proclaim_is_interpolator_3d : std::false_type {};

    template<typename T> using is_interpolator_2d = std::bool_constant<proclaim_is_interpolator_2d<remove_ref_cv_t<T>>::value>;
    template<typename T> using is_interpolator_3d = std::bool_constant<proclaim_is_interpolator_3d<remove_ref_cv_t<T>>::value>;

    template<typename T> constexpr bool is_interpolator_2d_v = is_interpolator_2d<T>::value;
    template<typename T> constexpr bool is_interpolator_3d_v = is_interpolator_3d<T>::value;
    template<typename T> constexpr bool is_interpolator_v = is_interpolator_2d_v<T> || is_interpolator_3d_v<T>;
    template<typename... Ts> constexpr bool are_interpolator_v = bool_and<is_interpolator_v<Ts>...>::value;
}
