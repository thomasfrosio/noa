#pragma once

#include "noa/runtime/core/Traits.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(quaternion);
    template<typename... T> concept quaternion_f32 = quaternion<T...> and same_as<f32, value_type_t<T>...>;
    template<typename... T> concept quaternion_f64 = quaternion<T...> and same_as<f64, value_type_t<T>...>;

    NOA_GENERATE_PROCLAIM_FULL(interpolator);
    template<typename T, size_t N> struct proclaim_is_interpolator_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_nd = std::disjunction<proclaim_is_interpolator_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_nd_v = is_interpolator_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_nd = interpolator<T> and is_interpolator_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_nd = conjunction_or_false<is_interpolator_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_nd_v = are_interpolator_nd<N, T...>::value;

    NOA_GENERATE_PROCLAIM_FULL(interpolator_spectrum);
    template<typename T, size_t N> struct proclaim_is_interpolator_spectrum_nd : std::false_type {};
    template<typename T, size_t... N> using is_interpolator_spectrum_nd = std::disjunction<proclaim_is_interpolator_spectrum_nd<std::remove_cv_t<T>, N>...>;
    template<typename T, size_t... N> constexpr bool is_interpolator_spectrum_nd_v = is_interpolator_spectrum_nd<T, N...>::value;
    template<typename T, size_t... N> concept interpolator_spectrum_nd = interpolator_spectrum<T> and is_interpolator_spectrum_nd<T, N...>::value;
    template<size_t N, typename... T> using are_interpolator_spectrum_nd = conjunction_or_false<is_interpolator_spectrum_nd<T, N>...>;
    template<size_t N, typename... T> constexpr bool are_interpolator_spectrum_nd_v = are_interpolator_spectrum_nd<N, T...>::value;

    template<typename... T> concept interpolator_or_empty = conjunction_or_false<std::disjunction<is_interpolator<T>, std::is_empty<T>>...>::value;
    template<typename... T> concept interpolator_spectrum_or_empty = conjunction_or_false<std::disjunction<is_interpolator_spectrum<T>, std::is_empty<T>>...>::value;
    template<typename T, size_t... N> concept interpolator_nd_or_empty = interpolator_nd<T, N...> or empty<T>;
    template<typename T, size_t... N> concept interpolator_spectrum_nd_or_empty = interpolator_spectrum_nd<T, N...> or empty<T>;
}
