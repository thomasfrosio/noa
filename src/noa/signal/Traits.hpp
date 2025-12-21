#pragma once

#include "noa/runtime/core/Traits.hpp"

namespace noa::traits {
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
}
