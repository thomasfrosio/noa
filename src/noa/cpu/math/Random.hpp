#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_random_v =
            noa::traits::is_any_v<T, i16, i32, i64, u16, u32, u64, f16, f32, f64, c16, c32, c64> &&
            (std::is_same_v<noa::traits::value_type_t<T>, U> || std::is_same_v<T, U> ||
             (std::is_same_v<T, f16> && noa::traits::is_real_v<U>) ||
             (std::is_same_v<T, c16> && noa::traits::is_real_v<noa::traits::value_type_t<U>>));
}

namespace noa::cpu::math {
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::uniform_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U min, U max, i64 threads);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::normal_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U mean, U stddev, i64 threads);

    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_random_v<T, U>>>
    void randomize(noa::math::log_normal_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U mean, U stddev, i64 threads);

    template<typename T, typename = std::enable_if_t<details::is_valid_random_v<T, traits::value_type_t<T>>>>
    void randomize(noa::math::poisson_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   f32 lambda, i64 threads);
}