/**
 * @file Math.h
 * @brief Various "math related functions".
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/traits/Base.h"

#define ULP 2
#define EPSILON 1e-6f


namespace Noa::Math {
    /**
     * Whether or not two floating-points are "significantly" equal.
     * @details For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
     *          the values used and multiplied by the desired precision in ULPs. The magnitude is
     *          often set as max(abs(x), abs(y)), but this function is setting the magnitude as
     *          abs(x+y), which is basically equivalent and is should be more efficient. Relative
     *          epsilons and Unit in the Last Place (ULPs) comparisons are usually meaningless for
     *          close-to-zero numbers, hence the absolute comparison with @a epsilon, acting as
     *          a safety net.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    constexpr bool isEqual(T x, T y, T epsilon = EPSILON) {
        const auto diff = std::abs(x - y);
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * Whether or not @a x is less or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    constexpr bool isLessOrEqual(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = x - y;
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * Whether or not @a x is greater or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    constexpr bool isGreaterOrEqual(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = y - x;
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * Whether or not @a x is "significantly" withing @a min and @a max.
     * @note    If one or all values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    inline bool isWithin(T x, T min, T max, T epsilon = EPSILON) noexcept {
        return isGreaterOrEqual<ulp>(x, min, epsilon) && isLessOrEqual<ulp>(x, max, epsilon);
    }


    /**
     * Whether or not @a x is "significantly" less than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    constexpr bool isLess(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = y - x;
        if (!std::isfinite(diff))
            return false;

        return diff > epsilon ||
               diff > (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * Whether or not @a x is "significantly" greater than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    constexpr bool isGreater(T x, T y, T epsilon = EPSILON) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = x - y;
        if (!std::isfinite(diff))
            return false;

        return diff > epsilon ||
               diff > (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }
}

#undef ULP
#undef EPSILON
