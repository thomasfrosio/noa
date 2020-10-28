/**
 * @file Assert.h
 * @brief Various assertions, mostly about floating point comparisons.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"


namespace Noa::Assert {

    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         First value.
     * @param[in] y         Second value.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x and y are (almost) equal.
     *
     * @note                For the relative epsilon, the machine epsilon has to be scaled to the
     *                      magnitude of the values used and multiplied by the desired precision
     *                      in ULPs. The magnitude is often set as max(abs(x), abs(y)), but this
     *                      function is setting the magnitude as abs(x+y), which is basically
     *                      equivalent and is more efficient.
     *                      Relative epsilons and ULPs comparison are usually meaningless for
     *                      close-to-zero numbers, hence the absolute comparison, acting as a
     *                      safety net.
     *
     * @note                If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<typename T, int ulp = 4>
    constexpr bool isEqual(T x, T y, T epsilon = 1e-6) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = std::abs(x - y);
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         First value.
     * @param[in] y         Second value.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x is less or (almost) equal than y.
     *
     * @note                If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<typename T, int ulp = 4>
    constexpr bool isLessOrEqual(T x, T y, T epsilon = 1e-6) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = x - y;
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         First value.
     * @param[in] y         Second value.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x is greater or (almost) equal than y.
     *
     * @note                If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<typename T, int ulp = 4>
    constexpr bool isGreaterOrEqual(T x, T y, T epsilon = 1e-6) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = y - x;
        if (!std::isfinite(diff))
            return false;

        return diff <= epsilon ||
               diff <= (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         Value to assert.
     * @param[in] min       Minimum allowed.
     * @param[in] min       Maximum allowed.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x is (almost) withing min and max.
     *
     * @note                If one or all values are NaN and|or +/-Inf, returns false.
     */
    template<typename T>
    inline bool isWithin(T x, T min, T max, T epsilon = 1e-6) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        return isGreaterOrEqual(x, min, epsilon) && isLessOrEqual(x, max, epsilon);
    }


    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         First value.
     * @param[in] y         Second value.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x is (definitely) less than y.
     *
     * @note                If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<typename T, int ulp = 4>
    constexpr bool isLess(T x, T y, T epsilon = 1e-6) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = y - x;
        if (!std::isfinite(diff))
            return false;

        return diff > epsilon ||
               diff > (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }


    /**
     * @tparam T            Floating-point.
     * @tparam ulp          Unit in the Last Place (ULP), used to compute the relative epsilon.
     * @param[in] x         First value.
     * @param[in] y         Second value.
     * @param[in] epsilon   Epsilon used for the absolute difference comparison.
     * @return              Whether or not x is (definitely) greater than y.
     *
     * @note                If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<typename T, int ulp = 4>
    constexpr bool isGreater(T x, T y, T epsilon = 1e-6) noexcept {
        static_assert(::Noa::Traits::is_float_v<T>);

        const auto diff = x - y;
        if (!std::isfinite(diff))
            return false;

        return diff > epsilon ||
               diff > (std::abs(x + y) * std::numeric_limits<T>::epsilon() * ulp);
    }
}
