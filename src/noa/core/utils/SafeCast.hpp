#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Half.hpp"

namespace noa {
    /// Whether it is numerically safe (no loss of range) to cast \p src to type \p TTo.
    /// \details This function detects loss of range when a numeric type is converted and returns false if
    ///          the range cannot be preserved. This is related to clamp_cast: if this function returns false,
    ///          clamp_cast needs to clamp the value, otherwise, it calls static_cast.
    /// \note For integral to float/double conversion, this function always returns true. Indeed, it is not
    ///       accounting for the fact that some integral values cannot be represented in the IEEE 754 format
    ///       and will be rounded, usually to the nearest integral value, because there is no a loss of range.
    template<typename TTo, typename TFrom,
             nt::enable_if_bool_t<nt::is_numeric_v<TTo> && nt::is_numeric_v<TFrom>> = true>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const TFrom& src) noexcept {
        // See clamp_cast for more details.
        if constexpr (std::is_same_v<TTo, TFrom>) {
            return true;

        } else if constexpr(nt::is_complex_v<TFrom>) {
            static_assert(nt::is_complex_v<TTo>, "Cannot cast a complex value to a non-complex value");
            return is_safe_cast<typename TTo::value_type>(src.real) &&
                   is_safe_cast<typename TTo::value_type>(src.imag);

        } else if constexpr(nt::is_complex_v<TTo>) {
            return is_safe_cast<typename TTo::value_type>(src);

        } else if constexpr(nt::is_real_v<TTo>) {
            if constexpr (std::is_same_v<TTo, Half>) {
                if constexpr (sizeof(TFrom) == 1 || (sizeof(TFrom) == 2 && std::is_signed_v<TFrom>)) {
                    return true; // (u)int8_t/int16_t -> Half
                } else if constexpr (std::is_unsigned_v<TFrom>) {
                    return src <= TFrom(std::numeric_limits<TTo>::max()); // uint(16|32|64)_t -> Half
                } else { // int(32|64)_t -> Half, float/double -> Half
                    return TFrom(std::numeric_limits<TTo>::lowest()) <= src && src <= TFrom(std::numeric_limits<TTo>::max());
                }
            } else if constexpr (std::is_integral_v<TFrom> || (sizeof(TFrom) < sizeof(TTo))) {
                return true; // implicit integral/Half->float/double conversion or float->double
            } else { // double->float
                return TFrom(std::numeric_limits<TTo>::lowest()) <= src && src <= TFrom(std::numeric_limits<TTo>::max());
            }

        } else if constexpr (std::is_integral_v<TTo> && nt::is_real_v<TFrom>) {
            using int_limits = std::numeric_limits<TTo>;
            constexpr bool IS_WIDER_THAN_HALF = sizeof(TTo) > 2 || (sizeof(TTo) == 2 && std::is_unsigned_v<TTo>);
            if constexpr (std::is_same_v<TFrom, Half> && IS_WIDER_THAN_HALF) {
                if (is_nan(src) ||
                    src == Half::from_bits(0x7C00) ||
                    src == Half::from_bits(0xFC00)) {
                    return false;
                } else {
                    if constexpr (std::is_unsigned_v<TTo>)
                        return TFrom(0) <= src;
                    else
                        return true;
                }
            } else {
                return !is_nan(src) &&
                       static_cast<TFrom>(int_limits::min()) <= src &&
                       src <= (static_cast<TFrom>(int_limits::max()) + static_cast<TFrom>(1));
            }

        } else {
            using to_limits = std::numeric_limits<TTo>;

            if constexpr (std::is_unsigned_v<TFrom>) {
                // Source is unsigned, we only need to check the upper bound.
                // If destination is wider, this is optimized away and returns true.
                using wider_type = std::conditional_t<(sizeof(TFrom) < sizeof(TTo)), TTo, TFrom>;
                return wider_type(src) <= wider_type(to_limits::max());

            } else if constexpr (std::is_unsigned_v<TTo>) {
                // Source is signed, we need to check the lower and upper bound.
                // If destination is wider or same size, upper bound check is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) > sizeof(TTo)), TFrom, TTo>;
                return TFrom(0) <= src && wider_type(src) <= wider_type(to_limits::max());
            } else {
                // Both are signed, we need to check the lower and lower bound.
                using wider_type = std::conditional_t<(sizeof(TFrom) < sizeof(TTo)), TTo, TFrom>;
                return wider_type(to_limits::min()) <= src && wider_type(src) <= wider_type(to_limits::max());
            }
        }
        return false;
    }

#ifdef NOA_IS_OFFLINE
    /// Casts src to type TTo, with bound-checks. Throws if there is a loss of range.
    /// This should be very similar to boost::numeric_cast.
    /// If the output type has a wider range than the input type, this function should have no runtime
    /// overhead compared to static_cast.
    template<typename TTo, typename TFrom>
    [[nodiscard]] constexpr TTo safe_cast(const TFrom& src) {
        if (is_safe_cast<TTo>(src))
            return static_cast<TTo>(src);
        panic("Cannot safely cast {} to {} type", src, ns::to_human_readable<TTo>());
    }
#endif
}
