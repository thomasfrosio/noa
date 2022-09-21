#pragma once

#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Complex.h"
#include "noa/common/types/Half.h"

namespace noa {
    /// Whether it is numerically safe (no loss of range) to cast \p src to type \p TTo.
    /// \details This function detects loss of range when a numeric type is converted and returns false if
    ///          the range cannot be preserved. This is related to clamp_cast: if this function returns false,
    ///          clamp_cast needs to clamp the value, otherwise, it calls static_cast.
    /// \note For integral to float/double conversion, this function always returns true. Indeed, it is not
    ///       accounting for the fact that some integral values cannot be represented in the IEEE 754 format
    ///       and will be rounded, usually to the nearest integral value, because there is no a loss of range.
    template<typename TTo, typename TFrom,
             typename std::enable_if_t<traits::is_data_v<TTo> && traits::is_data_v<TFrom>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr bool isSafeCast(const TFrom& src) noexcept {
        // See clamp_cast for more details.
        if constexpr (std::is_same_v<TTo, TFrom>) {
            return true;

        } else if constexpr(traits::is_complex_v<TFrom>) {
            static_assert(traits::is_complex_v<TTo>);
            return isSafeCast<typename TTo::value_type>(src.real) &&
                   isSafeCast<typename TTo::value_type>(src.imag);

        } else if constexpr(traits::is_complex_v<TTo>) {
            return isSafeCast<typename TTo::value_type>(src);

        } else if constexpr(traits::is_float_v<TTo>) {
            if constexpr (std::is_same_v<TTo, half_t>) {
                if constexpr (sizeof(TFrom) == 1 || (sizeof(TFrom) == 2 && std::is_signed_v<TFrom>)) {
                    return true; // (u)int8_t/int16_t -> half_t
                } else if constexpr (std::is_unsigned_v<TFrom>) {
                    return src <= TFrom(math::Limits<TTo>::max()); // uint(16|32|64)_t -> half_t
                } else { // int(32|64)_t -> half_t, float/double -> half_t
                    return TFrom(math::Limits<TTo>::lowest()) <= src &&
                           src <= TFrom(math::Limits<TTo>::max());
                }
            } else if constexpr (std::is_integral_v<TFrom> || (sizeof(TFrom) < sizeof(TTo))) {
                return true; // implicit integral/half_t->float/double conversion or float->double
            } else { // double->float
                return TFrom(math::Limits<TTo>::lowest()) <= src &&
                       src <= TFrom(math::Limits<TTo>::max());
            }

        } else if constexpr (std::is_integral_v<TTo> && traits::is_float_v<TFrom>) {
            using int_limits = math::Limits<TTo>;
            constexpr bool IS_WIDER_THAN_HALF = sizeof(TTo) > 2 || (sizeof(TTo) == 2 && std::is_unsigned_v<TTo>);
            if constexpr (std::is_same_v<TFrom, half_t> && IS_WIDER_THAN_HALF) {
                if (math::isNaN(src) ||
                    src == half_t(half_t::Mode::BINARY, 0x7C00) ||
                    src == half_t(half_t::Mode::BINARY, 0xFC00)) {
                    return false;
                } else {
                    if constexpr (std::is_unsigned_v<TTo>)
                        return TFrom(0) <= src;
                    else
                        return true;
                }
            } else {
                return !math::isNaN(src) &&
                       static_cast<TFrom>(int_limits::min()) <= src &&
                       src <= (static_cast<TFrom>(int_limits::max()) + static_cast<TFrom>(1));
            }

        } else {
            using to_limits = math::Limits<TTo>;

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

    /// Casts \p src to type \p TTo, with bound-checks. Throws if there is a loss of range.
    /// \note This should be very similar to boost::numeric_cast.
    /// \note If the output type has a wider range than the input type, this function should have no runtime
    ///       overhead compared to static_cast.
    template<typename TTo, typename TFrom>
    [[nodiscard]] constexpr TTo safe_cast(const TFrom& src) {
        if (isSafeCast<TTo>(src))
            return static_cast<TTo>(src);
        NOA_THROW("Cannot safely cast {} to {} type", src, string::human<TTo>());
    }
}
