#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/types/Half.hpp"

namespace noa {
    /// Whether it is numerically safe (no loss of range) to cast \p src to type \p To.
    /// \details This function detects loss of range when a numeric type is converted and returns false if
    ///          the range cannot be preserved. This is related to clamp_cast: if this function returns false,
    ///          clamp_cast needs to clamp the value, otherwise, it calls static_cast.
    /// \note For integral to float/double conversion, this function always returns true. Indeed, it is not
    ///       accounting for the fact that some integral values cannot be represented in the IEEE 754 format
    ///       and will be rounded, usually to the nearest integral value, because there is no a loss of range.
    template<nt::numeric To, nt::numeric From>
    [[nodiscard]] NOA_FHD constexpr bool is_safe_cast(const From& src) noexcept {
        // See clamp_cast for more details.
        if constexpr (std::is_same_v<To, From>) {
            return true;

        } else if constexpr(nt::is_complex_v<From>) {
            static_assert(nt::is_complex_v<To>, "Cannot cast a complex value to a non-complex value");
            return is_safe_cast<typename To::value_type>(src.real) and
                   is_safe_cast<typename To::value_type>(src.imag);

        } else if constexpr(nt::is_complex_v<To>) {
            return is_safe_cast<typename To::value_type>(src);

        } else if constexpr(nt::is_real_v<To>) {
            if constexpr (std::is_same_v<To, Half>) {
                if constexpr (sizeof(From) == 1 or (sizeof(From) == 2 and std::is_signed_v<From>)) {
                    return true; // (u)int8_t/int16_t -> Half
                } else if constexpr (std::is_unsigned_v<From>) {
                    return src <= From(std::numeric_limits<To>::max()); // uint(16|32|64)_t -> Half
                } else { // int(32|64)_t -> Half, float/double -> Half
                    return From(std::numeric_limits<To>::lowest()) <= src and src <= From(std::numeric_limits<To>::max());
                }
            } else if constexpr (std::is_integral_v<From> or (sizeof(From) < sizeof(To))) {
                return true; // implicit integral/Half->float/double conversion or float->double
            } else { // double->float
                return From(std::numeric_limits<To>::lowest()) <= src and src <= From(std::numeric_limits<To>::max());
            }

        } else if constexpr (std::is_integral_v<To> and nt::is_real_v<From>) {
            using int_limits = std::numeric_limits<To>;
            constexpr bool IS_WIDER_THAN_HALF = sizeof(To) > 2 or (sizeof(To) == 2 and std::is_unsigned_v<To>);
            if constexpr (std::is_same_v<From, Half> and IS_WIDER_THAN_HALF) {
                if (is_nan(src) or
                    src == Half::from_bits(0x7C00) or
                    src == Half::from_bits(0xFC00)) {
                    return false;
                } else {
                    if constexpr (std::is_unsigned_v<To>)
                        return From(0) <= src;
                    else
                        return true;
                }
            } else {
                return not is_nan(src) and
                       static_cast<From>(int_limits::min()) <= src and
                       src <= (static_cast<From>(int_limits::max()) + static_cast<From>(1));
            }

        } else {
            using to_limits = std::numeric_limits<To>;

            if constexpr (std::is_unsigned_v<From>) {
                // Source is unsigned, we only need to check the upper bound.
                // If the destination is wider, this is optimized away and returns true.
                using wider_type = std::conditional_t<(sizeof(From) < sizeof(To)), To, From>;
                return wider_type(src) <= wider_type(to_limits::max());

            } else if constexpr (std::is_unsigned_v<To>) {
                // Source is signed, we need to check the lower and upper bound.
                // If the destination is wider or same size, upper bound check is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(From) > sizeof(To)), From, To>;
                return From(0) <= src and wider_type(src) <= wider_type(to_limits::max());
            } else {
                // Both are signed, we need to check the lower and lower bound.
                using wider_type = std::conditional_t<(sizeof(From) < sizeof(To)), To, From>;
                return wider_type(to_limits::min()) <= src and wider_type(src) <= wider_type(to_limits::max());
            }
        }
        return false;
    }
}

namespace noa::traits {
    template<typename From, typename To>
    concept safe_castable_to = requires (const From& v) {{ is_safe_cast<To>(v) } -> std::convertible_to<bool>; };
}

namespace noa {
    /// Casts src to type To, with bound-checks.
    /// Panics if there is a loss of range.
    /// This should be very similar to boost::numeric_cast.
    /// If the output type has a wider range than the input type,
    /// this function should have no runtime overhead compared to static_cast.
    template<typename To, typename From> requires nt::safe_castable_to<From, To>
    [[nodiscard]] constexpr To safe_cast(const From& src) {
        if (is_safe_cast<To>(src))
            return static_cast<To>(src);
        panic("Cannot safely cast {} to {} type", src, ns::stringify<To>());
    }

    template<typename To, typename From, typename... Ts> requires nt::safe_castable_to<From, To>
    [[nodiscard]] constexpr To safe_cast(const From& src, guts::FormatWithLocation<std::type_identity_t<Ts>...> fmt, Ts&&... fmt_args) {
        if (is_safe_cast<To>(src))
            return static_cast<To>(src);
        panic_at_location(fmt.location, fmt.fmt, std::forward<Ts>(fmt_args)...);
    }
}
