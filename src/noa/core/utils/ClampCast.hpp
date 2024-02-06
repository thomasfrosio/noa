#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Half.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/math/Comparison.hpp"

namespace noa {
    /// Safely converts a value of a given data type to another data type. If out-of-range conditions happen,
    /// the value is clamped to the range of the destination type in a way that is always well defined.
    /// \see https://en.cppreference.com/w/cpp/language/implicit_conversion
    /// \note For floating-point to integral types, NaN returns 0, -/+Inf returns the min/max integral value.
    /// \note If the output type has a wider range than the input type, this function should have no runtime
    ///       overhead compared to static_cast.
    template<typename TTo, typename TFrom,
             nt::enable_if_bool_t<nt::is_numeric_v<TTo> && nt::is_numeric_v<TFrom>> = true>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const TFrom& src) noexcept {
        if constexpr (std::is_same_v<TTo, TFrom>) {
            return src;

        } else if constexpr(nt::is_complex_v<TFrom>) {
            static_assert(nt::is_complex_v<TTo>, "Cannot cast a complex value to a non-complex value");
            return {clamp_cast<typename TTo::value_type>(src.real),
                    clamp_cast<typename TTo::value_type>(src.imag)};

        } else if constexpr(nt::is_complex_v<TTo>) {
            return {clamp_cast<typename TTo::value_type>(src),
                    typename TTo::value_type{0}};

        } else if constexpr(nt::is_real_v<TTo>) {
            // Floating-point conversions:
            //      - If smaller -> larger type, it is a promotion and the value does not change.
            //      - The larger type may be represented exactly by the smaller type. In this case, the value doesn't
            //        change.
            //      - If the larger type is between two representable values of the smaller type, the result is one of
            //        those two values (it is implementation-defined which one, although if IEEE arithmetic is supported,
            //        rounding defaults to nearest).
            //      - If the larger type is out of the smaller type range, the behavior is undefined and clamping is
            //        required.
            //
            // Integral to floating-point conversions:
            //      - Integer types can be converted to any floating-point type. If the value cannot be represented
            //        correctly, it is implementation defined whether the closest higher or the closest lower
            //        representable value will be selected, although if IEEE arithmetic is supported, rounding defaults
            //        to nearest. If the source type is bool, the value false is converted to zero, and the value true
            //        is converted to one.
            //      - Half can only represent the range of (u)int8_t and int16_t. Note that the rounding may cause
            //        the casting to be non-reversible (e.g. int(32767) -> Half -> int(32768)). For larger integral
            //        or floating-point types, full clamping is required.
            if constexpr (std::is_same_v<TTo, Half>) {
                if constexpr (sizeof(TFrom) == 1 || (sizeof(TFrom) == 2 && std::is_signed_v<TFrom>)) {
                    return TTo(src); // (u)int8_t/int16_t -> Half
                } else if constexpr (std::is_unsigned_v<TFrom>) {
                    return TTo(min(src, TFrom(std::numeric_limits<TTo>::max()))); // uint(16|32|64)_t -> Half
                } else { // int(32|64)_t -> Half, float/double -> Half
                    return TTo(clamp(src, TFrom(std::numeric_limits<TTo>::lowest()), TFrom(std::numeric_limits<TTo>::max())));
                }
            } else if constexpr (std::is_integral_v<TFrom> || (sizeof(TFrom) < sizeof(TTo))) {
                return TTo(src); // implicit integral/Half->float/double conversion or float->double
            } else { // double->float
                return TTo(clamp(src, TFrom(std::numeric_limits<TTo>::lowest()), TFrom(std::numeric_limits<TTo>::max())));
            }

        } else if constexpr (std::is_integral_v<TTo> && nt::is_real_v<TFrom>) {
            // Floating-point to integral conversions:
            //      - Floating-point type can be converted to any integer type. The fractional part is truncated.
            //      - If bool, this is a bool conversion, i.e. a value of zero gives false, anything else gives true.
            //      - If the floating-point value cannot fit into the integral type, the behavior is undefined.
            //        See https://stackoverflow.com/a/26097083 for some exceptional cases.
            //        See https://stackoverflow.com/a/3793950 for largest int value that can be accurately represented
            //        by IEEE-754 floats.
            //      - Half is an exception since some integral types have a wider range. In these cases, no need to
            //        clamp, but still check for NaN and +/-Inf.
            using int_limits = std::numeric_limits<TTo>;
            constexpr bool IS_WIDER_THAN_HALF = sizeof(TTo) > 2 || (sizeof(TTo) == 2 && std::is_unsigned_v<TTo>);
            if constexpr (std::is_same_v<TFrom, Half> && IS_WIDER_THAN_HALF) {
                if (is_nan(src)) {
                    return 0;
                } else if (src == Half(Half::Mode::BINARY, 0x7C00)) { // +inf
                    return int_limits::max();
                } else if (src == Half(Half::Mode::BINARY, 0xFC00)) { // -inf
                    return int_limits::min();
                } else {
                    if constexpr (std::is_unsigned_v<TTo>)
                        return src < TFrom(0) ? 0 : TTo(src);
                    else
                        return TTo(src);
                }
            } else {
                if (is_nan(src))
                    return 0;
                else if (src <= static_cast<TFrom>(int_limits::min()))
                    return int_limits::min();
                else if (src >= static_cast<TFrom>(int_limits::max()))
                    // static_cast is integral to floating-point conversion, so it is defined and can either,
                    // 1) round up and in this case if src is int_max it evaluates to false and go to the else
                    // branch, which is fine since it is in range, or 2) round down and return int_max.
                    return int_limits::max();
                else
                    return TTo(src);
            }

        } else {
            // Integral conversions:
            //      - If the conversion is listed under integral promotions, it is a promotion and not a conversion,
            //        and the value will always be preserved.
            //      - If the destination is unsigned, whether the destination type is wider or narrower, signed integers
            //        are sign-extended or truncated and unsigned integers are zero-extended or truncated, respectively.
            //        As such, if the destination is wider and the source is positive, the value is preserved.
            //      - If the destination is signed, the value does not change if the source integer can be represented
            //        in the destination type. Otherwise, the result is implementation-defined (until C++20).
            //      - If bool is involved, everything is well-defined and clamping is never required.
            using to_limits = std::numeric_limits<TTo>;

            if constexpr (std::is_unsigned_v<TFrom>) {
                // Source is unsigned, we only need to check the upper bound.
                // If destination is signed and wider, this is optimized away.
                // If destination is unsigned and wider, this is optimized away (zero-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) < sizeof(TTo)), TTo, TFrom>;
                return TTo(min(wider_type(src), wider_type(to_limits::max())));

            } else if constexpr (std::is_unsigned_v<TTo>) {
                // Source is signed, we need to check the lower and upper bound.
                // If destination is wider or same size, upper bound check is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) > sizeof(TTo)), TFrom, TTo>;
                if (src < TFrom(0))
                    return TTo(0);
                else
                    return TTo(min(wider_type(src), wider_type(to_limits::max())));
            } else {
                // Both are signed, we need to check the lower and lower bound.
                // If destination is wider, this is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) < sizeof(TTo)), TTo, TFrom>;
                return TTo(clamp(wider_type(src), wider_type(to_limits::min()), wider_type(to_limits::max())));
            }
        }
        return TTo();
    }
}
