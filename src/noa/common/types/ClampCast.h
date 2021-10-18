#pragma once

#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>

#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Complex.h"

namespace noa {
    /// Safely converts a value of a given data type to another data type. If out-of-range conditions happen,
    /// the value is clamped to the range of the destination type in a way that is always well defined.
    /// \see https://en.cppreference.com/w/cpp/language/implicit_conversion
    template<typename TTo, typename TFrom>
    constexpr TTo clamp_cast(const TFrom& src) noexcept {
        static_assert(noa::traits::is_data_v<TTo> && noa::traits::is_data_v<TFrom>);
        if constexpr (std::is_same_v<TTo, TFrom>) {
            return src;

        } else if constexpr(noa::traits::is_complex_v<TFrom>) {
            static_assert(noa::traits::is_complex_v<TTo>); // only Complex<T>->Complex<U>
            return {clamp_cast<typename TTo::value_type>(src.real),
                    clamp_cast<typename TTo::value_type>(src.imag)};

        } else if constexpr(noa::traits::is_complex_v<TTo>) {
            return clamp_cast<typename TTo::value_type>(src); // calls implicit constructor Complex(U); imaginary is 0.

        } else if constexpr(std::is_floating_point_v<TTo>) {
            // Floating-point conversions:
            //      - float to double is a promotion and the value does not change.
            //      - If double can be represented exactly as float, it does not change.
            //      - If double is between two representable float values, the result is one of those two values (it is
            //        implementation-defined which one, although if IEEE arithmetic is supported, rounding defaults to
            //        nearest).
            //      - If double is out of the float range, the behavior is undefined.
            //
            // Integral to floating-point conversions:
            //      - Integer types can be converted to any floating-point type. If the value cannot be represented
            //        correctly, it is implementation defined whether the closest higher or the closest lower
            //        representable value will be selected, although if IEEE arithmetic is supported, rounding defaults
            //        to nearest. If the source type is bool, the value false is converted to zero, and the value true
            //        is converted to one.
            if constexpr (std::is_integral_v<TFrom> || std::is_same_v<TTo, double>) {
                return TTo(src); // implicit integral->float conversion or float->double
            } else { // double->float
                return TTo(std::clamp(src,
                                      TFrom(std::numeric_limits<TTo>::lowest()),
                                      TFrom(std::numeric_limits<TTo>::max())));
            }

        } else if constexpr (std::is_integral_v<TTo> && std::is_floating_point_v<TFrom>) {
            // Floating-point to integral conversions:
            //      - Floating-point type can be converted to any integer type. The fractional part is truncated.
            //      - If bool, this is a bool conversion, i.e. a value of zero gives false, anything else gives true.
            //      - If the floating-point value cannot fit into the integral type, the behavior is undefined.
            //        See https://stackoverflow.com/a/26097083) for some exceptional cases.
            //        See https://stackoverflow.com/a/3793950) for largest int value that can be accurately represented
            //        by IEEE-754 floats.
            using int_limits = std::numeric_limits<TTo>;
            if (std::isnan(src))
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
                return TTo(std::min(wider_type(src), wider_type(to_limits::max())));

            } else if constexpr (std::is_unsigned_v<TTo>) {
                // Source is signed, we need to check the lower and upper bound.
                // If destination is wider or same size, upper bound check is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) > sizeof(TTo)), TFrom, TTo>;
                if (src < TFrom(0))
                    return TTo(0);
                else
                    return TTo(std::min(wider_type(src), wider_type(to_limits::max())));
            } else {
                // Both are signed, we need to check the lower and lower bound.
                // If destination is wider, this is optimized away (sign-extension).
                using wider_type = std::conditional_t<(sizeof(TFrom) < sizeof(TTo)), TTo, TFrom>;
                return TTo(std::clamp(wider_type(src),
                                      wider_type(to_limits::min()),
                                      wider_type(to_limits::max())));
            }
        }
    }
}
