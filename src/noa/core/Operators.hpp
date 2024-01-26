#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/Types.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    #define CAST_(x) static_cast<decltype(dst)>(x)
    struct Copy {   constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(src); }};
    struct Negate { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(-src); }};
    struct Square { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(src * src); }};
    struct Round  { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(round(src)); }};
    struct Rint   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(rint(src)); }};
    struct Ceil   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(ceil(src)); }};
    struct Floor  { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(floor(src)); }};
    struct Trunc  { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(trunc(src)); }};
    struct Sqrt   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(sqrt(src)); }};
    struct Rsqrt  { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(rsqrt(src)); }};
    struct Exp    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(exp(src)); }};
    struct Log    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(log(src)); }};
    struct Abs    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(abs(src)); }};
    struct Cos    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(cos(src)); }};
    struct Sin    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(sin(src)); }};
    struct Real   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(real(src)); }};
    struct Imag   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(imag(src)); }};
    struct Conj   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(conj(src)); }};
    struct OneMinus   { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(decltype(src){1} - src); }};
    struct Inverse    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(decltype(src){1} / src); }};
    struct NonZero    { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(src != decltype(src){}); } };
    struct LogicalNot { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(!src); }};
    struct Normalize  { constexpr void operator()(const auto& src, auto& dst) const { dst = CAST_(normalize(src)); }};
    #undef CAST_

    struct AbsOneLog {
        constexpr void operator()(const auto& src, auto& dst) const {
            dst = static_cast<decltype(dst)>(log(nt::value_type_t<decltype(src)>{1} + abs(src)));
        }
    };

    struct OneLog {
        constexpr void operator()(const auto& src, auto& dst) const {
            dst = static_cast<decltype(dst)>(log(nt::value_type_t<decltype(src)>{1} + src));
        }
    };

    struct AbsSquared {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& src, auto& dst) const {
            if constexpr (nt::is_complex_v<T>) {
                dst = abs_squared(src);
            } else {
                auto tmp = abs(src);
                dst = tmp * tmp;
            }
        }
    };

    struct Plus {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs + rhs; } // ewise
        constexpr auto operator()(const auto& src, auto& dst) const { dst += src; } // reduce
    };

    struct Minus {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs - rhs; }
    };

    struct Multiply {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs * rhs; } // ewise
        constexpr auto operator()(const auto& src, auto& dst) const { dst *= src; } // reduce
    };

    struct Divide {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs / rhs; }
    };

    struct Modulo {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs % rhs; }
    };

    struct MultiplyConjugate {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs * conj(rhs); }
    };

    // If the divisor is too close to zero, do not divide and set the output to zero instead.
    struct DivideSafe {
        template<typename T, typename U>
        constexpr auto operator()(const T& lhs, const U& rhs, auto& dst) const {
            if constexpr (nt::are_real_or_complex_v<T, U>) {
                constexpr auto epsilon = std::numeric_limits<nt::value_type_t<U>>::epsilon();
                dst = abs(rhs) < epsilon ? T{} : lhs / rhs;
            } else if constexpr (nt::are_int_v<T, U>) {
                dst = rhs == 0 ? T{} : T(lhs / rhs); // short is implicitly promoted to int so cast it back
            } else {
                static_assert(nt::always_false_v<T>);
            }
        }
    };

    struct DistanceSquared {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const {
            auto tmp = lhs - rhs;
            dst = tmp * tmp;
        }
    };

    struct Pow {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x, const T& e, auto& dst) const {
            dst = pow(x, e);
        }
    };

    struct Equal {
        constexpr auto operator()(const auto& lhs, const auto& rhs, auto& dst) const { dst = lhs == rhs; } // ewise
        constexpr auto operator()(const auto& src, auto& dst) const { dst = src == dst; } // reduce
    };

    struct NotEqual {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs != rhs);
        }
    };

    struct Less {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs < rhs);
        }
    };

    struct LessEqual {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs <= rhs);
        }
    };

    struct Greater {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs > rhs);
        }
    };

    struct GreaterEqual {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs >= rhs);
        }
    };

    struct LogicalAnd {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs && rhs);
        }
    };

    struct LogicalOr {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs || rhs);
        }
    };

    struct Min {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(min(lhs, rhs));
        }
    };

    struct Max {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(max(lhs, rhs));
        }
    };

    struct PlusMinus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs + mhs - rhs);
        }
    };

    struct PlusMultiply {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>((lhs + mhs) * rhs);
        }
    };

    struct PlusDivide {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>((lhs + mhs) / rhs);
        }
    };

    struct MinusPlus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs - mhs + rhs);
        }

        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, auto& sum) const {
            sum += static_cast<decltype(sum)>(lhs - rhs);
        }

        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced = ireduced + reduced;
        }
    };

    struct MinusMultiply {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>((lhs - mhs) * rhs);
        }
    };

    struct MinusDivide {
        template<typename T>
        constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, T& out) const {
            out = static_cast<T>((lhs - mhs) / rhs);
        }
    };

    struct MultiplyPlus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs * mhs + rhs);
        }

        NOA_FHD constexpr void init(const auto& value, const auto& mask, auto& sum) const {
            sum += static_cast<decltype(sum)>(value * mask);
        }

        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced = ireduced + reduced;
        }
    };

    struct MultiplyMinus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs * mhs - rhs);
        }
    };

    struct MultiplyDivide {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>((lhs * mhs) / rhs);
        }
    };

    struct DividePlus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs / mhs + rhs);
        }
    };

    struct DivideMinus {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs / mhs - rhs);
        }
    };

    struct DivideMultiply {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs / mhs * rhs);
        }
    };

    struct DivideEpsilon {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, auto& out) const {
            out = static_cast<decltype(out)>(lhs / (mhs + rhs));
        }
    };

    struct Within {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& low, const auto& high, auto& out) const {
            out = static_cast<decltype(out)>(low < lhs && lhs < high);
        }
    };

    struct WithinEqual {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& low, const auto& high, auto& out) const {
            out = static_cast<decltype(out)>(low <= lhs && lhs <= high);
        }
    };

    struct Clamp {
        NOA_FHD constexpr auto operator()(const auto& lhs, const auto& low, const auto& high, auto& out) const {
            out = static_cast<decltype(out)>(clamp(lhs, low, high));
        }
    };
}
