#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    #define NOA_UNARY_OP_(name, src_op)                                     \
    struct name {                                                           \
        using allow_vectorization = bool;                                   \
        template<typename T>                                                \
        NOA_HD constexpr auto operator()(const T& src) const {              \
            src_op;                                                         \
        }                                                                   \
        template<typename T, typename U>                                    \
        NOA_HD constexpr void operator()(const T& src, U& dst) const {      \
            dst = static_cast<U>((*this)(src));                             \
        }                                                                   \
    }
    NOA_UNARY_OP_(Copy, return src);
    NOA_UNARY_OP_(Negate, return -src);
    NOA_UNARY_OP_(Square, return src * src);
    NOA_UNARY_OP_(Round, return round(src));
    NOA_UNARY_OP_(Rint, return rint(src));
    NOA_UNARY_OP_(Ceil, return ceil(src));
    NOA_UNARY_OP_(Floor, return floor(src));
    NOA_UNARY_OP_(Trunc, return trunc(src));
    NOA_UNARY_OP_(Sqrt, return sqrt(src));
    NOA_UNARY_OP_(Rsqrt, return rsqrt(src));
    NOA_UNARY_OP_(Exp, return exp(src));
    NOA_UNARY_OP_(Log, return log(src));
    NOA_UNARY_OP_(Abs, return abs(src));
    NOA_UNARY_OP_(Cos, return cos(src));
    NOA_UNARY_OP_(Sin, return sin(src));
    NOA_UNARY_OP_(Real, return real(src));
    NOA_UNARY_OP_(Imag, return imag(src));
    NOA_UNARY_OP_(Conj, return conj(src));
    NOA_UNARY_OP_(OneMinus, return T{1} - src);
    NOA_UNARY_OP_(Inverse, return T{1} / src);
    NOA_UNARY_OP_(Zero, return src == T{});
    NOA_UNARY_OP_(NonZero, return src != T{});
    NOA_UNARY_OP_(LogicalNot, return !src);
    NOA_UNARY_OP_(Normalize, return normalize(src));
    NOA_UNARY_OP_(AbsOneLog, return log(nt::value_type_t<T>{1} + abs(src)));
    NOA_UNARY_OP_(OneLog, return log(nt::value_type_t<T>{1} + src));
    NOA_UNARY_OP_(AbsSquared,
                  if constexpr (nt::complex<T>) {
                      return abs_squared(src);
                  } else {
                      auto tmp = abs(src);
                      return tmp * tmp;
                  });
    #undef NOA_UNARY_OP_

    #define NOA_BINARY_OP_(name, src_op)                                                \
    struct name {                                                                       \
        using allow_vectorization = bool;                                               \
        template<typename T, typename U>                                                \
        NOA_HD constexpr auto operator()(const T& lhs, const U& rhs) const {            \
            src_op;                                                                     \
        }                                                                               \
        template<typename T, typename U, typename V>                                    \
        NOA_HD constexpr void operator()(const T& lhs, const U& rhs, V& dst) const {    \
            dst = static_cast<V>((*this)(lhs, rhs));                                    \
        }                                                                               \
    }
    NOA_BINARY_OP_(Equal, return lhs == rhs);
    NOA_BINARY_OP_(NotEqual, return lhs != rhs);
    NOA_BINARY_OP_(Less, return lhs < rhs);
    NOA_BINARY_OP_(LessEqual, return lhs <= rhs);
    NOA_BINARY_OP_(Greater, return lhs > rhs);
    NOA_BINARY_OP_(GreaterEqual, return lhs >= rhs);
    NOA_BINARY_OP_(LogicalAnd, return lhs && rhs);
    NOA_BINARY_OP_(LogicalOr, return lhs || rhs);
    NOA_BINARY_OP_(Min, return min(lhs, rhs));
    NOA_BINARY_OP_(Max, return max(lhs, rhs));
    NOA_BINARY_OP_(Plus, return lhs + rhs);
    NOA_BINARY_OP_(Minus, return lhs - rhs);
    NOA_BINARY_OP_(Multiply, return lhs * rhs);
    NOA_BINARY_OP_(Divide, return lhs / rhs);
    NOA_BINARY_OP_(Modulo, return lhs % rhs);
    NOA_BINARY_OP_(MultiplyConjugate, return lhs * conj(rhs));
    NOA_BINARY_OP_(DistanceSquared, auto tmp = lhs - rhs; return tmp * tmp);
    NOA_BINARY_OP_(DivideSafe,
                   if constexpr (nt::real_or_complex<T, U>) {
                       constexpr auto epsilon = std::numeric_limits<nt::value_type_t<U>>::epsilon();
                       return abs(rhs) < epsilon ? T{} : lhs / rhs;
                   } else if constexpr (nt::integer<T, U>) {
                       return rhs == 0 ? T{} : lhs / rhs;
                   } else {
                       static_assert(nt::always_false<T>);
                   });
    #undef NOA_BINARY_OP_

    #define NOA_TRINARY_OP_(name, src_op)                                                           \
    struct name {                                                                                   \
        using allow_vectorization = bool;                                                           \
        template<typename T, typename U, typename V>                                                \
        NOA_HD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {          \
            src_op;                                                                                 \
        }                                                                                           \
        template<typename T, typename U, typename V, typename W>                                    \
        NOA_HD constexpr void operator()(const T& lhs, const U& mhs, const V& rhs, W& dst) const {  \
            dst = static_cast<W>((*this)(lhs, mhs, rhs));                                           \
        }                                                                                           \
    }
    NOA_TRINARY_OP_(Within, return mhs < lhs and lhs < rhs);
    NOA_TRINARY_OP_(WithinEqual, return mhs <= lhs and lhs <= rhs);
    NOA_TRINARY_OP_(Clamp, return clamp(lhs, mhs, rhs));
    NOA_TRINARY_OP_(PlusMinus, return lhs + mhs - rhs);
    NOA_TRINARY_OP_(PlusMultiply, return (lhs + mhs) * rhs);
    NOA_TRINARY_OP_(PlusDivide, return (lhs + mhs) / rhs);
    NOA_TRINARY_OP_(MinusPlus, return lhs - mhs + rhs);
    NOA_TRINARY_OP_(MinusMultiply, return (lhs - mhs) * rhs);
    NOA_TRINARY_OP_(MinusDivide, return (lhs - mhs) / rhs);
    NOA_TRINARY_OP_(MultiplyPlus, return lhs * mhs + rhs);
    NOA_TRINARY_OP_(MultiplyMinus, return lhs * mhs - rhs);
    NOA_TRINARY_OP_(MultiplyDivide, return (lhs * mhs) / rhs);
    NOA_TRINARY_OP_(DividePlus, return lhs / mhs + rhs);
    NOA_TRINARY_OP_(DivideMinus, return lhs / mhs - rhs);
    NOA_TRINARY_OP_(DivideMultiply, return lhs / mhs * rhs);
    NOA_TRINARY_OP_(DivideEpsilon, return lhs / (mhs + rhs));
    #undef NOA_TRINARY_OP_

    template<typename T>
    struct Fill {
        using allow_vectorization = bool;
        T value;

        template<typename U>
        NOA_HD constexpr void operator()(U& dst) const {
            dst = static_cast<U>(value);
        }
    };

    template<typename T>
    struct Scale {
        using allow_vectorization = bool;
        T value;

        template<typename U>
        NOA_HD constexpr void operator()(U& dst) const {
            dst *= value;
        }
    };

    struct ZeroInitialize {
        using allow_vectorization = bool;

        template<typename... U>
        NOA_HD constexpr void operator()(U&... dst) const {
            ((dst = U{}), ...);
        }
    };

    struct Cast {
        using allow_vectorization = bool;
        bool clamp{};

        template<typename T, typename U> requires nt::compatible_or_spectrum_types<T, U>
        NOA_HD constexpr void operator()(const T& src, U& dst) const {
            if constexpr (nt::complex<T> and nt::real<U>) {
                auto ps = abs_squared(src);
                dst = clamp ? clamp_cast<U>(ps) : static_cast<U>(ps);
            } else {
                dst = clamp ? clamp_cast<U>(src) : static_cast<U>(src);
            }
        }
    };

    struct Pow {
        using allow_vectorization = bool;

        template<typename T>
        NOA_HD constexpr auto operator()(const T& x, const T& e) const {
            return pow(x, e);
        }
        template<typename T, typename U>
        NOA_HD constexpr void operator()(const T& x, const T& e, U& dst) const {
            dst = static_cast<U>((*this)(x, e));
        }
    };

    struct ComplexFuse {
        using allow_vectorization = bool;

        template<typename C>
        NOA_HD constexpr void operator()(auto r, auto i, C& c) {
            c = C::from_values(r, i);
        }
    };
    struct ComplexDecompose {
        using allow_vectorization = bool;

        template<typename R, typename I>
        NOA_HD constexpr void operator()(const auto& c, R& r, I& i) {
            r = static_cast<R>(c.real);
            i = static_cast<I>(c.imag);
        }
    };

    struct NormalizeMinMax {
        using allow_vectorization = bool;

        template<typename T>
        NOA_HD constexpr auto operator()(const T& value, const T& min, const T& max) {
            return (value - min) / (max - min);
        }
        template<typename T, typename U>
        NOA_HD constexpr void operator()(const T& value, const T& min, const T& max, U& output) {
            output = static_cast<U>((*this)(value, min, max));
        }
    };
    struct NormalizeMeanStddev {
        using allow_vectorization = bool;

        template<typename T, typename U>
        NOA_HD constexpr auto operator()(const T& value, const T& mean, const U& stddev) {
            return (value - mean) / stddev;
        }
        template<typename T, typename U, typename V>
        NOA_HD constexpr void operator()(const T& value, const T& mean, const U& stddev, V& output) {
            output = static_cast<V>((*this)(value, mean, stddev));
        }
    };
    struct NormalizeNorm {
        using allow_vectorization = bool;

        template<typename T, typename U>
        NOA_HD constexpr void operator()(const T& value, const U& norm) {
            return value / norm;
        }
        template<typename T, typename U, typename V>
        NOA_HD constexpr void operator()(const T& value, const U& norm, V& output) {
            output = static_cast<V>((*this)(value, norm));
        }
    };
};
