#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/types/Complex.hpp"

namespace noa {
    // -- Unary operators -- //

    struct Copy {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs; }
    };

    struct Negate {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return -lhs; }
    };

    struct OneMinus {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T{1} - lhs; }
    };

    struct Inverse {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T{1} / lhs; }
    };

    struct Square {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs * lhs; }
    };

    struct Round {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return round(lhs); }
    };

    struct Rint {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return rint(lhs); }
    };

    struct Ceil {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ceil(lhs); }
    };

    struct Floor {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return floor(lhs); }
    };

    struct Trunc {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return trunc(lhs); }
    };

    struct NonZero {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs != T(0); }
    };

    struct LogicalNot {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return !lhs; }
    };

    struct Sqrt {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return sqrt(x); }
    };

    struct Rsqrt {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return rsqrt(x); }
    };

    struct Exp {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return exp(x); }
    };

    struct Log {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return log(x); }
    };

    struct Abs {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return abs(x); }
    };

    struct Cos {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return cos(x); }
    };

    struct Sin {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return sin(x); }
    };

    struct Normalize {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return normalize(x); }
    };

    struct Real {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return real(x); }
    };

    struct Imag {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return imag(x); }
    };

    struct Conj {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return conj(x); }
    };

    struct AbsSquared {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            if constexpr (nt::is_complex_v<T>) {
                return abs_squared(x);
            } else {
                auto tmp = abs(x);
                return tmp * tmp;
            }
        }
    };

    struct AbsOneLog {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return log(value_t{1} + abs(x));
        }
    };

    struct OneLog {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return log(value_t{1} + x);
        }
    };

    // -- Binary operators -- //

    struct Plus {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs + rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs + mhs + rhs;
        }
    };

    struct Minus {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs - rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs - mhs - rhs;
        }
    };

    struct Multiply {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs * rhs;
        }
    };

    struct MultiplyConjugate {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * conj(rhs); }
    };

    struct Divide {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs / rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs / mhs) / rhs;
        }
    };

    struct Modulo {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs % rhs; }
    };

    // If divisor is too close to zero, do not divide and set the output to zero instead.
    struct DivideSafe {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const {
            if constexpr (nt::are_real_or_complex_v<T, U>) {
                using epsilon_t = nt::value_type_t<U>;
                #if defined(__CUDA_ARCH__)
                const epsilon_t epsilon = std::numeric_limits<epsilon_t>::epsilon();
                #else
                constexpr epsilon_t epsilon = std::numeric_limits<epsilon_t>::epsilon();
                #endif
                return abs(rhs) < epsilon ? T{0} : lhs / rhs;
            } else if constexpr (nt::are_int_v<T, U>) {
                return rhs == 0 ? T{0} : T(lhs / rhs); // short is implicitly promoted to int so cast it back
            } else {
                static_assert(nt::always_false_v<T>);
            }
        }
    };

    struct DistanceSquared {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const {
            const auto tmp = lhs - rhs;
            return tmp * tmp;
        }
    };

    struct Pow {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x, const T& e) const { return pow(x, e); }
    };

    struct Equal {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs == rhs; }
    };

    struct NotEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs != rhs; }
    };

    struct Less {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs < rhs; }
    };

    struct LessEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs <= rhs; }
    };

    struct Greater {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs > rhs; }
    };

    struct GreaterEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs >= rhs; }
    };

    struct LogicalAnd {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs && rhs; }
    };

    struct LogicalOr {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs || rhs; }
    };

    struct Min {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return min(lhs, rhs); }
    };

    struct Max {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return max(lhs, rhs); }
    };

    // -- Trinary operators -- //

    struct PlusMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs + mhs - rhs;
        }
    };

    struct PlusMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) * rhs;
        }
    };

    struct PlusDivide {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) / rhs;
        }
    };

    struct MinusPlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs - mhs + rhs;
        }
    };

    struct MinusMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs - mhs) * rhs;
        }
    };

    struct MinusDivide {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs - mhs) / rhs;
        }
    };

    struct MultiplyPlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs + rhs;
        }
    };

    struct MultiplyMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs - rhs;
        }
    };

    struct MultiplyDivide {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs * mhs) / rhs;
        }
    };

    struct DividePlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs + rhs;
        }
    };

    struct DivideMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs - rhs;
        }
    };

    struct DivideMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs * rhs;
        }
    };

    struct DivideEpsilon {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& input, const U& div, const V& epsilon) const {
            return input / (div + epsilon);
        }
    };

    struct Within {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low < lhs && lhs < high;
        }
    };

    struct WithinEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low <= lhs && lhs <= high;
        }
    };

    struct Clamp {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& low, const T& high) const {
            return clamp(lhs, low, high);
        }
    };

    // -- Find offset --

    struct FirstMin {
        template<typename Value, typename Offset>
        NOA_FHD constexpr auto operator()(
                const Pair<Value, Offset>& current,
                const Pair<Value, Offset>& candidate
        ) const noexcept {
            if (candidate.first < current.first ||
                (current.first == candidate.first && candidate.second < current.second))
                return candidate;
            return current;
        }
    };

    struct FirstMax {
        template<typename Value, typename Offset>
        NOA_FHD constexpr auto operator()(
                const Pair<Value, Offset>& current,
                const Pair<Value, Offset>& candidate
        ) const noexcept {
            if (candidate.first > current.first ||
                (current.first == candidate.first && candidate.second < current.second))
                return candidate;
            return current;
        }
    };

    struct LastMin {
        template<typename Value, typename Offset>
        NOA_FHD constexpr auto operator()(
                const Pair<Value, Offset>& current,
                const Pair<Value, Offset>& candidate
        ) const noexcept {
            if (candidate.first < current.first ||
                (current.first == candidate.first && candidate.second > current.second))
                return candidate;
            return current;
        }
    };

    struct LastMax {
        template<typename Value, typename Offset>
        NOA_FHD constexpr auto operator()(
                const Pair<Value, Offset>& current,
                const Pair<Value, Offset>& candidate
        ) const noexcept {
            if (candidate.first > current.first ||
                (current.first == candidate.first && candidate.second > current.second))
                return candidate;
            return current;
        }
    };

    template<typename Value>
    struct Arange {
        Value start{0};
        Value step{1};

        template<typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
        [[nodiscard]] NOA_HD constexpr Value operator()(Index index) const noexcept {
            return start + static_cast<Value>(index) * step;
        }
    };

    template<typename Value, typename Index>
    struct Linspace {
        Value start;
        Value step;
        Value stop;
        Index index_end;
        bool endpoint;

        NOA_HD static constexpr auto from_range(
                Value start,
                Value stop,
                const Index& size,
                bool endpoint = true
        ) -> Linspace {
            Linspace linspace;
            linspace.start = start;
            linspace.stop = stop;
            linspace.start = start;
            linspace.index_end = min(Index{0}, size - 1);
            linspace.endpoint = endpoint;

            const auto count = size - static_cast<Index>(endpoint);
            const auto delta = stop - start;
            linspace.step = delta / static_cast<Value>(count);
            return linspace;
        }

        [[nodiscard]] NOA_HD constexpr Value operator()(Index i) const noexcept {
            return endpoint && i == index_end ? stop : start + static_cast<Value>(i) * step;
        }
    };
}
