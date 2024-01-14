#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/types/Complex.hpp"


/**
 * noa::ewise(a, b, [](f64 ia) { return ia * 2; });
 * noa::ewise(noa::zip(a, c), b, [](auto i) { auto [ia, ic] = i; return ia * 2 + ic; });
 * noa::ewise(noa::zip(a, c), noa::zip(b, d), [](auto i) { auto [ia, ic] = i; return Tuple{ia * 2 + ic, ic * 2}; });
 *
 * out = cos(plus(lhs, rhs));
 * plus(lhs, rhs, out);
 * cos(out);
 *
 *
 * Plus(Tuple<Ts...> tuple) {
 *     return tuple.apply([](auto&& args...) { return (args + ...); });
 * }

 * Plus(const auto&&... inputs, auto& output) {
 *     output = (inputs + ...);
 * }
 *
 *     constexpr auto plus(const auto& input) {
 *

struct ReduceMaskEwise {
    constexpr void init(Tuple<const f32&, const i32&>& input, Tuple<f64, f32> reduced) {
        auto [value, mask] = input;
        if (mask > 0) {
            auto& [sum, max] = reduced;
            sum += static_cast<f64>(value);
            max = std::max(max, value);
        }
        return reduced;
    }
    constexpr auto join(Tuple<f64, f32>& lhs, Tuple<f64, f32>& rhs) {
        auto& [lhs_sum, lhs_max] = lhs;
        auto& [rhs_sum, rhs_max] = rhs;
        return Tuple{lhs_sum + rhs_sum, std::max(lhs_mas, rhs_max)};
    }
    // default .final()
};


 *
 * noa::ewise(a, b, [](f64 ia, f64& ib) { ib = ia * 2; });
 * noa::ewise(noa::zip(a, c), b, [](auto ia, auto ic, auto& ib) { ib = ia * 2 + ic; });
 * noa::ewise(noa::zip(a, c), noa::zip(b, d), [](auto ia, auto ic, auto& ib, auto& id) { ib = ia * 2 + ic; id = ic * 2; });
 */

namespace noa {
    // Unary operators

    struct Copy { constexpr void operator()(const auto& src, auto& dst) const { dst = src; } };
    struct Negate { constexpr void operator()(const auto& src, auto& dst) const { dst = -src; } };
    struct OneMinus { template<typename T> constexpr void operator()(const T& src, auto& dst) const { dst = T{1} - src; } };
    struct Inverse { template<typename T> constexpr void operator()(const T& src, auto& dst) const { dst = T{1} / src; } };
    struct Square { constexpr void operator()(const auto& src, auto& dst) const { dst = src * src; } };
    struct Round { constexpr void operator()(const auto& src, auto& dst) const { dst = round(src); } };
    struct Rint { constexpr void operator()(const auto& src, auto& dst) const { dst = rint(src); } };
    struct Ceil { constexpr void operator()(const auto& src, auto& dst) const { dst = ceil(src); } };
    struct Floor { constexpr void operator()(const auto& src, auto& dst) const { dst = floor(src); } };
    struct Trunc { constexpr void operator()(const auto& src, auto& dst) const { dst = trunc(src); } };
    struct NonZero { template<typename T> constexpr void operator()(const T& src, auto& dst) const { dst = src != T{}; } };
    struct LogicalNot { constexpr void operator()(const auto& src, auto& dst) const { dst = !src; } };
    struct Sqrt { constexpr void operator()(const auto& src, auto& dst) const { dst = sqrt(src); } };
    struct Rsqrt { constexpr void operator()(const auto& src, auto& dst) const { dst = rsqrt(src); } };
    struct Exp { constexpr void operator()(const auto& src, auto& dst) const { dst = exp(src); } };
    struct Log { constexpr void operator()(const auto& src, auto& dst) const { dst = log(src); } };
    struct Abs { constexpr void operator()(const auto& src, auto& dst) const { dst = abs(src); } };
    struct Cos { constexpr void operator()(const auto& src, auto& dst) const { dst = cos(src); } };
    struct Sin { constexpr void operator()(const auto& src, auto& dst) const { dst = sin(src); } };
    struct Normalize { constexpr void operator()(const auto& src, auto& dst) const { dst = normalize(src); } };
    struct Real { constexpr void operator()(const auto& src, auto& dst) const { dst = real(src); } };
    struct Imag { constexpr void operator()(const auto& src, auto& dst) const { dst = imag(src); } };
    struct Conj { constexpr void operator()(const auto& src, auto& dst) const { dst = conj(src); } };
    struct AbsOneLog { constexpr void operator()(const auto& src, auto& dst) const { dst = log(nt::value_type_t<decltype(src)>{1} + abs(src));} };
    struct OneLog { constexpr void operator()(const auto& src, auto& dst) const { dst = log(nt::value_type_t<decltype(src)>{1} + src); } };

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
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs != rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct Less {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs < rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct LessEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs <= rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct Greater {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs > rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct GreaterEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs >= rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct LogicalAnd {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs && rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct LogicalOr {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs || rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct Min {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return min(lhs, rhs); }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    struct Max {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return max(lhs, rhs); }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs, V& out) const {
            out = static_cast<V>((*this)(lhs, rhs));
        }
    };

    // -- Trinary operators -- //

    struct PlusMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs + mhs - rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct PlusMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) * rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct PlusDivide {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) / rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct MinusPlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs - mhs + rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& lhs, const U& rhs, U& sum) const {
            sum += static_cast<U>(lhs - rhs);
        }

        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced = ireduced + reduced;
        }
    };

    struct MinusMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs - mhs) * rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct MinusDivide {
        template<typename T>
        constexpr auto operator()(const auto& lhs, const auto& mhs, const auto& rhs, T& out) const {
            out = static_cast<T>((lhs - mhs) / rhs);
        }
    };

    struct MultiplyPlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs + rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, const U& mask, U& sum) const {
            sum += static_cast<U>(value * mask);
        }

        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced = ireduced + reduced;
        }
    };

    struct MultiplyMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs - rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct MultiplyDivide {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs * mhs) / rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct DividePlus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs + rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct DivideMinus {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs - rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct DivideMultiply {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs * rhs;
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct DivideEpsilon {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& input, const U& div, const V& epsilon) const {
            return input / (div + epsilon);
        }

        template<typename T, typename U, typename V, typename W>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs, W& out) const {
            out = static_cast<W>((*this)(lhs, mhs, rhs));
        }
    };

    struct Within {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low < lhs && lhs < high;
        }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high, V& out) const {
            out = static_cast<V>((*this)(lhs, low, high));
        }
    };

    struct WithinEqual {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low <= lhs && lhs <= high;
        }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high, V& out) const {
            out = static_cast<V>((*this)(lhs, low, high));
        }
    };

    struct Clamp {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& low, const T& high) const {
            return clamp(lhs, low, high);
        }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high, V& out) const {
            out = static_cast<V>((*this)(lhs, low, high));
        }
    };

    // -- Find offset --

    struct FirstMin {
        template<typename Value, typename Offset>
        constexpr auto operator()(const Pair<Value, Offset>& current, Pair<Value, Offset>& reduced) const noexcept {
            if (current.first < reduced.first || (current.first == reduced.first && current.second < reduced.second))
                reduced = current;
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
