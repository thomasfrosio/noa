#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

// Core operators
namespace noa {
    #define NOA_UNARY_OP_(name, src_op)                                     \
    struct name {                                                           \
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
    NOA_UNARY_OP_(NonZero, return src != T{});
    NOA_UNARY_OP_(LogicalNot, return !src);
    NOA_UNARY_OP_(Normalize, return normalize(src));
    NOA_UNARY_OP_(AbsOneLog, return log(nt::value_type_t<T>{1} + abs(src)));
    NOA_UNARY_OP_(OneLog, return log(nt::value_type_t<T>{1} + src));
    NOA_UNARY_OP_(AbsSquared,
                  if constexpr (nt::is_complex_v<T>) {
                      return abs_squared(src);
                  } else {
                      auto tmp = abs(src);
                      return tmp * tmp;
                  });
    #undef NOA_UNARY_OP_

    #define NOA_BINARY_OP_(name, src_op)                                                \
    struct name {                                                                       \
        template<typename T, typename U>                                                \
        NOA_HD constexpr auto operator()(const T& lhs, const U& rhs) const {            \
            src_op;                                                                     \
        }                                                                               \
        template<typename T, typename U, typename V>                                    \
        NOA_HD constexpr void operator()(const T& lhs, const U& rhs, V& dst) const {    \
            dst = static_cast<U>((*this)(lhs, rhs));                                    \
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
                   if constexpr (nt::are_real_or_complex_v<T, U>) {
                       constexpr auto epsilon = std::numeric_limits<nt::value_type_t<U>>::epsilon();
                       return abs(rhs) < epsilon ? T{} : lhs / rhs;
                   } else if constexpr (nt::are_int_v<T, U>) {
                       return rhs == 0 ? T{} : lhs / rhs;
                   } else {
                       static_assert(nt::always_false_v<T>);
                   });
    #undef NOA_BINARY_OP_

    #define NOA_TRINARY_OP_(name, src_op)                                                           \
    struct name {                                                                                   \
        template<typename T, typename U, typename V>                                                \
        NOA_HD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {          \
            src_op;                                                                                 \
        }                                                                                           \
        template<typename T, typename U, typename V, typename W>                                    \
        NOA_HD constexpr void operator()(const T& lhs, const U& mhs, const V& rhs, W& dst) const {  \
            dst = static_cast<U>((*this)(lhs, mhs, rhs));                                           \
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
        T value;
        template<typename U>
        NOA_HD constexpr void operator()(U& dst) const {
            dst = static_cast<U>(value);
        }
    };

    struct Cast {
        bool clamp{};
        template<typename T>
        NOA_HD constexpr void operator()(const auto& src, T& dst) const {
            dst = clamp ? clamp_cast<T>(src) : static_cast<T>(src);
        }
    };

    struct Pow {
        template<typename T>
        NOA_HD constexpr auto operator()(const T& x, const T& e) const {
            return pow(x, e);
        }
        template<typename T, typename U>
        NOA_HD constexpr void operator()(const T& x, const T& e, U& dst) const {
            dst = static_cast<U>((*this)(x, e));
        }
    };
}

// Reduction operators
namespace noa {
    struct ReduceSum {
        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(value);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
    };

    struct ReduceBinarySubtractAndSum {
        template<typename T>
        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, T& sum) const {
            sum += static_cast<T>(lhs - rhs);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
    };

    struct ReduceBinaryMultiplyAndSum {
        template<typename T>
        NOA_FHD constexpr void init(const auto& value, const auto& mask, T& sum) const {
            sum += static_cast<T>(value * mask);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
    };

    struct ReduceAllEqual {
        template<typename T>
        NOA_FHD constexpr void init(const auto& lhs, const auto& rhs, T& reduced) const {
            reduced = static_cast<T>(lhs == rhs);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            if (not ireduced)
                reduced = false;
        }
    };

    /// Accurate sum reduction operator for (complex) floating-points using Kahan summation, with Neumaier variation.
    template<typename T>
    struct ReduceAccurateSum {
        using remove_defaulted_final [[maybe_unused]] = bool; // just to make sure the final() function is picked up
        using reduced_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;

        constexpr void init(const auto& input, reduced_type& sum, reduced_type& error) const noexcept {
            auto value = static_cast<reduced_type>(input);
            auto sum_value = value + sum;
            if constexpr (nt::is_real_v<reduced_type>) {
                error += abs(sum) >= abs(value) ?
                         (sum - sum_value) + value :
                         (value - sum_value) + sum;
            } else if constexpr (nt::is_complex_v<reduced_type>) {
                for (i64 i = 0; i < 2; ++i) {
                    error[i] += abs(sum[i]) >= abs(value[i]) ?
                                (sum[i] - sum_value[i]) + value[i] :
                                (value[i] - sum_value[i]) + sum[i];
                }
            }
            sum = sum_value;
        }

        constexpr void join(
                const reduced_type& local_sum, const reduced_type& local_error,
                reduced_type& global_sum, reduced_type& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, auto& final) {
            final = static_cast<decltype(final)>(global_sum + global_error);
        }
    };

    template<typename T>
    struct ReduceAccurateVariance {
        using accurate_input_type = std::conditional_t<nt::is_complex_v<T>, c64, f64>;
        using reduced_type = nt::value_type_t<T>;
        accurate_input_type mean{};

        template<typename I>
        NOA_FHD constexpr void init(const I& input, reduced_type& output) const noexcept {
            const auto tmp = static_cast<accurate_input_type>(input);
            if constexpr (nt::is_complex_v<I>) {
                const auto distance = abs(tmp - mean);
                output += distance * distance;
            } else {
                const auto distance = tmp - mean;
                output += distance * distance;
            }
        }

        NOA_FHD constexpr void join(const reduced_type& to_reduce, reduced_type& output) const noexcept {
            output += to_reduce;
        }
    };

    /// Index-wise reduce operator to find the first minimum value.
    template<typename Accessor, typename Offset>
    struct ReduceFirstMin {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second < reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the first maximum value.
    template<typename Accessor, typename Offset>
    struct ReduceFirstMax {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second < reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the last minimum value.
    template<typename Accessor, typename Offset>
    struct ReduceLastMin {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second > reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the last maximum value.
    template<typename Accessor, typename Offset>
    struct ReduceLastMax {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second > reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };
}
