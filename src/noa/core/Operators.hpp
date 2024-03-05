#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/types/Pair.hpp"
#include "noa/core/math/Comparison.hpp"
#include "noa/core/math/Generic.hpp"

// Core element-wise operators
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
        using allow_vectorization = bool;                                               \
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
        using allow_vectorization = bool;                                                           \
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
        using allow_vectorization = bool;
        T value;

        template<typename U>
        NOA_HD constexpr void operator()(U& dst) const {
            dst = static_cast<U>(value);
        }
    };

    struct Cast {
        using allow_vectorization = bool;
        bool clamp{};

        template<typename T>
        NOA_HD constexpr void operator()(const auto& src, T& dst) const {
            dst = clamp ? clamp_cast<T>(src) : static_cast<T>(src);
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

// Reduction operators
namespace noa {
    struct ReduceSum {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(value);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& ireduced, T& reduced) const {
            reduced += ireduced;
        }
    };

    template<typename S> requires nt::is_scalar_v<S>
    struct ReduceMean : ReduceSum {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        S size;

        template<typename T, typename U>
        NOA_FHD constexpr void final(const T& sum, U& mean) const {
            mean = static_cast<U>(sum / size);
        }
    };

    struct ReduceL2Norm : ReduceSum {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        template<typename T>
        NOA_FHD constexpr void init(const auto& value, T& sum) const {
            sum += static_cast<T>(abs_squared(value));
        }

        template<typename T>
        NOA_FHD constexpr void final(const auto& sum, T& norm) const {
            norm = static_cast<T>(sqrt(sum));
        }
    };

    struct ReduceMin {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void operator()(const T& value, T& min) const {
            min = noa::min(value, min);
        }
    };

    struct ReduceMax {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void operator()(const T& value, T& min) const {
            min = noa::min(value, min);
        }
    };

    struct ReduceMinMax {
        using allow_vectorization = bool;

        template<typename T>
        NOA_FHD constexpr void init(const T& value, T& min, T& max) const {
            min = noa::min(value, min);
            max = noa::max(value, max);
        }
        template<typename T>
        NOA_FHD constexpr void join(const T& imin, const T& imax, T& min, T& max) const {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
        }
    };

    struct ReduceMinMaxSum {
        using allow_vectorization = bool;

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, T& min, T& max, U& sum) const {
            min = noa::min(value, min);
            max = noa::max(value, max);
            sum += static_cast<U>(value);
        }
        template<typename T, typename U>
        NOA_FHD constexpr void join(const T& imin, const T& imax, const U& isum, T& min, T& max, U& sum) const {
            min = noa::min(imin, min);
            max = noa::max(imax, max);
            sum += isum;
        }
    };

    template<typename R>
    struct ReduceVariance {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        R size{};

        template<typename T, typename U>
        NOA_FHD constexpr void init(const T& value, const U& mean, R& reduced) const noexcept {
            if constexpr (nt::is_complex_v<T>) {
                const auto distance = abs(static_cast<U>(value) - mean);
                reduced += static_cast<R>(distance * distance);
            } else {
                const auto distance = static_cast<U>(value) - mean;
                reduced += static_cast<R>(distance * distance);
            }
        }
        NOA_FHD constexpr void join(const R& ireduced, R& reduced) const noexcept {
            reduced += ireduced;
        }
        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& variance) const noexcept {
            variance = static_cast<T>(reduced / size);
        }
    };

    template<typename R>
    struct ReduceStddev : ReduceVariance<R> {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        template<typename T>
        NOA_FHD constexpr void final(const R& reduced, T& stddev) const noexcept {
            auto variance = reduced / this->size;
            stddev = static_cast<T>(sqrt(variance));
        }
    };

    template<typename T>
    struct ReduceRMSD {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        T size;

        template<typename I>
        NOA_FHD constexpr void init(const I& lhs, const I& rhs, T& sum) const {
            auto diff = static_cast<T>(lhs) - static_cast<T>(rhs);
            sum += diff * diff;
        }
        NOA_FHD constexpr void join(const T& isum, T& sum) const {
            sum += isum;
        }
        template<typename F>
        NOA_FHD constexpr void final(const T& sum, F& rmsd) const {
            rmsd = static_cast<F>(sqrt(sum / size));
        }
    };

    struct ReduceAllEqual {
        using allow_vectorization = bool;

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
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using reduced_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;

        NOA_FHD constexpr void init(const auto& input, reduced_type& sum, reduced_type& error) const noexcept {
            auto value = static_cast<reduced_type>(input);
            kahan_sum(value, sum, error);
        }

        NOA_FHD constexpr void join(
                const reduced_type& local_sum, const reduced_type& local_error,
                reduced_type& global_sum, reduced_type& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        NOA_FHD constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, F& final) {
            final = static_cast<F>(global_sum + global_error);
        }
    };

    template<typename T>
    struct ReduceAccurateMean : ReduceAccurateSum<T> {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
        using reduced_type = ReduceAccurateSum<T>::reduced_type;
        using mean_type = nt::value_type_t<reduced_type>;
        mean_type mean;

        template<typename F>
        NOA_FHD constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, F& final) {
            final = static_cast<F>((global_sum + global_error) / mean);
        }
    };

    struct ReduceAccurateL2Norm {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;

        NOA_FHD constexpr void init(const auto& input, f64& sum, f64& error) const noexcept {
            kahan_sum(static_cast<f64>(abs_squared(input)), sum, error);
        }

        NOA_FHD constexpr void join(
                const f64& local_sum, const f64& local_error, f64& global_sum, f64& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        template<typename F>
        NOA_FHD constexpr void final(const f64& global_sum, const f64& global_error, F& final) {
            final = static_cast<F>(sqrt(global_sum + global_error));
        }
    };

    /// Index-wise reduce operator to find the first minimum value.
    template<typename Accessor, typename Offset>
    struct ReduceFirstMin {
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
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
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
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
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
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
        using allow_vectorization = bool;
        using remove_defaulted_final = bool;
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
