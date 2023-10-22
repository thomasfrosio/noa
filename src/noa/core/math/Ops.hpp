#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/types/Complex.hpp"

// -- Unary operators -- //
namespace noa {
    struct copy_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs; }
    };

    struct negate_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return -lhs; }
    };

    struct one_minus_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T{1} - lhs; }
    };

    struct inverse_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T{1} / lhs; }
    };

    struct square_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs * lhs; }
    };

    struct round_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return round(lhs); }
    };

    struct rint_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return rint(lhs); }
    };

    struct ceil_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ceil(lhs); }
    };

    struct floor_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return floor(lhs); }
    };

    struct trunc_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return trunc(lhs); }
    };

    struct nonzero_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs != T(0); }
    };

    struct logical_not_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return !lhs; }
    };

    struct sqrt_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return sqrt(x); }
    };

    struct rsqrt_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return rsqrt(x); }
    };

    struct exp_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return exp(x); }
    };

    struct log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return log(x); }
    };

    struct abs_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return abs(x); }
    };

    struct cos_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return cos(x); }
    };

    struct sin_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return sin(x); }
    };

    struct normalize_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return normalize(x); }
    };

    struct real_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return real(x); }
    };

    struct imag_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return imag(x); }
    };

    struct conj_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return conj(x); }
    };

    struct abs_squared_t {
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

    struct abs_one_log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return log(value_t{1} + abs(x));
        }
    };

    struct one_log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return log(value_t{1} + x);
        }
    };

    // -- Binary operators -- //

    struct plus_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs + rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs + mhs + rhs;
        }
    };

    struct minus_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs - rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs - mhs - rhs;
        }
    };

    struct multiply_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs * rhs;
        }
    };

    struct multiply_conj_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * conj(rhs); }
    };

    struct divide_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs / rhs; }

        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs / mhs) / rhs;
        }
    };

    struct modulo_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs % rhs; }
    };

    // If divisor is too close to zero, do not divide and set the output to zero instead.
    struct divide_safe_t {
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

    struct dist2_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const {
            const auto tmp = lhs - rhs;
            return tmp * tmp;
        }
    };

    struct pow_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x, const T& e) const { return pow(x, e); }
    };

    struct equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs == rhs; }
    };

    struct not_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs != rhs; }
    };

    struct less_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs < rhs; }
    };

    struct less_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs <= rhs; }
    };

    struct greater_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs > rhs; }
    };

    struct greater_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs >= rhs; }
    };

    struct logical_and_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs && rhs; }
    };

    struct logical_or_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return lhs || rhs; }
    };

    struct min_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return min(lhs, rhs); }
    };

    struct max_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return max(lhs, rhs); }
    };

    // -- Trinary operators -- //

    struct plus_minus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs + mhs - rhs;
        }
    };

    struct plus_multiply_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) * rhs;
        }
    };

    struct plus_divide_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs + mhs) / rhs;
        }
    };

    struct minus_plus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs - mhs + rhs;
        }
    };

    struct minus_multiply_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs - mhs) * rhs;
        }
    };

    struct minus_divide_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs - mhs) / rhs;
        }
    };

    struct multiply_plus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs + rhs;
        }
    };

    using fma_t = multiply_plus_t;

    struct multiply_minus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs * mhs - rhs;
        }
    };

    struct multiply_divide_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return (lhs * mhs) / rhs;
        }
    };

    struct divide_plus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs + rhs;
        }
    };

    struct divide_minus_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs - rhs;
        }
    };

    struct divide_multiply_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& mhs, const V& rhs) const {
            return lhs / mhs * rhs;
        }
    };

    struct divide_epsilon_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& input, const U& div, const V& epsilon) const {
            return input / (div + epsilon);
        }
    };

    struct within_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low < lhs && lhs < high;
        }
    };

    struct within_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return low <= lhs && lhs <= high;
        }
    };

    struct clamp_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& low, const T& high) const {
            return clamp(lhs, low, high);
        }
    };

    // -- Find offset --

    struct first_min_t {
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

    struct first_max_t {
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

    struct last_min_t {
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

    struct last_max_t {
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
}

namespace noa {
    template<typename Value>
    struct arange_t {
        Value start{0};
        Value step{1};

        template<typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
        [[nodiscard]] NOA_HD constexpr Value operator()(Index index) const noexcept {
            return start + static_cast<Value>(index) * step;
        }
    };

    template<typename Value, typename Index>
    struct linspace_t {
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
        ) -> linspace_t {
            linspace_t linspace;
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
