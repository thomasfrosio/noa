#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/traits/Numerics.hpp"
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
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T(1) - lhs; }
    };

    struct inverse_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return T(1) / lhs; }
    };

    struct square_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return lhs * lhs; }
    };

    struct round_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ::noa::math::round(lhs); }
    };

    struct rint_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ::noa::math::rint(lhs); }
    };

    struct ceil_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ::noa::math::ceil(lhs); }
    };

    struct floor_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ::noa::math::floor(lhs); }
    };

    struct trunc_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return ::noa::math::trunc(lhs); }
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
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::sqrt(x); }
    };

    struct rsqrt_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::rsqrt(x); }
    };

    struct exp_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::exp(x); }
    };

    struct log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::log(x); }
    };

    struct abs_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::abs(x); }
    };

    struct cos_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::cos(x); }
    };

    struct sin_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::sin(x); }
    };

    struct normalize_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::normalize(x); }
    };

    struct real_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::real(x); }
    };

    struct imag_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::imag(x); }
    };

    struct conj_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::conj(x); }
    };

    struct abs_squared_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            if constexpr (nt::is_complex_v<T>) {
                return ::noa::math::abs_squared(x);
            } else {
                auto tmp = ::noa::math::abs(x);
                return tmp * tmp;
            }
        }
    };

    struct abs_one_log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return ::noa::math::log(value_t{1} + ::noa::math::abs(x));
        }
    };

    struct one_log_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const {
            using value_t = nt::value_type_t<T>;
            return ::noa::math::log(value_t{1} + x);
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
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * ::noa::math::conj(rhs); }
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
                const epsilon_t epsilon = ::noa::math::Limits<epsilon_t>::epsilon();
                #else
                constexpr epsilon_t epsilon = ::noa::math::Limits<epsilon_t>::epsilon();
                #endif
                return ::noa::math::abs(rhs) < epsilon ? T{0} : lhs / rhs;
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
        NOA_FHD constexpr auto operator()(const T& x, const T& e) const { return ::noa::math::pow(x, e); }
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
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return ::noa::math::min(lhs, rhs); }
    };

    struct max_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return ::noa::math::max(lhs, rhs); }
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
            return ::noa::math::clamp(lhs, low, high);
        }
    };

    // -- Find offset --

    struct first_min_t {
        template<typename Value, typename Offset>
        NOA_FHD constexpr auto operator()(
                const noa::Pair<Value, Offset>& current,
                const noa::Pair<Value, Offset>& candidate
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
                const noa::Pair<Value, Offset>& current,
                const noa::Pair<Value, Offset>& candidate
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
                const noa::Pair<Value, Offset>& current,
                const noa::Pair<Value, Offset>& candidate
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
                const noa::Pair<Value, Offset>& current,
                const noa::Pair<Value, Offset>& candidate
        ) const noexcept {
            if (candidate.first > current.first ||
                (current.first == candidate.first && candidate.second > current.second))
                return candidate;
            return current;
        }
    };
}
