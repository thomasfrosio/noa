/// \file noa/common/Math.h
/// \brief Various mathematical functions for built-in types.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"

// -- Arithmetic operators -- //
namespace noa::math {
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

    struct plus_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs + rhs; }
    };

    struct minus_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs - rhs; }
    };

    struct multiply_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * rhs; }
    };

    struct divide_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs / rhs; }
    };

    struct modulo_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs % rhs; }
    };

    struct divide_safe_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const {
            if constexpr (::noa::traits::is_float_v<U>) {
                return ::noa::math::abs(rhs) < ::noa::math::Limits<U>::epsilon() ? T(0) : lhs / rhs;
            } else if constexpr (std::is_integral_v<U>) {
                return rhs == 0 ? 0 : lhs / rhs;
            } else {
                static_assert(::noa::traits::always_false_v<T>);
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

    struct fma_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& input, const U& mul, const V& add) const {
            return input * mul + add;
        }
    };
}

// -- Comparison operators -- //
namespace noa::math {
    template<typename R = bool>
    struct equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs == rhs); }
    };

    template<typename R = bool>
    struct not_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs != rhs); }
    };

    template<typename R = bool>
    struct less_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs < rhs); }
    };

    template<typename R = bool>
    struct less_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs <= rhs); }
    };

    template<typename R = bool>
    struct greater_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs > rhs); }
    };

    template<typename R = bool>
    struct greater_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return static_cast<R>(lhs >= rhs); }
    };

    template<typename R = bool>
    struct within_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return static_cast<R>(low < lhs && lhs < high);
        }
    };

    template<typename R = bool>
    struct within_equal_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& low, const U& high) const {
            return static_cast<R>(low <= lhs && lhs <= high);
        }
    };

    template<typename R = bool>
    struct not_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return static_cast<R>(!lhs); }
    };

    struct min_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return ::noa::math::min(lhs, rhs); }
    };

    struct max_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& rhs) const { return ::noa::math::max(lhs, rhs); }
    };

    struct clamp_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs, const T& low, const T& high) const {
            return ::noa::math::clamp(lhs, low, high);
        }
    };
}

// -- Generic math operators -- //
namespace noa::math {
    struct sqrt_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::sqrt(x); }
    };

    struct rsqrt_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x) const { return ::noa::math::rsqrt(x); }
    };

    struct pow_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& x, const T& e) const { return ::noa::math::pow(x, e); }
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
}