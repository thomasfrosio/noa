/// \file noa/common/Math.h
/// \brief Various mathematical functions for built-in types.
/// \author Thomas - ffyr2w
/// \date 20 Jul 2020

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"

// -- Unary operators -- //
namespace noa::math {
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
        NOA_FHD constexpr auto operator()(const T& lhs) const { return noa::math::round(lhs); }
    };

    struct rint_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return noa::math::rint(lhs); }
    };

    struct ceil_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return noa::math::ceil(lhs); }
    };

    struct floor_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return noa::math::floor(lhs); }
    };

    struct trunc_t {
        template<typename T>
        NOA_FHD constexpr auto operator()(const T& lhs) const { return noa::math::trunc(lhs); }
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
            auto tmp = ::noa::math::abs(x);
            return tmp * tmp;
        }
    };
}

// -- Binary operators -- //
namespace noa::math {
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

    struct multiply_conj_t {
        template<typename T, typename U>
        NOA_FHD constexpr auto operator()(const T& lhs, const U& rhs) const { return lhs * noa::math::conj(rhs); }
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
                return rhs == 0 ? T(0) : T(lhs / rhs); // short is implicitly promoted to int so cast it back
            } else {
                static_assert(::noa::traits::always_false_v<T>);
            }
            return T(0); // unreachable
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
}

// -- Trinary operators -- //
namespace noa::math {
    struct fma_t {
        template<typename T, typename U, typename V>
        NOA_FHD constexpr auto operator()(const T& input, const U& mul, const V& add) const {
            return input * mul + add;
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
}
