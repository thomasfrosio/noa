#pragma once

#include <cstdlib>
#include <cmath>

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Base.hpp"
#include "noa/core/math/Constant.hpp"

namespace noa {
    /// Returns the next power of 2.
    /// If x is a power of 2 or is equal to 1, returns x.
    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto next_power_of_2(T x) noexcept -> T {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    /// Rounds up to the nearest multiple of a number.
    /// \warning This should only be used for positive numbers, and the base should be greater than zero.
    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto next_multiple_of(T value, std::type_identity_t<T> base) noexcept -> T {
        return ((value + base - 1) / base) * base;
    }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_multiple_of(T value, std::type_identity_t<T> base) noexcept -> bool { return (value % base) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_even(T value) noexcept -> bool { return !(value % 2); }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_odd(T value) noexcept -> bool { return value % 2; }

    template<nt::uinteger T>
    [[nodiscard]] NOA_FHD constexpr auto is_power_of_2(T value) noexcept -> bool { return (value & (value - 1)) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto divide_up(T dividend, T divisor) noexcept -> T { return (dividend + divisor - 1) / divisor; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD auto round(T x) noexcept -> T { return x; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD auto rint(T x) noexcept -> T { return x; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto sign(T x) noexcept -> T { return x >= 0 ? 1 : -1; }

    template<typename T> requires (nt::integer<T> or nt::any_of<T, f32, f64>)
    [[nodiscard]] NOA_FHD constexpr auto abs(T x) noexcept -> T {
        if constexpr (nt::uinteger<T>) {
            return x;
        } else if constexpr (nt::integer<T>) {
            if constexpr (nt::almost_same_as<T, long>)
                return std::labs(x);
            else if constexpr (nt::almost_same_as<T, long long>)
                return std::llabs(x);
            else if constexpr (nt::almost_same_as<T, i8>)
                return static_cast<i8>(::abs(x));
            else if constexpr (nt::almost_same_as<T, i16>)
                return static_cast<i16>(::abs(x));
            else
                return std::abs(x);
        } else {
            return std::abs(x);
        }
    }
    template<nt::scalar T>
    [[nodiscard]] NOA_FHD constexpr auto abs_squared(T x) noexcept {
        return x * x;
    }

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr auto cast_or_abs_squared(const U& value) noexcept -> T {
        if constexpr (nt::complex<U> and nt::real<T>)
            return static_cast<T>(abs_squared(value));
        else
            return static_cast<T>(value);
    }

    template<nt::real_or_complex T>
    NOA_IHD constexpr void kahan_sum(T value, T& sum, T& error) noexcept {
        auto sum_value = value + sum;
        if constexpr (nt::real<T>) {
            error += abs(sum) >= abs(value) ?
                     (sum - sum_value) + value :
                     (value - sum_value) + sum;
        } else if constexpr (nt::complex<T>) {
            for (isize i = 0; i < 2; ++i) {
                error[i] += abs(sum[i]) >= abs(value[i]) ?
                            (sum[i] - sum_value[i]) + value[i] :
                            (value[i] - sum_value[i]) + sum[i];
            }
        }
        sum = sum_value;
    }

    template<typename T, typename U>
    requires ((nt::scalar<T, U> and nt::same_as<T, U>) or (nt::real_or_complex<T, U> and nt::same_value_type<T, U>))
    constexpr auto divide_safe(const T& lhs, const U& rhs) noexcept {
        if constexpr (nt::real_or_complex<T, U>) {
            constexpr auto epsilon = std::numeric_limits<nt::value_type_t<U>>::epsilon();
            if constexpr (nt::complex<U>)
                return abs(rhs.real) < epsilon or abs(rhs.imag) < epsilon ? U{} : lhs / rhs;
            else
                return abs(rhs) < epsilon ? T{} : lhs / rhs;
        } else if constexpr (nt::integer<T, U>) {
            return rhs == 0 ? T{} : lhs / rhs;
        } else {
            static_assert(nt::always_false<T>);
        }
    }
}
