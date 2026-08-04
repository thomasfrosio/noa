#pragma once

#include <complex>
#include <cfloat>

#include "noa/base/Bounds.hpp"
#include "noa/base/Config.hpp"
#include "noa/base/Half.hpp"
#include "noa/base/Math.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/Traits.hpp"

namespace noa::inline types {
    template<typename, usize, usize>
    class Vec;

    /// Complex number (aggregate type of two floating-point values).
    /// \details This type differs/expands on std::complex in the following ways:
    ///     - it works on CUDA by default (although, std::complex is less of any issue with recent CUDA version).
    ///     - it is an aggregate, with the member variables .real and .imag.
    ///     - it supports the subscript operator[] to access the real and imaginary components.
    /// \note While we could tend to prefer leaving this type uninitialized, in order to support direct initialization
    ///       from a scalar, e.g. Complex{1.}, we need to have the member variables zero-initialized (same as std::complex).
    template<typename T>
    class alignas(sizeof(T) * 2) Complex {
    public:
        static_assert(nt::real<T>);
        T real{};
        T imag{};

    public:
        using value_type = T;
        using mutable_value_type = T;

    public: // Component accesses
        static constexpr usize SIZE = 2;
        static constexpr isize SSIZE = 2;

        template<std::integral I>
        NOA_HD constexpr auto operator[](I i) noexcept -> value_type& {
            noa::bounds_check(SSIZE, i);
            if (i == I{1})
                return this->imag;
            else
                return this->real;
        }

        template<std::integral I>
        NOA_HD constexpr auto operator[](I i) const noexcept -> const value_type& {
            noa::bounds_check(SSIZE, i);
            if (i == I{1})
                return this->imag;
            else
                return this->real;
        }

    public: // Factory static functions
        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_real(U u) noexcept -> Complex {
            return {static_cast<value_type>(u), value_type{}};
        }

        template<typename U, typename V>
        [[nodiscard]] NOA_HD static constexpr auto from_values(U r, V i) noexcept -> Complex {
            return {static_cast<value_type>(r), static_cast<value_type>(i)};
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_complex(const Complex<U>& v) noexcept -> Complex {
            return {static_cast<value_type>(v.real),
                    static_cast<value_type>(v.imag)};
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_complex(const std::complex<U>& v) noexcept -> Complex {
            return {static_cast<value_type>(reinterpret_cast<const U(&)[2]>(v)[0]),
                    static_cast<value_type>(reinterpret_cast<const U(&)[2]>(v)[1])};
        }

        template<typename U, usize A>
        [[nodiscard]] NOA_HD static constexpr auto from_vec(const Vec<U, 2, A>& v) noexcept -> Complex {
            return {static_cast<value_type>(v[0]),
                    static_cast<value_type>(v[1])};
        }

        template<typename U>
        [[nodiscard]] NOA_HD static constexpr auto from_pointer(U* ptr) noexcept -> Complex {
            return {static_cast<value_type>(ptr[0]),
                    static_cast<value_type>(ptr[1])};
        }

    public: // Conversion operators
        /// Explicit conversion constructor to a Complex with different precision:
        /// \c static_cast<Complex<U>>(Complex<T>{})
        template<typename U>
        [[nodiscard]] NOA_HD constexpr explicit operator Complex<U>() const noexcept {
            return Complex<U>::from_complex(*this);
        }

    public: // Assignment operators
        NOA_HD constexpr auto operator=(value_type v) noexcept -> Complex& {
            this->real = v;
            this->imag = value_type{0};
            return *this;
        }

        NOA_HD constexpr auto operator+=(Complex rhs) noexcept -> Complex& {
            *this = *this + rhs;
            return *this;
        }
        NOA_HD constexpr auto operator-=(Complex rhs) noexcept -> Complex& {
            *this = *this - rhs;
            return *this;
        }
        NOA_HD constexpr auto operator*=(Complex rhs) noexcept -> Complex& {
            *this = *this * rhs;
            return *this;
        }
        NOA_HD constexpr auto operator/=(Complex rhs) noexcept -> Complex& {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr auto operator+=(value_type rhs) noexcept -> Complex& {
            *this = *this + rhs;
            return *this;
        }
        NOA_HD constexpr auto operator-=(value_type rhs) noexcept -> Complex& {
            *this = *this - rhs;
            return *this;
        }
        NOA_HD constexpr auto operator*=(value_type rhs) noexcept -> Complex& {
            *this = *this * rhs;
            return *this;
        }
        NOA_HD constexpr auto operator/=(value_type rhs) noexcept -> Complex& {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        [[nodiscard]] NOA_HD friend constexpr auto operator+(Complex v) noexcept -> Complex {
            return v;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator-(Complex v) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                *tmp = -(*tmp);
                return v;
            }
            #endif
            return {-v.real, -v.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator+(Complex lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp += *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real + rhs.real, lhs.imag + rhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator+(value_type lhs, Complex rhs) noexcept -> Complex {
            return {lhs + rhs.real, rhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator+(Complex lhs, value_type rhs) noexcept -> Complex {
            return {lhs.real + rhs, lhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator-(Complex lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real - rhs.real, lhs.imag - rhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator-(value_type lhs, Complex rhs) noexcept -> Complex {
            return {lhs - rhs.real, -rhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator-(Complex lhs, value_type rhs) noexcept -> Complex {
            return {lhs.real - rhs, lhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator*(Complex lhs, Complex rhs) noexcept -> Complex {
            if constexpr (std::is_same_v<value_type, f16>) {
                return (lhs.as<f16::arithmetic_type>() * rhs.as<f16::arithmetic_type>()).template as<value_type>();
            }
            return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                    lhs.real * rhs.imag + lhs.imag * rhs.real};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator*(value_type lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp *= __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs * rhs.real, lhs * rhs.imag};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator*(Complex lhs, value_type rhs) noexcept -> Complex {
            return rhs * lhs;
        }

        // Adapted from cuComplex.h
        // "This implementation guards against intermediate underflow and overflow
        // by scaling. Such guarded implementations are usually the default for
        // complex library implementations, with some also offering an unguarded,
        // faster version."
        [[nodiscard]] NOA_HD friend constexpr auto operator/(Complex lhs, Complex rhs) noexcept -> Complex {
            if constexpr (std::is_same_v<value_type, f16>) {
                return (lhs.as<f16::arithmetic_type>() / rhs.as<f16::arithmetic_type>()).template as<value_type>();
            }

            auto s = abs(rhs.real) + abs(rhs.imag);
            auto oos = value_type(1.0) / s;

            auto ars = lhs.real * oos;
            auto ais = lhs.imag * oos;
            auto brs = rhs.real * oos;
            auto bis = rhs.imag * oos;

            s = (brs * brs) + (bis * bis);
            oos = value_type(1.0) / s;

            return {((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos};
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator/(value_type lhs, Complex rhs) noexcept -> Complex {
            return Complex::from_real(lhs) / rhs;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator/(Complex lhs, value_type rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp /= __half2half2(rhs.native());
                return lhs;
            }
            #endif
            return {lhs.real / rhs, lhs.imag / rhs};
        }

        // -- Equality Operators -- //
        [[nodiscard]] NOA_HD friend constexpr auto operator==(Complex lhs, Complex rhs) noexcept -> bool {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>)
                return *reinterpret_cast<__half2*>(&lhs) == *reinterpret_cast<__half2*>(&rhs);
            #endif
            return lhs.real == rhs.real && lhs.imag == rhs.imag;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator==(value_type lhs, Complex rhs) noexcept -> bool {
            return Complex::from_real(lhs) == rhs;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator==(Complex lhs, value_type rhs) noexcept -> bool {
            return lhs == Complex::from_real(rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator!=(Complex lhs, Complex rhs) noexcept -> bool {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, f16>)
                return *reinterpret_cast<__half2*>(&lhs) != *reinterpret_cast<__half2*>(&rhs);
            #endif
            return !(lhs == rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator!=(value_type lhs, Complex rhs) noexcept -> bool {
            return Complex::from_real(lhs) != rhs;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator!=(Complex lhs, value_type rhs) noexcept -> bool {
            return lhs != Complex::from_real(rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator==(Complex lhs, std::complex<value_type> rhs) noexcept -> bool {
            return lhs == Complex::from_complex(rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator==(std::complex<value_type> lhs, Complex rhs) noexcept -> bool {
            return Complex::from_complex(lhs) == rhs;
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator!=(Complex lhs, std::complex<value_type> rhs) noexcept -> bool {
            return !(lhs == rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator!=(std::complex<value_type> lhs, Complex rhs) noexcept -> bool {
            return !(lhs == rhs);
        }

        [[nodiscard]] NOA_HD friend constexpr auto operator<(Complex lhs, Complex rhs) noexcept -> bool {
            return abs_squared(lhs) < abs_squared(rhs);
        }
        [[nodiscard]] NOA_HD friend constexpr auto operator<=(Complex lhs, Complex rhs) noexcept -> bool {
            return abs_squared(lhs) <= abs_squared(rhs);
        }
        [[nodiscard]] NOA_HD friend constexpr auto operator>(Complex lhs, Complex rhs) noexcept -> bool {
            return abs_squared(lhs) > abs_squared(rhs);
        }
        [[nodiscard]] NOA_HD friend constexpr auto operator>=(Complex lhs, Complex rhs) noexcept -> bool {
            return abs_squared(lhs) >= abs_squared(rhs);
        }

    public:
        [[nodiscard]] NOA_HD constexpr auto to_vec() const noexcept {
            return Vec<value_type, 2, 0>{real, imag};
        }

        template<typename U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept -> Complex<U> {
            return {static_cast<U>(real), static_cast<U>(imag)};
        }
    };

    using c16 = Complex<f16>;
    using c32 = Complex<f32>;
    using c64 = Complex<f64>;
}

namespace noa {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto real(Complex<T> x) noexcept -> T { return x.real; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto real(std::complex<T> x) noexcept -> T { return x.real(); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto imag(Complex<T> x) noexcept -> T { return x.imag; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto imag(std::complex<T> x) noexcept -> T { return x.imag(); }

    /// Returns the phase angle (in radians) of the complex number z.
    template<typename T>
    [[nodiscard]] NOA_FHD auto arg(Complex<T> x) noexcept -> T {
        return atan2(x.imag, x.real);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD auto sqrt(Complex<T> x) noexcept -> Complex<T> {
        #if defined(__CUDA_ARCH__)
        T& r = x.real;
        T& i = x.imag;
        if (r == T{}) {
            T t = sqrt(abs(i) / 2);
            return Complex<T>{t, i < T{} ? -t : t};
        } else {
            T t = sqrt(2 * (abs(x) + abs(r)));
            T u = t / 2;
            return r > T{} ?
                Complex<T>{u, i / t} :
                Complex<T>{abs(i) / t, i < T{} ? -u : u};
        }
        #else
        return Complex<T>::from_complex(std::sqrt(std::complex<f32>{x.real, x.imag}));
        #endif
    }

    template<typename T>
    [[nodiscard]] NOA_FHD auto phase(Complex<T> x) noexcept -> T {
        return arg(x);
    }

    /// Returns the magnitude of the complex number x.
    template<typename T>
    [[nodiscard]] NOA_FHD auto abs(Complex<T> x) noexcept -> T {
        return hypot(x.real, x.imag);
    }

    /// Returns the length-normalized of the complex number x to 1, reducing it to its phase.
    template<typename T>
    [[nodiscard]] NOA_FHD auto normalize(Complex<T> x) noexcept -> Complex<T> {
        if constexpr (std::is_same_v<T, f16>)
            return Complex<T>(normalize(x.template as<f16::arithmetic_type>()));
        T magnitude = abs(x);
        if (magnitude > T{0})
            magnitude = T{1} / magnitude;
        return x * magnitude;
    }

    /// Returns the squared magnitude of the complex number x.
    template<typename T>
    NOA_IHD auto abs_squared(Complex<T> x) noexcept -> T {
        if constexpr (std::is_same_v<T, f16>)
            return f16(abs_squared(x.template as<f16::arithmetic_type>()));
        return x.real * x.real + x.imag * x.imag;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto conj(Complex<T> x) noexcept -> Complex<T> {
        return {x.real, -x.imag};
    }

    /// Returns a complex number with magnitude length (should be positive) and phase angle theta.
    template<typename T>
    [[nodiscard]] NOA_FHD auto polar(T length, T theta) noexcept -> Complex<T> {
        return {length * cos(theta), length * sin(theta)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto dot(Complex<T> a, Complex<T> b) noexcept -> T {
        if constexpr (std::is_same_v<T, f16>)
            return fma(a[0], b[0], a[1] * b[1]);
        return a[0] * b[0] + a[1] * b[1];
    }

    /// Whether two complex floating-points are equal or almost equal to each other.
    template<i32 ULP = 2, typename T>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
        Complex<T> x, Complex<T> y,
        std::type_identity_t<T> epsilon = static_cast<T>(1e-6)
    ) noexcept -> bool {
        return allclose<ULP>(x.real, y.real, epsilon) and
               allclose<ULP>(x.imag, y.imag, epsilon);
    }
}

namespace noa::traits {
    template<> struct proclaim_is_complex<c16> : std::true_type {};
    template<> struct proclaim_is_complex<c32> : std::true_type {};
    template<> struct proclaim_is_complex<c64> : std::true_type {};

    template<> struct double_precision<c16> { using type = c64; };
    template<> struct double_precision<c32> { using type = c64; };
    template<> struct double_precision<c64> { using type = c64; };
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Complex<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Complex<T>& vec, FormatContext& ctx) const {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.real, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.imag, ctx);
            *out = ')';
            return out;
        }

        template<typename FormatContext>
        auto format(const noa::Complex<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.real, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.imag, ctx);
            *out = ')';
            return out;
        }
    };
}

namespace noa::inline types {
    template<typename T>
    auto operator<<(std::ostream& os, Complex<T> z) -> std::ostream& {
        os << fmt::format("({:.3f},{:.3f})", z.real, z.imag);
        return os;
    }
}

namespace noa::details {
    template<> struct Stringify<Complex<f16>> { static auto get() -> std::string { return "c16"; }};
    template<> struct Stringify<Complex<f32>> { static auto get() -> std::string { return "c32"; }};
    template<> struct Stringify<Complex<f64>> { static auto get() -> std::string { return "c64"; }};
}
