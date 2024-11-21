#pragma once

#include <complex>
#include <cfloat>

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/types/Half.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/core/indexing/Bounds.hpp"

namespace noa::inline types {
    template<typename, size_t, size_t>
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
        static constexpr size_t SIZE = 2;
        static constexpr int64_t SSIZE = 2;

        template<std::integral I>
        NOA_HD constexpr auto operator[](I i) noexcept -> value_type& {
            ni::bounds_check(SSIZE, i);
            if (i == I{1})
                return this->imag;
            else
                return this->real;
        }

        template<std::integral I>
        NOA_HD constexpr auto operator[](I i) const noexcept -> const value_type& {
            ni::bounds_check(SSIZE, i);
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

        template<typename U, size_t A>
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
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator+(Complex v) noexcept -> Complex {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(Complex v) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                *tmp = -(*tmp);
                return v;
            }
            #endif
            return {-v.real, -v.imag};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr auto operator+(Complex lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp += *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real + rhs.real, lhs.imag + rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator+(value_type lhs, Complex rhs) noexcept -> Complex {
            return {lhs + rhs.real, rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator+(Complex lhs, value_type rhs) noexcept -> Complex {
            return {lhs.real + rhs, lhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(Complex lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real - rhs.real, lhs.imag - rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(value_type lhs, Complex rhs) noexcept -> Complex {
            return {lhs - rhs.real, -rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator-(Complex lhs, value_type rhs) noexcept -> Complex {
            return {lhs.real - rhs, lhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(Complex lhs, Complex rhs) noexcept -> Complex {
            if constexpr (std::is_same_v<value_type, Half>) {
                return (lhs.as<Half::arithmetic_type>() * rhs.as<Half::arithmetic_type>()).template as<value_type>();
            }
            return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                    lhs.real * rhs.imag + lhs.imag * rhs.real};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(value_type lhs, Complex rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp *= __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs * rhs.real, lhs * rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator*(Complex lhs, value_type rhs) noexcept -> Complex {
            return rhs * lhs;
        }

        // Adapted from cuComplex.h
        // "This implementation guards against intermediate underflow and overflow
        // by scaling. Such guarded implementations are usually the default for
        // complex library implementations, with some also offering an unguarded,
        // faster version."
        [[nodiscard]] friend NOA_HD constexpr auto operator/(Complex lhs, Complex rhs) noexcept -> Complex {
            if constexpr (std::is_same_v<value_type, Half>) {
                return (lhs.as<Half::arithmetic_type>() / rhs.as<Half::arithmetic_type>()).template as<value_type>();
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

        [[nodiscard]] friend NOA_HD constexpr auto operator/(value_type lhs, Complex rhs) noexcept -> Complex {
            return Complex::from_real(lhs) / rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator/(Complex lhs, value_type rhs) noexcept -> Complex {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp /= __half2half2(rhs.native());
                return lhs;
            }
            #endif
            return {lhs.real / rhs, lhs.imag / rhs};
        }

        // -- Equality Operators -- //
        [[nodiscard]] friend NOA_HD constexpr auto operator==(Complex lhs, Complex rhs) noexcept -> bool {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>)
                return *reinterpret_cast<__half2*>(&lhs) == *reinterpret_cast<__half2*>(&rhs);
            #endif
            return lhs.real == rhs.real && lhs.imag == rhs.imag;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(value_type lhs, Complex rhs) noexcept -> bool {
            return Complex::from_real(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Complex lhs, value_type rhs) noexcept -> bool {
            return lhs == Complex::from_real(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Complex lhs, Complex rhs) noexcept -> bool {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>)
                return *reinterpret_cast<__half2*>(&lhs) != *reinterpret_cast<__half2*>(&rhs);
            #endif
            return !(lhs == rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(value_type lhs, Complex rhs) noexcept -> bool {
            return Complex::from_real(lhs) != rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Complex lhs, value_type rhs) noexcept -> bool {
            return lhs != Complex::from_real(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(Complex lhs, std::complex<value_type> rhs) noexcept -> bool {
            return lhs == Complex::from_complex(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator==(std::complex<value_type> lhs, Complex rhs) noexcept -> bool {
            return Complex::from_complex(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(Complex lhs, std::complex<value_type> rhs) noexcept -> bool {
            return !(lhs == rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr auto operator!=(std::complex<value_type> lhs, Complex rhs) noexcept -> bool {
            return !(lhs == rhs);
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
        if constexpr (std::is_same_v<T, Half>)
            return Complex<T>(normalize(x.template as<Half::arithmetic_type>()));
        T magnitude = abs(x);
        if (magnitude > T{0})
            magnitude = T{1} / magnitude;
        return x * magnitude;
    }

    /// Returns the squared magnitude of the complex number x.
    template<typename T>
    NOA_IHD auto abs_squared(Complex<T> x) noexcept -> T {
        if constexpr (std::is_same_v<T, Half>)
            return Half(abs_squared(x.template as<Half::arithmetic_type>()));
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
        if constexpr (std::is_same_v<T, Half>)
            return fma(a[0], b[0], a[1] * b[1]);
        return a[0] * b[0] + a[1] * b[1];
    }
}

namespace noa::traits {
    template<> struct proclaim_is_complex<c16> : std::true_type {};
    template<> struct proclaim_is_complex<c32> : std::true_type {};
    template<> struct proclaim_is_complex<c64> : std::true_type {};
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

namespace noa::string {
    template<typename T>
    struct Stringify<Complex<T>> {
        static auto get() -> std::string {
            if constexpr (std::is_same_v<T, Half>)
                return "c16";
            else if constexpr (std::is_same_v<T, float>)
                return "c32";
            else
                return "c64";
        }
    };
}
