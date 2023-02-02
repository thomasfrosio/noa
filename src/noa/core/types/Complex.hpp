#pragma once

#include <complex>
#include <cfloat>
#include <type_traits>

#include "noa/core/Assert.hpp"
#include "noa/core/Definitions.hpp"
#include "noa/core/Math.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Half.hpp"

namespace noa {
    template<typename, size_t>
    class Vec;

    template<typename Real>
    class alignas(sizeof(Real) * 2) Complex {
    public:
        static_assert(traits::is_real_v<Real>);
        Real real{}, imag{};

    public:
        using value_type = Real;

    public: // Component accesses
        static constexpr size_t COUNT = 2;
        static constexpr size_t SIZE = 2;
        static constexpr int64_t SSIZE = 2;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr value_type& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            if (i == I(1))
                return this->imag;
            else
                return this->real;
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        NOA_HD constexpr const value_type& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            if (i == I(1))
                return this->imag;
            else
                return this->real;
        }

    public: // Default constructors
        NOA_HD constexpr /* implicit */ Complex(value_type re = value_type(),
                                                value_type im = value_type()) noexcept
                : real(re), imag(im) {}

    public: // Conversion constructors
        template<typename U>
        NOA_HD constexpr explicit Complex(U v) noexcept
                : real(static_cast<value_type>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(U* ptr) noexcept
                : real(static_cast<value_type>(ptr[0])),
                  imag(static_cast<value_type>(ptr[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(Complex<U> v) noexcept
                : real(static_cast<value_type>(v.real)),
                  imag(static_cast<value_type>(v.imag)) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(Vec<U, 2> v) noexcept
                : real(static_cast<value_type>(v[0])),
                  imag(static_cast<value_type>(v[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(std::complex<U> v) noexcept
                : real(static_cast<value_type>(reinterpret_cast<const U(&)[2]>(v)[0])),
                  imag(static_cast<value_type>(reinterpret_cast<const U(&)[2]>(v)[1])) {}

    public: // Assignment operators
        NOA_HD constexpr Complex& operator=(value_type v) noexcept {
            this->real = v;
            this->imag = value_type{0};
            return *this;
        }

        NOA_HD constexpr Complex& operator+=(Complex rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator-=(Complex rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator*=(Complex rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator/=(Complex rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr Complex& operator+=(value_type rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator-=(value_type rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator*=(value_type rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator/=(value_type rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Complex operator+(Complex v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator-(Complex v) noexcept {
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
        [[nodiscard]] friend NOA_HD constexpr Complex operator+(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp += *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real + rhs.real, lhs.imag + rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator+(value_type lhs, Complex rhs) noexcept {
            return {lhs + rhs.real, rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator+(Complex lhs, value_type rhs) noexcept {
            return {lhs.real + rhs, lhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator-(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real - rhs.real, lhs.imag - rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator-(value_type lhs, Complex rhs) noexcept {
            return {lhs - rhs.real, -rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator-(Complex lhs, value_type rhs) noexcept {
            return {lhs.real - rhs, lhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator*(Complex lhs, Complex rhs) noexcept {
            if constexpr (std::is_same_v<value_type, Half>)
                return Complex{Complex<Half::arithmetic_type>(lhs) * Complex<Half::arithmetic_type>(rhs)};
            return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                    lhs.real * rhs.imag + lhs.imag * rhs.real};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator*(value_type lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp *= __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs * rhs.real, lhs * rhs.imag};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator*(Complex lhs, value_type rhs) noexcept {
            return rhs * lhs;
        }

        // Adapted from cuComplex.h
        // "This implementation guards against intermediate underflow and overflow
        // by scaling. Such guarded implementations are usually the default for
        // complex library implementations, with some also offering an unguarded,
        // faster version."
        [[nodiscard]] friend NOA_HD constexpr Complex operator/(Complex lhs, Complex rhs) noexcept {
            if constexpr (std::is_same_v<value_type, Half>)
                return Complex{Complex<Half::arithmetic_type>(lhs) / Complex<Half::arithmetic_type>(rhs)};

            auto s = math::abs(rhs.real) + math::abs(rhs.imag);
            auto oos = value_type(1.0) / s;

            auto ars = lhs.real * oos;
            auto ais = lhs.imag * oos;
            auto brs = rhs.real * oos;
            auto bis = rhs.imag * oos;

            s = (brs * brs) + (bis * bis);
            oos = value_type(1.0) / s;

            return {((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos};
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator/(value_type lhs, Complex rhs) noexcept {
            return Complex(lhs) / rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr Complex operator/(Complex lhs, value_type rhs) noexcept {
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
        [[nodiscard]] friend NOA_HD constexpr bool operator==(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>)
                return *reinterpret_cast<__half2*>(&lhs) == *reinterpret_cast<__half2*>(&rhs);
            #endif
            return lhs.real == rhs.real && lhs.imag == rhs.imag;
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(value_type lhs, Complex rhs) noexcept {
            return Complex(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(Complex lhs, value_type rhs) noexcept {
            return lhs == Complex(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<value_type, Half>)
                return *reinterpret_cast<__half2*>(&lhs) != *reinterpret_cast<__half2*>(&rhs);
            #endif
            return !(lhs == rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(value_type lhs, Complex rhs) noexcept {
            return Complex(lhs) != rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Complex lhs, value_type rhs) noexcept {
            return lhs != Complex(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(Complex lhs, std::complex<value_type> rhs) noexcept {
            return lhs == Complex(rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator==(std::complex<value_type> lhs, Complex rhs) noexcept {
            return Complex(lhs) == rhs;
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(Complex lhs, std::complex<value_type> rhs) noexcept {
            return !(lhs == rhs);
        }

        [[nodiscard]] friend NOA_HD constexpr bool operator!=(std::complex<value_type> lhs, Complex rhs) noexcept {
            return !(lhs == rhs);
        }

    public:
        [[nodiscard]] NOA_HD constexpr auto to_vec() const noexcept {
            return Vec<value_type, 2>(real, imag);
        }

    public: // Support for noa::string::human<Vec>();
        [[nodiscard]] static std::string name() {
            if constexpr (std::is_same_v<value_type, Half>)
                return "c16";
            else if constexpr (std::is_same_v<value_type, float>)
                return "c32";
            else
                return "c64";
        }
    };
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T real(Complex<T> x) noexcept { return x.real; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T real(std::complex<T> x) noexcept { return x.real(); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T imag(Complex<T> x) noexcept { return x.imag; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T imag(std::complex<T> x) noexcept { return x.imag(); }

    // Returns the phase angle (in radians) of the complex number z.
    template<typename T>
    [[nodiscard]] NOA_FHD T arg(Complex<T> x) {
        return atan2(x.imag, x.real);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD T phase(Complex<T> x) {
        return arg(x);
    }

    // Returns the magnitude of the complex number x.
    template<typename T>
    [[nodiscard]] NOA_FHD T abs(Complex<T> x) {
        return hypot(x.real, x.imag);
    }

    // Returns the length-normalized of the complex number x to 1, reducing it to its phase.
    template<typename T>
    [[nodiscard]] NOA_FHD Complex<T> normalize(Complex<T> x) {
        if constexpr (std::is_same_v<T, Half>)
            return Complex<T>(normalize(Complex<Half::arithmetic_type>(x)));
        T magnitude = abs(x);
        if (magnitude > T{0})
            magnitude = T{1} / magnitude;
        return x * magnitude;
    }

    // Returns the squared magnitude of the complex number x.
    template<typename T>
    NOA_IHD T abs_squared(Complex<T> x);

    template<>
    [[nodiscard]] NOA_IHD float abs_squared<float>(Complex<float> x) {
        constexpr float THRESHOLD = 1.0842021724855044e-19f; // sqrt(FLT_MIN);
        if (abs(x.real) < THRESHOLD && abs(x.imag) < THRESHOLD) {
            const float a = x.real * 4.0f;
            const float b = x.imag * 4.0f;
            return (a * a + b * b) / 16.0f;
        }
        return x.real * x.real + x.imag * x.imag;
    }

    template<>
    [[nodiscard]] NOA_IHD double abs_squared<double>(Complex<double> x) {
        constexpr double THRESHOLD = 1.4916681462400413e-154; // sqrt(DBL_MIN)
        if (abs(x.real) < THRESHOLD && abs(x.imag) < THRESHOLD) {
            const double a = x.real * 4.0;
            const double b = x.imag * 4.0;
            return (a * a + b * b) / 16.0;
        }
        return x.real * x.real + x.imag * x.imag;
    }

    template<>
    [[nodiscard]] NOA_IHD Half abs_squared<Half>(Complex<Half> x) {
        return Half(abs_squared(Complex<Half::arithmetic_type>(x)));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Complex<T> conj(Complex<T> x) noexcept {
        return {x.real, -x.imag};
    }

    // Returns a complex number with magnitude length (should be positive) and phase angle theta.
    template<typename T>
    [[nodiscard]] NOA_FHD Complex<T> polar(T length, T theta) {
        return {length * cos(theta), length * sin(theta)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T dot(Complex<T> a, Complex<T> b) noexcept {
        if constexpr (std::is_same_v<T, Half>)
            return fma(a[0], b[0], a[1] * b[1]);
        return a[0] * b[0] + a[1] * b[1];
    }
}

namespace noa::traits {
    template<> struct proclaim_is_complex<Complex<Half>> : std::true_type {};
    template<> struct proclaim_is_complex<Complex<float>> : std::true_type {};
    template<> struct proclaim_is_complex<Complex<double>> : std::true_type {};
}

namespace noa {
    template<typename T>
    std::ostream& operator<<(std::ostream& os, Complex<T> z) {
        os << string::format("({:.3f},{:.3f})", z.real, z.imag);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Complex<T>> : formatter<T> {
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
