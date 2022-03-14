/// \file noa/common/types/Complex.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// \brief A complex number that can be used on the device.

#pragma once

#include <complex>
#include <cfloat>
#include <type_traits>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool2.h"
#include "noa/common/types/Half.h"

namespace noa {
    template<typename>
    class Float2;

    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Complex {
    public:
        static_assert(noa::traits::is_float_v<T>);
        T real{}, imag{};

    public:
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t COUNT = 2;

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<I>(i) < COUNT);
            if (i == I(1))
                return this->imag;
            else
                return this->real;
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(i < COUNT);
            if (i == I(1))
                return this->imag;
            else
                return this->real;
        }

    public: // Default constructors
        constexpr Complex(const Complex&) noexcept = default;
        constexpr Complex(Complex&&) noexcept = default;

        NOA_HD constexpr /* implicit */ Complex(T re = T(), T im = T()) noexcept
                : real(re), imag(im) {}

    public: // Conversion constructors
        template<typename U>
        NOA_HD constexpr explicit Complex(U v) noexcept
                : real(static_cast<T>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(U* ptr) noexcept
                : real(static_cast<T>(ptr[0])), imag(static_cast<T>(ptr[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(Complex<U> v) noexcept
                : real(static_cast<T>(v.real)), imag(static_cast<T>(v.imag)) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(Float2<U> v) noexcept
                : real(static_cast<T>(v[0])), imag(static_cast<T>(v[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(Int2<U> v) noexcept
                : real(static_cast<T>(v[0])), imag(static_cast<T>(v[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Complex(std::complex<U> v) noexcept
                : real(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[0])),
                  imag(static_cast<T>(reinterpret_cast<const U(&)[2]>(v)[1])) {}

    public: // Assignment operators
        constexpr Complex& operator=(const Complex& v) noexcept = default;
        constexpr Complex& operator=(Complex&& v) noexcept = default;

        NOA_HD constexpr Complex& operator=(T v) noexcept {
            this->real = v;
            this->imag = T(0);
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

        NOA_HD constexpr Complex& operator+=(T rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator-=(T rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator*=(T rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }
        NOA_HD constexpr Complex& operator/=(T rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Complex operator+(Complex v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Complex operator-(Complex v) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&v);
                *tmp = -(*tmp);
                return v;
            }
            #endif
            return {-v.real, -v.imag};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Complex operator+(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp += *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real + rhs.real, lhs.imag + rhs.imag};
        }

        friend NOA_HD constexpr Complex operator+(T lhs, Complex rhs) noexcept {
            return {lhs + rhs.real, rhs.imag};
        }

        friend NOA_HD constexpr Complex operator+(Complex lhs, T rhs) noexcept {
            return {lhs.real + rhs, lhs.imag};
        }

        friend NOA_HD constexpr Complex operator-(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp -= *reinterpret_cast<__half2*>(&rhs);
                return lhs;
            }
            #endif
            return {lhs.real - rhs.real, lhs.imag - rhs.imag};
        }

        friend NOA_HD constexpr Complex operator-(T lhs, Complex rhs) noexcept {
            return {lhs - rhs.real, -rhs.imag};
        }

        friend NOA_HD constexpr Complex operator-(Complex lhs, T rhs) noexcept {
            return {lhs.real - rhs, lhs.imag};
        }

        friend NOA_HD constexpr Complex operator*(Complex lhs, Complex rhs) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return Complex{Complex<HALF_ARITHMETIC_TYPE>(lhs) * Complex<HALF_ARITHMETIC_TYPE>(rhs)};
            return {lhs.real * rhs.real - lhs.imag * rhs.imag,
                    lhs.real * rhs.imag + lhs.imag * rhs.real};
        }

        friend NOA_HD constexpr Complex operator*(T lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&rhs);
                *tmp *= __half2half2(lhs.native());
                return rhs;
            }
            #endif
            return {lhs * rhs.real, lhs * rhs.imag};
        }

        friend NOA_HD constexpr Complex operator*(Complex lhs, T rhs) noexcept {
            return rhs * lhs;
        }

        // Adapted from cuComplex.h
        // "This implementation guards against intermediate underflow and overflow
        // by scaling. Such guarded implementations are usually the default for
        // complex library implementations, with some also offering an unguarded,
        // faster version."
        friend NOA_HD constexpr Complex operator/(Complex lhs, Complex rhs) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return Complex{Complex<HALF_ARITHMETIC_TYPE>(lhs) / Complex<HALF_ARITHMETIC_TYPE>(rhs)};

            T s = math::abs(rhs.real) + math::abs(rhs.imag);
            T oos = T(1.0) / s;

            T ars = lhs.real * oos;
            T ais = lhs.imag * oos;
            T brs = rhs.real * oos;
            T bis = rhs.imag * oos;

            s = (brs * brs) + (bis * bis);
            oos = T(1.0) / s;

            return {((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos};
        }

        friend NOA_HD constexpr Complex operator/(T lhs, Complex rhs) noexcept {
            return Complex(lhs) / rhs;
        }

        friend NOA_HD constexpr Complex operator/(Complex lhs, T rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>) {
                auto* tmp = reinterpret_cast<__half2*>(&lhs);
                *tmp /= __half2half2(rhs.native());
                return lhs;
            }
            #endif
            return {lhs.real / rhs, lhs.imag / rhs};
        }

        // -- Equality Operators -- //
        friend NOA_HD constexpr bool operator==(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return *reinterpret_cast<__half2*>(&lhs) == *reinterpret_cast<__half2*>(&rhs);
            #endif
            return lhs.real == rhs.real && lhs.imag == rhs.imag;
        }

        friend NOA_HD constexpr bool operator==(T lhs, Complex rhs) noexcept {
            return Complex(lhs) == rhs;
        }

        friend NOA_HD constexpr bool operator==(Complex lhs, T rhs) noexcept {
            return lhs == Complex(rhs);
        }

        friend NOA_HD constexpr bool operator!=(Complex lhs, Complex rhs) noexcept {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            if constexpr (std::is_same_v<T, half_t>)
                return *reinterpret_cast<__half2*>(&lhs) != *reinterpret_cast<__half2*>(&rhs);
            #endif
            return !(lhs == rhs);
        }

        friend NOA_HD constexpr bool operator!=(T lhs, Complex rhs) noexcept {
            return Complex(lhs) != rhs;
        }

        friend NOA_HD constexpr bool operator!=(Complex lhs, T rhs) noexcept {
            return lhs != Complex(rhs);
        }

        friend NOA_HD constexpr bool operator==(Complex lhs, std::complex<T> rhs) noexcept {
            return lhs == Complex(rhs);
        }

        friend NOA_HD constexpr bool operator==(std::complex<T> lhs, Complex rhs) noexcept {
            return Complex(lhs) == rhs;
        }

        friend NOA_HD constexpr bool operator!=(Complex lhs, std::complex<T> rhs) noexcept {
            return !(lhs == rhs);
        }

        friend NOA_HD constexpr bool operator!=(std::complex<T> lhs, Complex rhs) noexcept {
            return !(lhs == rhs);
        }
    };

    namespace math {
        /// Returns the real part of the complex number \a x.
        template<typename T>
        NOA_FHD constexpr T real(Complex<T> x) noexcept { return x.real; }

        template<typename T>
        NOA_FHD constexpr T real(std::complex<T> x) noexcept { return x.real(); }

        /// Returns the imaginary part of the complex number \a x.
        template<typename T>
        NOA_FHD constexpr T imag(Complex<T> x) noexcept { return x.imag; }

        template<typename T>
        NOA_FHD constexpr T imag(std::complex<T> x) noexcept { return x.imag(); }

        /// Returns the phase angle (in radians) of the complex number \a z.
        template<typename T>
        NOA_FHD T arg(Complex<T> x) {
            return atan2(x.imag, x.real);
        }

        /// Returns the magnitude of the complex number \a x.
        template<typename T>
        NOA_FHD T abs(Complex<T> x) {
            return hypot(x.real, x.imag);
        }

        template<typename T>
        NOA_FHD T length(Complex<T> x) { return abs(x); }

        /// Returns the length-normalized of the complex number \a x to 1, reducing it to its phase.
        template<typename T>
        NOA_FHD Complex<T> normalize(Complex<T> x) {
            if constexpr (std::is_same_v<T, half_t>)
                return Complex<T>(normalize(Complex<HALF_ARITHMETIC_TYPE>(x)));
            T magnitude = abs(x);
            if (magnitude > T{0}) // hum ...
                magnitude = T{1} / magnitude;
            return x * magnitude;
        }

        /// Returns the squared magnitude of the complex number \a x.
        template<typename T>
        NOA_IHD T norm(Complex<T> x);

        template<>
        NOA_IHD float norm<float>(Complex<float> x) {
            if (abs(x.real) < sqrt(FLT_MIN) && abs(x.imag) < sqrt(FLT_MIN)) {
                float a = x.real * 4.0f;
                float b = x.imag * 4.0f;
                return (a * a + b * b) / 16.0f;
            }
            return x.real * x.real + x.imag * x.imag;
        }

        template<>
        NOA_IHD double norm<double>(Complex<double> x) {
            if (abs(x.real) < sqrt(DBL_MIN) && abs(x.imag) < sqrt(DBL_MIN)) {
                double a = x.real * 4.0;
                double b = x.imag * 4.0;
                return (a * a + b * b) / 16.0;
            }
            return x.real * x.real + x.imag * x.imag;
        }

        template<>
        NOA_IHD half_t norm<half_t>(Complex<half_t> x) {
            return half_t(norm(Complex<HALF_ARITHMETIC_TYPE>(x)));
        }

        template<typename T>
        NOA_FHD T lengthSq(Complex<T> x) { return norm(x); }

        /// Returns the complex conjugate of \a x.
        template<typename T>
        NOA_FHD constexpr Complex<T> conj(Complex<T> x) noexcept {
            return {x.real, -x.imag};
        }

        /// Returns a complex number with magnitude \a length (should be positive) and phase angle \a theta.
        template<typename T>
        NOA_FHD Complex<T> polar(T length, T theta) {
            return {length * cos(theta), length * sin(theta)};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(Complex<T> a, Complex<T> b, T e = NOA_EPSILON_) {
            return isEqual<ULP>(a.real, b.real, e) && isEqual<ULP>(a.imag, b.imag, e);
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(Complex<T> a, T b, T e = NOA_EPSILON_) {
            return isEqual<ULP>(a.real, b, e) && isEqual<ULP>(a.imag, b, e);
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, Complex<T> b, T e = NOA_EPSILON_) {
            return isEqual<ULP>(a, b.real, e) && isEqual<ULP>(a, b.imag, e);
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<>
        struct proclaim_is_complex<Complex<half_t>> : std::true_type {};
        template<>
        struct proclaim_is_complex<Complex<float>> : std::true_type {};
        template<>
        struct proclaim_is_complex<Complex<double>> : std::true_type {};
    }

    using chalf_t = Complex<half_t>;
    using cfloat_t = Complex<float>;
    using cdouble_t = Complex<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(Complex<T> v) noexcept {
        return {v.real, v.imag};
    }

    template<>
    NOA_IH std::string string::typeName<cdouble_t>() { return "cdouble"; }
    template<>
    NOA_IH std::string string::typeName<cfloat_t>() { return "cfloat"; }
    template<>
    NOA_IH std::string string::typeName<chalf_t>() { return "chalf"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Complex<T> z) {
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
