/// \file noa/common/types/Complex.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// \brief A complex number that can be used on the device.

#pragma once

#include <complex>
#include <cfloat>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool2.h"

namespace noa {
    template<typename>
    class Float2;

    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Complex {
    public:
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        T real{}, imag{};

    public:
        typedef T value_type;

    public: // Component accesses
        static constexpr size_t COUNT = 2;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Complex() noexcept = default;
        template<typename U> NOA_HD constexpr Complex(U v) noexcept; // not explicit on purpose, i.e. mimic float
        template<typename U> NOA_HD constexpr explicit Complex(U* ptr);
        template<typename R, typename I> NOA_HD constexpr Complex(R re, I im) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Complex<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Complex(const std::complex<U>& v) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Complex<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(U* ptr);
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Complex<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator=(const std::complex<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Complex<T>& operator+=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator-=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator*=(const Complex<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator/=(const Complex<U>& rhs);

        template<typename U> NOA_HD constexpr Complex<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Complex<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator+(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator+(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator-(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Complex<T> operator*(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator*(T lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Complex<T> operator*(const Complex<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Complex<T> operator/(const Complex<T>& lhs, const Complex<T>& rhs);
    template<typename T> NOA_HD constexpr Complex<T> operator/(T lhs, const Complex<T>& rhs);
    template<typename T> NOA_HD constexpr Complex<T> operator/(const Complex<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(T lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, const Complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(T lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator==(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator==(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr bool operator!=(const Complex<T>& lhs, const std::complex<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr bool operator!=(const std::complex<T>& lhs, const Complex<T>& rhs) noexcept;

    namespace math {
        /// Returns the real part of the complex number \a x.
        template<typename T> NOA_FHD constexpr T real(Complex<T> x) noexcept { return x.real; }
        template<typename T> NOA_FHD constexpr T real(std::complex<T> x) noexcept { return x.real(); }

        /// Returns the imaginary part of the complex number \a x.
        template<typename T> NOA_FHD constexpr T imag(Complex<T> x) noexcept { return x.imag; }
        template<typename T> NOA_FHD constexpr T imag(std::complex<T> x) noexcept { return x.imag(); }

        /// Returns the phase angle (in radians) of the complex number \a z.
        template<typename T> NOA_FHD T arg(const Complex<T>& x);

        /// Returns the magnitude of the complex number \a x.
        template<typename T> NOA_FHD T abs(const Complex<T>& x);
        template<typename T> NOA_FHD T length(const Complex<T>& x) { return abs(x); }

        /// Returns the length-normalized of the complex number \a x to 1, reducing it to its phase.
        template<typename T> NOA_FHD Complex<T> normalize(const Complex<T>& x);

        /// Returns the squared magnitude of the complex number \a x.
        template<typename T> NOA_IHD T norm(const Complex<T>& x);
        template<typename T> NOA_IHD T lengthSq(const Complex<T>& x) { return norm(x); }

        /// Returns the complex conjugate of \a x.
        template<typename T> NOA_FHD constexpr Complex<T> conj(const Complex<T>& x) noexcept;

        /// Returns a complex number with magnitude \a length (should be positive) and phase angle \a theta.
        template<typename T> NOA_FHD Complex<T> polar(T length, T theta);

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Complex<T>& a, const Complex<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Complex<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, const Complex<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<> struct proclaim_is_complex<Complex<float>> : std::true_type {};
        template<> struct proclaim_is_complex<Complex<double>> : std::true_type {};
    }

    using cfloat_t = Complex<float>;
    using cdouble_t = Complex<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(const Complex<T>& v) noexcept {
        return {v.real, v.imag};
    }

    template<> NOA_IH std::string string::typeName<cdouble_t>() { return "cdouble"; }
    template<> NOA_IH std::string string::typeName<cfloat_t>() { return "cfloat"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Complex<T>& z) {
        os << string::format("({:.3f},{:.3f})", z.real, z.imag);
        return os;
    }
}

#define NOA_INCLUDE_COMPLEX_
#include "noa/common/types/details/Complex.inl"
#undef NOA_INCLUDE_COMPLEX_
