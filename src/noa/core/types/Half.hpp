#pragma once

#include <limits>
#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/utils/Strings.hpp"

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wbool-compare"
#elif defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

// Override half-precision host implementation to use this type for arithmetic operations
// and math functions, since this can result in improved performance. For CUDA device
// code, math functions not supported in half-precision are cast to this type.
#ifndef HALF_ARITHMETIC_TYPE
#define HALF_ARITHMETIC_TYPE float
#endif

// For device code, use __half. For host code, use half_float::float.
// Their underlying type is u16, and they can be used interchangeably.
#ifdef __CUDA_ARCH__
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#else
#include <half/half.hpp>
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif

// TODO C++23 replace with std::float16_t and std::bfloat16_t

namespace noa::inline types {
    /// 16-bit precision float (IEEE-754-2008).
    /// \details This structure implements the datatype for storing half-precision floating-point numbers. Compared to
    ///          bfloat16, this type treads range for precision. There are 15361 representable numbers within the
    ///          interval [0.0, 1.0]. Its range is [6.10e-5, -/+6.55e4], so be careful with overflows.
    ///          Also note that, after passed -/+ 2048, not all integral values are representable in this format.
    ///          For instance, int(half(int(2049)) == 2048. This behavior is also true for single and double precision
    ///          floats, but for much larger values (see https://stackoverflow.com/a/3793950).
    ///          This structure can be used on host code (using the "half" library from Christian Rau) and CUDA device
    ///          code (using the __half precision intrinsics from CUDA). Both implementations are using an unsigned
    ///          short as underlying type.
    /// \note For device code, arithmetic operators and some math functions are only supported for devices with
    ///       compute capability >= 5.3 or 8.0. For devices with compute capability lower than that, higher precision
    ///       overloads are used internally (see HALF_ARITHMETIC_TYPE).
    class f16 {
    public: // --- typedefs --- //
        using arithmetic_type = HALF_ARITHMETIC_TYPE;

        #ifdef __CUDA_ARCH__
        using native_type = __half;
        #else
        using native_type = half_float::half;
        #endif

        // This can be dangerous to have different types between GPU and CPU, so check here that
        // everything is defined as expected and that we can pass this type to the GPU and vice-versa.
        static_assert(std::is_standard_layout_v<native_type>);
        static_assert(std::is_nothrow_move_assignable_v<native_type>);
        static_assert(std::is_nothrow_move_constructible_v<native_type>);
        static_assert(sizeof(native_type) == 2);
        static_assert(alignof(native_type) == alignof(u16));

    public:
        /// Default constructor.
        /// FIXME half_float zero-initializes, CUDA's __half doesn't.
        constexpr f16() noexcept = default;

        /// Conversion constructor.
        /// \details Any explicit initialization from a builtin type goes through that constructor. No conversion
        ///          warnings will be raised. Supported types are: native_type, f16, float, double, (u)char, (u)short,
        ///          (u)int, (u)long, (u)long long.
        /// \note This is equivalent to static_cast, so the value can overflow during the conversion.
        ///       Use clamp_cast or safe_cast to have a well-defined overflow behavior.
        /// FIXME Ideally, we would make it implicit, but the issue is that the compiler will not generate a loss
        ///       of precision warning, so f16 a = 100000 would never warn. The issue with explicit however, is that
        ///       something like c16{1, 0} is not allowed and we have to write c16{f16{1}, f16{0}}...
        template<typename T>
        NOA_HD constexpr explicit f16(T x) : m_data(from_value<native_type>(x)) {}

    public:
        [[nodiscard]] static NOA_HD constexpr f16 from_bits(u16 bits) noexcept { return f16(Empty{}, bits); }

        // Cast to/from native_type. Most builtin types are supported.
        // Clang < 15 required this function to be defined before the cast operator X().
        template<typename T, typename U>
        [[nodiscard]] static NOA_HD constexpr T from_value(U value) {
            #ifdef __CUDA_ARCH__
            if constexpr (std::is_same_v<T, U>) {
                return value;
            } else if constexpr (std::is_same_v<T, native_type>) { // built-in -> native_type
                if constexpr (std::is_same_v<U, float>) {
                    return __float2half_rn(value);
                } else if constexpr (std::is_same_v<U, double>) {
                    return __double2half(value);
                } else if constexpr (std::is_same_v<U, signed char> or
                                     std::is_same_v<U, char> or
                                     std::is_same_v<U, bool>) {
                    return __short2half_rn(static_cast<short>(value));
                } else if constexpr (std::is_same_v<U, unsigned char>) {
                    return __ushort2half_rn(static_cast<unsigned short>(value));
                } else if constexpr (std::is_same_v<U, short>) {
                    return __short2half_rn(value);
                } else if constexpr (std::is_same_v<U, ushort>) {
                    return __ushort2half_rn(value);
                } else if constexpr (std::is_same_v<U, int> or (std::is_same_v<U, long> && sizeof(long) == 4)) {
                    return __int2half_rn(value);
                } else if constexpr (std::is_same_v<U, uint> or (std::is_same_v<U, ulong> && sizeof(long) == 4)) {
                    return __uint2half_rn(value);
                } else if constexpr (std::is_same_v<U, long long> or std::is_same_v<U, long>) {
                    return __ll2half_rn(value);
                } else if constexpr (std::is_same_v<U, unsigned long long> or std::is_same_v<U, ulong>) {
                    return __ull2half_rn(value);
                } else {
                    static_assert(nt::always_false<>);
                }
            } else if constexpr (std::is_same_v<U, native_type>) { // native_type -> built-in
                if constexpr (std::is_same_v<T, float>) {
                    return __half2float(value);
                } else if constexpr (std::is_same_v<T, double>) {
                    return static_cast<double>(__half2float(value));
                } else if constexpr (std::is_same_v<T, bool>) {
                    return static_cast<bool>(__half2short_rn(value));
                } else if constexpr (std::is_same_v<T, signed char>) {
                    return static_cast<signed char>(__half2short_rn(value));
                } else if constexpr (std::is_same_v<T, char>) {
                    return static_cast<char>(__half2short_rn(value));
                } else if constexpr (std::is_same_v<T, unsigned char>) {
                    return static_cast<unsigned char>(__half2ushort_rn(value));
                } else if constexpr (std::is_same_v<T, short>) {
                    return __half2short_rn(value);
                } else if constexpr (std::is_same_v<T, ushort>) {
                    return __half2ushort_rn(value);
                } else if constexpr (std::is_same_v<T, int> or (std::is_same_v<T, long> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2int_rn(value));
                } else if constexpr (std::is_same_v<T, uint> or (std::is_same_v<T, ulong> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2uint_rn(value));
                } else if constexpr (std::is_same_v<T, long long> or std::is_same_v<T, long>) {
                    return static_cast<T>(__half2ll_rn(value));
                } else if constexpr (std::is_same_v<T, unsigned long long> or std::is_same_v<T, ulong>) {
                    return static_cast<T>(__half2ull_rn(value));
                } else {
                    static_assert(nt::always_false<>);
                }
            } else {
                static_assert(nt::always_false<>);
            }
            #else
            if constexpr (std::is_same_v<T, U>) {
                return value;
            } else if constexpr (std::is_same_v<T, native_type> or std::is_same_v<U, native_type>) {
                // half_float::half_cast has a bug in int2half for the min value so check it beforehand.
                if constexpr (std::is_integral_v<U> and std::is_signed_v<U>) {
                    if (value == std::numeric_limits<U>::min()) {
                        if constexpr (sizeof(U) == 1)
                            return half_float::reinterpret_as_half(0xD800); // -128
                        else if constexpr(sizeof(U) == 2)
                            return half_float::reinterpret_as_half(0xF800); // -32768
                        else
                            return half_float::reinterpret_as_half(0xFC00); // -inf
                    }
                }
                return half_float::half_cast<T>(value);
            } else {
                static_assert(nt::always_false<T>);
            }
            #endif
        }

        // On the host, it is half_float::half. On CUDA devices, it is __half.
        [[nodiscard]] NOA_HD constexpr auto native() const noexcept -> const native_type& { return m_data; }
        [[nodiscard]] NOA_HD constexpr auto native() noexcept -> native_type& { return m_data; }

    public: // --- Conversion to built-in types --- //
        NOA_HD explicit constexpr operator float() const {
            return from_value<float>(m_data);
        }
        NOA_HD explicit constexpr operator double() const {
            return from_value<double>(m_data);
        }
        NOA_HD explicit constexpr operator bool() const {
            return from_value<bool>(m_data);
        }
        NOA_HD explicit constexpr operator char() const {
            return from_value<char>(m_data);
        }
        NOA_HD explicit constexpr operator signed char() const {
            return from_value<signed char>(m_data);
        }
        NOA_HD explicit constexpr operator unsigned char() const {
            return from_value<unsigned char>(m_data);
        }
        NOA_HD explicit constexpr operator short() const {
            return from_value<short>(m_data);
        }
        NOA_HD explicit constexpr operator unsigned short() const {
            return from_value<unsigned short>(m_data);
        }
        NOA_HD explicit constexpr operator int() const {
            return from_value<int>(m_data);
        }
        NOA_HD explicit constexpr operator unsigned int() const {
            return from_value<unsigned int>(m_data);
        }
        NOA_HD explicit constexpr operator long() const {
            return from_value<long>(m_data);
        }
        NOA_HD explicit constexpr operator unsigned long() const {
            return from_value<unsigned long>(m_data);
        }
        NOA_HD explicit constexpr operator long long() const {
            return from_value<long long>(m_data);
        }
        NOA_HD explicit constexpr operator unsigned long long() const {
            return from_value<unsigned long long>(m_data);
        }

    public: // --- Arithmetic operators --- //
        NOA_HD f16& operator+=(f16 rhs) {
            return *this = *this + rhs;
        }
        NOA_HD f16& operator-=(f16 rhs) {
            return *this = *this - rhs;
        }
        NOA_HD f16& operator*=(f16 rhs) {
            return *this = *this * rhs;
        }
        NOA_HD f16& operator/=(f16 rhs) {
            return *this = *this / rhs;
        }
        NOA_HD f16& operator+=(float rhs) {
            return *this = f16(static_cast<float>(*this) + rhs);
        }
        NOA_HD f16& operator-=(float rhs) {
            return *this = f16(static_cast<float>(*this) - rhs);
        }
        NOA_HD f16& operator*=(float rhs) {
            return *this = f16(static_cast<float>(*this) * rhs);
        }
        NOA_HD f16& operator/=(float rhs) {
            return *this = f16(static_cast<float>(*this) / rhs);
        }

        NOA_HD f16& operator++() {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = __half(static_cast<f16::arithmetic_type>(m_data) + 1);
            #else
            ++m_data;
            #endif
            return *this;
        }
        NOA_HD f16& operator--() {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = __half(static_cast<f16::arithmetic_type>(m_data) - 1);
            #else
            --m_data;
            #endif
            return *this;
        }

        NOA_HD f16 operator++(int) {
            f16 out(*this);
            ++(*this);
            return out;
        }
        NOA_HD f16 operator--(int) {
            f16 out(*this);
            --(*this);
            return out;
        }

        [[nodiscard]] NOA_HD friend f16 operator+(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return f16(static_cast<f16::arithmetic_type>(lhs) + static_cast<f16::arithmetic_type>(rhs));
            #else
            return f16(lhs.m_data + rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend f16 operator-(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return f16(static_cast<f16::arithmetic_type>(lhs) - static_cast<f16::arithmetic_type>(rhs));
            #else
            return f16(lhs.m_data - rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend f16 operator*(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return f16(static_cast<f16::arithmetic_type>(lhs) * static_cast<f16::arithmetic_type>(rhs));
            #else
            return f16(lhs.m_data * rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend f16 operator/(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return f16(static_cast<f16::arithmetic_type>(lhs) / static_cast<f16::arithmetic_type>(rhs));
            #else
            return f16(lhs.m_data / rhs.m_data);
            #endif
        }

        [[nodiscard]] NOA_HD friend f16 operator+(f16 lhs) {
            return lhs;
        }
        [[nodiscard]] NOA_HD friend f16 operator-(f16 lhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return f16(-static_cast<f16::arithmetic_type>(lhs));
            #else
            return f16(-lhs.m_data);
            #endif
        }

        [[nodiscard]] NOA_HD friend bool operator==(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) == static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data == rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator!=(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) != static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data != rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator>(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) > static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data > rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator<(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) < static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data < rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator>=(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) >= static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data >= rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator<=(f16 lhs, f16 rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<f16::arithmetic_type>(lhs) <= static_cast<f16::arithmetic_type>(rhs);
            #else
            return lhs.m_data <= rhs.m_data;
            #endif
        }

    private:
        // Private constructor reinterpreting the bits. Used by f16::from_bits(u16).
        #if defined(__CUDA_ARCH__)
        constexpr f16(Empty, u16 bits) noexcept : m_data(__half_raw{bits}) {}
        #else
        constexpr f16(Empty, u16 bits) noexcept : m_data(half_float::reinterpret_as_half(bits)) {}
        // reinterpret_as_half is not native to the half_float namespace. It was added to our version
        // to allow reinterpretation from bits to half_float::half in constexpr context.
        #endif

    private:
        native_type m_data;
    };

    static_assert(sizeof(f16) == 2);
    static_assert(alignof(f16) == 2);
}

namespace noa::traits {
    template<> struct proclaim_is_real<f16> : std::true_type {};
    template<> struct double_precision<f16> { using type = f64; };
}

namespace noa {
    [[nodiscard]] NOA_FHD f16 fma(f16 x, f16 y, f16 z) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(__hfma(x.native(), y.native(), z.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(fma(static_cast<f16::arithmetic_type>(x),
                        static_cast<f16::arithmetic_type>(y),
                        static_cast<f16::arithmetic_type>(z)));
        #else
        return f16(half_float::fma(x.native(), y.native(), z.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 cos(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hcos(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(cos(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::cos(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sin(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hsin(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(sin(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::sin(x.native()));
        #endif
    }

    NOA_FHD void sincos(f16 x, f16* s, f16* c) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        *s = sin(x);
        *c = cos(x);
        #else
        half_float::sincos(x.native(), &s->native(), &c->native());
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sinc(f16 x) {
        return f16(sinc(static_cast<f16::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD f16 cosh(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(cos(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::cosh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sinh(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(cos(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::sinh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 tan(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(tan(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::tan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 acos(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(acos(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::acos(x.native()));
        #endif
    }
    [[nodiscard]] NOA_FHD f16 acosh(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(acosh(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::acosh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 asin(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(asin(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::asin(x.native()));
        #endif
    }
    [[nodiscard]] NOA_FHD f16 asinh(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(asinh(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::asinh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atan(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(atan(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::atan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atan2(f16 y, f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(atan2(static_cast<f16::arithmetic_type>(y),
                          static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::atan2(y.native(), x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atanh(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hatanh(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(atanh(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::atan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rad2deg(f16 x) {
        return f16(rad2deg(static_cast<f16::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD f16 deg2rad(f16 x) {
        return f16(deg2rad(static_cast<f16::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD f16 pow(f16 x, f16 exp) {
        #if defined(__CUDA_ARCH__)
        return f16(pow(static_cast<f16::arithmetic_type>(x), static_cast<f16::arithmetic_type>(exp)));
        #else
        return f16(half_float::pow(x.native(), exp.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 exp(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hexp(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(exp(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::exp(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hlog(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(log(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::log(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log10(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hlog10(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(log10(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::log10(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log1p(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(log1p(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::log1p(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 hypot(f16 x, f16 y) {
        return f16(hypot(static_cast<f16::arithmetic_type>(x), static_cast<f16::arithmetic_type>(y)));
    }

    [[nodiscard]] NOA_FHD f16 sqrt(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hsqrt(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(sqrt(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::sqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rsqrt(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hrsqrt(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(rsqrt(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::rsqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 round(f16 x) {
        #if defined(__CUDA_ARCH__)
        return f16(round(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::round(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rint(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hrint(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(rint(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::rint(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 ceil(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hceil(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(ceil(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::ceil(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 floor(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(hfloor(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(floor(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::floor(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 trunc(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(htrunc(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(trunc(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::trunc(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 is_nan(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(__hisnan(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(isnan(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::isnan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 is_inf(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(__hisinf(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(isinf(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::isinf(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD bool is_finite(f16 x) {
        return !(is_inf(x) || is_nan(x));
    }

    [[nodiscard]] NOA_FHD f16 abs(f16 x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return f16(__habs(x.native()));
        #elif defined(__CUDA_ARCH__)
        return f16(abs(static_cast<f16::arithmetic_type>(x)));
        #else
        return f16(half_float::abs(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 min(f16 x, f16 y) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return f16(__hmin(x.native(), y.native()));
        #elif defined(__CUDA_ARCH__)
        return x < y ? x : y;
        #else
        return f16(half_float::fmin(x.native(), y.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 max(f16 x, f16 y) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return f16(__hmax(x.native(), y.native()));
        #elif defined(__CUDA_ARCH__)
        return y < x ? x : y;
        #else
        return f16(half_float::fmax(x.native(), y.native()));
        #endif
    }
}

#include <limits>
namespace std {
    // "Non-standard libraries may add specializations for library-provided types"
    // https://en.cppreference.com/w/cpp/types/numeric_limits
    template<>
    class numeric_limits<noa::f16> {
    public:
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = false;
        static constexpr bool is_exact = false;
        static constexpr bool is_modulo = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_iec559 = true;
        static constexpr bool has_infinity = true;
        static constexpr bool has_quiet_NaN = true;
        static constexpr bool has_signaling_NaN = true;
        static constexpr float_denorm_style has_denorm = std::denorm_present;
        static constexpr bool has_denorm_loss = false;

        #if HALF_ERRHANDLING_THROWS
        static constexpr bool traps = true;
        #else
        static constexpr bool traps = false;
        #endif

        static constexpr bool tinyness_before = false;
        static constexpr float_round_style round_style = std::round_to_nearest;

        static constexpr int digits = 11;
        static constexpr int digits10 = 3;
        static constexpr int max_digits10 = 5;
        static constexpr int radix = 2;
        static constexpr int min_exponent = -13;
        static constexpr int min_exponent10 = -4;
        static constexpr int max_exponent = 16;
        static constexpr int max_exponent10 = 4;

        static constexpr noa::f16 min() noexcept {
            return noa::f16::from_bits(0x0400);
        }

        static constexpr noa::f16 lowest() noexcept {
            return noa::f16::from_bits(0xFBFF);
        }

        static constexpr noa::f16 max() noexcept {
            return noa::f16::from_bits(0x7BFF);
        }

        static constexpr noa::f16 epsilon() noexcept {
            return noa::f16::from_bits(0x1400);
        }

        static constexpr noa::f16 round_error() noexcept {
            return noa::f16::from_bits(0x3800); // nearest
        }

        static constexpr noa::f16 infinity() noexcept {
            return noa::f16::from_bits(0x7C00);
        }

        static constexpr noa::f16 quiet_NaN() noexcept {
            return noa::f16::from_bits(0x7FFF);
        }

        static constexpr noa::f16 signaling_NaN() noexcept {
            return noa::f16::from_bits(0x7DFF);
        }

        static constexpr noa::f16 denorm_min() noexcept {
            return noa::f16::from_bits(0x0001);
        }
    };
}

namespace noa::details {
    template<>
    struct Stringify<f16> {
        static auto get() -> std::string { return "f16"; }
    };
}

namespace noa {
    inline std::ostream& operator<<(std::ostream& os, f16 half) {
        return os << static_cast<f32>(half);
    }
}

namespace fmt {
    template<>
    struct formatter<noa::f16> : formatter<float> {
        template<typename FormatContext>
        auto format(const noa::f16& vec, FormatContext& ctx) {
            return formatter<float>::format(static_cast<float>(vec), ctx);
        }

        template<typename FormatContext>
        auto format(const noa::f16& vec, FormatContext& ctx) const {
            return formatter<float>::format(static_cast<float>(vec), ctx);
        }
    };
}
