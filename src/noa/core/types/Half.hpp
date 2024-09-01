#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/utils/Strings.hpp"

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wbool-compare"
#elif defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
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
// Their underlying type is uint16_t, and they can be used interchangeably.
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
    class Half {
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
        static_assert(alignof(native_type) == alignof(uint16_t));

    public:
        /// Default constructor.
        /// FIXME half_float zero-initializes, CUDA's __half doesn't.
        constexpr Half() noexcept = default;

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
        NOA_HD constexpr explicit Half(T x) : m_data(from_value<native_type>(x)) {}

    public:
        [[nodiscard]] static NOA_HD constexpr Half from_bits(uint16_t bits) noexcept { return Half(Empty{}, bits); }

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
                } else if constexpr (std::is_same_v<U, signed char> ||
                                     std::is_same_v<U, char> ||
                                     std::is_same_v<U, bool>) {
                    return __short2half_rn(static_cast<short>(value));
                } else if constexpr (std::is_same_v<U, unsigned char>) {
                    return __ushort2half_rn(static_cast<unsigned short>(value));
                } else if constexpr (std::is_same_v<U, short>) {
                    return __short2half_rn(value);
                } else if constexpr (std::is_same_v<U, ushort>) {
                    return __ushort2half_rn(value);
                } else if constexpr (std::is_same_v<U, int> || (std::is_same_v<U, long> && sizeof(long) == 4)) {
                    return __int2half_rn(value);
                } else if constexpr (std::is_same_v<U, uint> || (std::is_same_v<U, ulong> && sizeof(long) == 4)) {
                    return __uint2half_rn(value);
                } else if constexpr (std::is_same_v<U, long long> || std::is_same_v<U, long>) {
                    return __ll2half_rn(value);
                } else if constexpr (std::is_same_v<U, unsigned long long> || std::is_same_v<U, ulong>) {
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
                } else if constexpr (std::is_same_v<T, int> || (std::is_same_v<T, long> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2int_rn(value));
                } else if constexpr (std::is_same_v<T, uint> || (std::is_same_v<T, ulong> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2uint_rn(value));
                } else if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, long>) {
                    return static_cast<T>(__half2ll_rn(value));
                } else if constexpr (std::is_same_v<T, unsigned long long> || std::is_same_v<T, ulong>) {
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
            } else if constexpr (std::is_same_v<T, native_type> || std::is_same_v<U, native_type>) {
                // half_float::half_cast has a bug in int2half for the min value so check it beforehand.
                if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
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

        // Returns a copy of the native type.
        // On the host, it is half_float::half. On CUDA devices, it is __half.
        [[nodiscard]] NOA_HD constexpr native_type native() const noexcept { return m_data; }

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
        NOA_HD Half& operator+=(Half rhs) {
            return *this = *this + rhs;
        }
        NOA_HD Half& operator-=(Half rhs) {
            return *this = *this - rhs;
        }
        NOA_HD Half& operator*=(Half rhs) {
            return *this = *this * rhs;
        }
        NOA_HD Half& operator/=(Half rhs) {
            return *this = *this / rhs;
        }
        NOA_HD Half& operator+=(float rhs) {
            return *this = Half(static_cast<float>(*this) + rhs);
        }
        NOA_HD Half& operator-=(float rhs) {
            return *this = Half(static_cast<float>(*this) - rhs);
        }
        NOA_HD Half& operator*=(float rhs) {
            return *this = Half(static_cast<float>(*this) * rhs);
        }
        NOA_HD Half& operator/=(float rhs) {
            return *this = Half(static_cast<float>(*this) / rhs);
        }

        NOA_HD Half& operator++() {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = __half(static_cast<Half::arithmetic_type>(m_data) + 1);
            #else
            ++m_data;
            #endif
            return *this;
        }
        NOA_HD Half& operator--() {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = __half(static_cast<Half::arithmetic_type>(m_data) - 1);
            #else
            --m_data;
            #endif
            return *this;
        }

        NOA_HD Half operator++(int) {
            Half out(*this);
            ++(*this);
            return out;
        }
        NOA_HD Half operator--(int) {
            Half out(*this);
            --(*this);
            return out;
        }

        [[nodiscard]] NOA_HD friend Half operator+(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<Half::arithmetic_type>(lhs) + static_cast<Half::arithmetic_type>(rhs));
            #else
            return Half(lhs.m_data + rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend Half operator-(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<Half::arithmetic_type>(lhs) - static_cast<Half::arithmetic_type>(rhs));
            #else
            return Half(lhs.m_data - rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend Half operator*(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<Half::arithmetic_type>(lhs) * static_cast<Half::arithmetic_type>(rhs));
            #else
            return Half(lhs.m_data * rhs.m_data);
            #endif
        }
        [[nodiscard]] NOA_HD friend Half operator/(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<Half::arithmetic_type>(lhs) / static_cast<Half::arithmetic_type>(rhs));
            #else
            return Half(lhs.m_data / rhs.m_data);
            #endif
        }

        [[nodiscard]] NOA_HD friend Half operator+(Half lhs) {
            return lhs;
        }
        [[nodiscard]] NOA_HD friend Half operator-(Half lhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(-static_cast<Half::arithmetic_type>(lhs));
            #else
            return Half(-lhs.m_data);
            #endif
        }

        [[nodiscard]] NOA_HD friend bool operator==(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) == static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data == rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator!=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) != static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data != rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator>(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) > static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data > rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator<(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) < static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data < rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator>=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) >= static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data >= rhs.m_data;
            #endif
        }
        [[nodiscard]] NOA_HD friend bool operator<=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<Half::arithmetic_type>(lhs) <= static_cast<Half::arithmetic_type>(rhs);
            #else
            return lhs.m_data <= rhs.m_data;
            #endif
        }

    private:
        // Private constructor reinterpreting the bits. Used by f16::from_bits(u16).
        #if defined(__CUDA_ARCH__)
        constexpr Half(Empty, uint16_t bits) noexcept : m_data(__half_raw{bits}) {}
        #else
        constexpr Half(Empty, uint16_t bits) noexcept : m_data(half_float::reinterpret_as_half(bits)) {}
        // reinterpret_as_half is not native to the half_float namespace. It was added to our version
        // to allow reinterpretation from bits to half_float::half in constexpr context.
        #endif

    private:
        native_type m_data;
    };

    using f16 = Half;
    static_assert(sizeof(f16) == 2);
    static_assert(alignof(f16) == 2);
}

namespace noa {
    template<>
    struct nt::proclaim_is_real<Half> : std::true_type {};

    [[nodiscard]] NOA_FHD Half fma(Half x, Half y, Half z) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(__hfma(x.native(), y.native(), z.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(fma(static_cast<Half::arithmetic_type>(x),
                        static_cast<Half::arithmetic_type>(y),
                        static_cast<Half::arithmetic_type>(z)));
        #else
        return Half(half_float::fma(x.native(), y.native(), z.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half cos(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hcos(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(cos(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::cos(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half sin(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hsin(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(sin(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::sin(x.native()));
        #endif
    }

    NOA_FHD void sincos(Half x, Half* s, Half* c) {
        *s = sin(x); // FIXME use sincos instead?
        *c = cos(x);
    }

    [[nodiscard]] NOA_FHD Half tan(Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(tan(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::tan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half acos(Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(acos(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::acos(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half asin(Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(asin(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::asin(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half atan(Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(atan(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::atan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half atan2(Half y, Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(atan2(static_cast<Half::arithmetic_type>(y),
                          static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::atan2(y.native(), x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half rad2deg(Half x) {
        return Half(rad2deg(static_cast<Half::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD Half deg2rad(Half x) {
        return Half(deg2rad(static_cast<Half::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD Half pow(Half x, Half exp) {
        #if defined(__CUDA_ARCH__)
        return Half(pow(static_cast<Half::arithmetic_type>(x), static_cast<Half::arithmetic_type>(exp)));
        #else
        return Half(half_float::pow(x.native(), exp.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half exp(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hexp(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(exp(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::exp(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half log(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hlog(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(log(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::log(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half log10(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hlog10(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(log10(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::log10(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half hypot(Half x, Half y) {
        return Half(hypot(static_cast<Half::arithmetic_type>(x), static_cast<Half::arithmetic_type>(y)));
    }

    [[nodiscard]] NOA_FHD Half sqrt(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hsqrt(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(sqrt(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::sqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half rsqrt(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hrsqrt(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(rsqrt(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::rsqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half round(Half x) {
        #if defined(__CUDA_ARCH__)
        return Half(round(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::round(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half rint(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hrint(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(rint(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::rint(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half ceil(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hceil(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(ceil(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::ceil(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half floor(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(hfloor(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(floor(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::floor(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half trunc(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(htrunc(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(trunc(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::trunc(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half is_nan(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(__hisnan(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(isnan(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::isnan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half is_inf(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(__hisinf(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(isinf(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::isinf(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD bool is_finite(Half x) {
        return !(is_inf(x) || is_nan(x));
    }

    [[nodiscard]] NOA_FHD Half abs(Half x) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        return Half(__habs(x.native()));
        #elif defined(__CUDA_ARCH__)
        return Half(abs(static_cast<Half::arithmetic_type>(x)));
        #else
        return Half(half_float::abs(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half min(Half x, Half y) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return Half(__hmin(x.native(), y.native()));
        #elif defined(__CUDA_ARCH__)
        return x < y ? x : y;
        #else
        return Half(half_float::fmin(x.native(), y.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD Half max(Half x, Half y) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        return Half(__hmax(x.native(), y.native()));
        #elif defined(__CUDA_ARCH__)
        return y < x ? x : y;
        #else
        return Half(half_float::fmax(x.native(), y.native()));
        #endif
    }
}

namespace std {
    // "Non-standard libraries may add specializations for library-provided types"
    // https://en.cppreference.com/w/cpp/types/numeric_limits
    template<class T>
    class numeric_limits;

    template<>
    class numeric_limits<noa::Half> {
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

        static constexpr noa::Half min() noexcept {
            return noa::Half::from_bits(0x0400);
        }

        static constexpr noa::Half lowest() noexcept {
            return noa::Half::from_bits(0xFBFF);
        }

        static constexpr noa::Half max() noexcept {
            return noa::Half::from_bits(0x7BFF);
        }

        static constexpr noa::Half epsilon() noexcept {
            return noa::Half::from_bits(0x1400);
        }

        static constexpr noa::Half round_error() noexcept {
            return noa::Half::from_bits(0x3800); // nearest
        }

        static constexpr noa::Half infinity() noexcept {
            return noa::Half::from_bits(0x7C00);
        }

        static constexpr noa::Half quiet_NaN() noexcept {
            return noa::Half::from_bits(0x7FFF);
        }

        static constexpr noa::Half signaling_NaN() noexcept {
            return noa::Half::from_bits(0x7DFF);
        }

        static constexpr noa::Half denorm_min() noexcept {
            return noa::Half::from_bits(0x0001);
        }
    };
}

#ifdef NOA_IS_OFFLINE
namespace noa::string {
    template<>
    struct Stringify<Half> {
        static auto get() -> std::string { return "f16"; }
    };
}

namespace noa {
    inline std::ostream& operator<<(std::ostream& os, Half half) {
        return os << static_cast<float>(half);
    }
}

namespace fmt {
    template<>
    struct formatter<noa::Half> : formatter<float> {
        template<typename FormatContext>
        auto format(const noa::Half& vec, FormatContext& ctx) {
            return formatter<float>::format(static_cast<float>(vec), ctx);
        }

        template<typename FormatContext>
        auto format(const noa::Half& vec, FormatContext& ctx) const {
            return formatter<float>::format(static_cast<float>(vec), ctx);
        }
    };
}
#endif
