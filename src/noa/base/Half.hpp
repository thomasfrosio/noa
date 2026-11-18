#pragma once

#include <limits>
#include "noa/base/Config.hpp"
#include "noa/base/Math.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/Traits.hpp"

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wbool-compare"
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#endif

// Fallback to this type when half-precision implementation is not available
// or not appropriate (too slow or too low precision).
#ifndef HALF_ARITHMETIC_TYPE
#define HALF_ARITHMETIC_TYPE float
#endif

// Always share the underlying fp16 type between host and device passes.
// When the provided fp16 doesn't support an operation, our Half type should handle it
// (usually by upgrading to HALF_ARITHMETIC_TYPE behind the scene).
#if defined(NOA_ENABLE_CUDA)
    #include <cuda_runtime_api.h>
    #include <cuda_fp16.h>
#elif defined(NOA_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
#elif defined(NOA_ENABLE_SYCL)
    #include <sycl/sycl.hpp>
    #include <bit>
#else
    #include <half/half.hpp>
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#endif

// TODO C++23 replace with std::float16_t and std::bfloat16_t?

namespace noa::inline types {
    /// 16-bit precision float (IEEE-754-2008).
    /// \details
    ///     There are 15361 representable numbers within the interval [0.0, 1.0].
    ///     Its range is [6.10e-5, -/+6.55e4], so be careful with overflows.
    ///     After passed -/+ 2048, not all integral values are representable. For instance, int(half(int(2049)) == 2048.
    ///     This behavior is also true for single and double precision floats, but for much larger values
    ///     (see https://stackoverflow.com/a/3793950).
    class f16 {
    public: // --- typedefs --- //
        using arithmetic_type = HALF_ARITHMETIC_TYPE;

        // in class f16:
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        using native_type = __half;
        #elif defined(NOA_ENABLE_SYCL)
        using native_type = sycl::half;
        #else
        using native_type = half_float::half;
        #endif

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
        ///       something like c16{1, 0} is not allowed and we have to write c16{f16{1}, f16{0}}. Because f16 is
        ///       mostly a type to cast to/from, this is probably fine (e.g. c32{..}.as<f16>()).
        template<typename T>
        NOA_HD constexpr explicit f16(T x) : m_data(from_value<native_type>(x)) {}

    public:
        [[nodiscard]] static NOA_HD constexpr f16 from_bits(u16 bits) noexcept { return f16(Empty{}, bits); }

        // Cast built-in to/from native_type.
        template<typename T, typename U>
        [[nodiscard]] static NOA_HD constexpr T from_value(U value) {
            #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
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
                    static_assert(nt::always_false<T>);
                }
            } else if constexpr (std::is_same_v<U, native_type>) { // native_type -> built-in
                if constexpr (std::is_same_v<T, float>) {
                    return __half2float(value);
                } else if constexpr (std::is_same_v<T, double>) {
                    return static_cast<double>(__half2float(value));
                } else if constexpr (std::is_same_v<T, bool>) {
                    return static_cast<bool>(__half2short_rz(value));
                } else if constexpr (std::is_same_v<T, signed char>) {
                    return static_cast<signed char>(__half2short_rz(value));
                } else if constexpr (std::is_same_v<T, char>) {
                    return static_cast<char>(__half2short_rz(value));
                } else if constexpr (std::is_same_v<T, unsigned char>) {
                    return static_cast<unsigned char>(__half2ushort_rz(value));
                } else if constexpr (std::is_same_v<T, short>) {
                    return __half2short_rz(value);
                } else if constexpr (std::is_same_v<T, ushort>) {
                    return __half2ushort_rz(value);
                } else if constexpr (std::is_same_v<T, int> or (std::is_same_v<T, long> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2int_rz(value));
                } else if constexpr (std::is_same_v<T, uint> or (std::is_same_v<T, ulong> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2uint_rz(value));
                } else if constexpr (std::is_same_v<T, long long> or std::is_same_v<T, long>) {
                    return static_cast<T>(__half2ll_rz(value));
                } else if constexpr (std::is_same_v<T, unsigned long long> or std::is_same_v<T, ulong>) {
                    return static_cast<T>(__half2ull_rz(value));
                } else {
                    static_assert(nt::always_false<T>);
                }
            } else {
                static_assert(nt::always_false<T>);
            }
            #elif defined(NOA_ENABLE_SYCL)
            if constexpr (std::is_same_v<T, U>) {
                return value;
            } else if constexpr (std::is_same_v<T, native_type>) { // built-in -> native_type
                return native_type(static_cast<float>(value));
            } else if constexpr (std::is_same_v<U, native_type>) { // native_type -> built-in
                const auto f = static_cast<float>(value);
                if constexpr (std::is_same_v<T, bool>)
                    return static_cast<short>(f) != 0; // same as the CUDA branch (round toward zero, then != 0)
                else
                    return static_cast<T>(f); // truncates toward zero (=_rz).
            } else {
                static_assert(nt::always_false<T>);
            }
            #else
            if constexpr (std::is_same_v<T, U>) {
                return value;
            } else if constexpr (std::is_same_v<T, native_type> or std::is_same_v<U, native_type>) {
                // half_float::half_cast has a bug in int2half for the min value so check it beforehand.
                if constexpr (std::is_integral_v<U> and std::is_signed_v<U>) { // integer to half
                    if (value == std::numeric_limits<U>::min()) {
                        if constexpr (sizeof(U) == 1)
                            return half_float::reinterpret_as_half(0xD800); // -128
                        else if constexpr(sizeof(U) == 2)
                            return half_float::reinterpret_as_half(0xF800); // -32768
                        else
                            return half_float::reinterpret_as_half(0xFC00); // -inf
                    }
                }
                if constexpr (std::is_integral_v<T>) { // half to integer
                    // Explicitly round to zero when converting to an integer type.
                    // This should match static_cast<int>(float) where the fractional part is dropped,
                    // and matches CUDA's f16 branch with the __half2*_rz host/device functions.
                    return half_float::half_cast<T, std::round_toward_zero>(value);
                } else {
                    return half_float::half_cast<T>(value);
                }
            } else {
                static_assert(nt::always_false<T>);
            }
            #endif
        }

        [[nodiscard]] NOA_HD constexpr auto native() const noexcept -> const native_type& { return m_data; }
        [[nodiscard]] NOA_HD constexpr auto native() noexcept -> native_type& { return m_data; }

    public:
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

    public:
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
            #if defined(__NVCOMPILER)
            NV_IF_TARGET(NV_PROVIDES_SM_53,
                (++m_data;),
                (m_data = native_type(static_cast<f16::arithmetic_type>(m_data) + 1);))
            #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = native_type(static_cast<f16::arithmetic_type>(m_data) + 1);
            #else
            ++m_data; // CUDA (sm_53+ and host), SYCL
            #endif
            return *this;
        }
        NOA_HD f16& operator--() {
            #if defined(__NVCOMPILER)
            NV_IF_TARGET(NV_PROVIDES_SM_53,
                (--m_data;),
                (m_data = native_type(static_cast<f16::arithmetic_type>(m_data) - 1);))
            #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = native_type(static_cast<f16::arithmetic_type>(m_data) - 1);
            #else
            --m_data; // CUDA (sm_53+ and host), SYCL
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
            return f16(lhs.m_data + rhs.m_data);
        }
        [[nodiscard]] NOA_HD friend f16 operator-(f16 lhs, f16 rhs) {
            return f16(lhs.m_data - rhs.m_data);
        }
        [[nodiscard]] NOA_HD friend f16 operator*(f16 lhs, f16 rhs) {
            return f16(lhs.m_data * rhs.m_data);
        }
        [[nodiscard]] NOA_HD friend f16 operator/(f16 lhs, f16 rhs) {
            return f16(lhs.m_data / rhs.m_data);
        }

        [[nodiscard]] NOA_HD friend f16 operator+(f16 lhs) {
            return lhs;
        }
        [[nodiscard]] NOA_HD friend f16 operator-(f16 lhs) {
            return f16(-lhs.m_data);
        }

        [[nodiscard]] NOA_HD friend bool operator==(f16 lhs, f16 rhs) {
            return lhs.m_data == rhs.m_data;
        }
        [[nodiscard]] NOA_HD friend bool operator!=(f16 lhs, f16 rhs) {
            return lhs.m_data != rhs.m_data;
        }
        [[nodiscard]] NOA_HD friend bool operator>(f16 lhs, f16 rhs) {
            return lhs.m_data > rhs.m_data;
        }
        [[nodiscard]] NOA_HD friend bool operator<(f16 lhs, f16 rhs) {
            return lhs.m_data < rhs.m_data;
        }
        [[nodiscard]] NOA_HD friend bool operator>=(f16 lhs, f16 rhs) {
            return lhs.m_data >= rhs.m_data;
        }
        [[nodiscard]] NOA_HD friend bool operator<=(f16 lhs, f16 rhs) {
            return lhs.m_data <= rhs.m_data;
        }

    private:
        // Private constructor reinterpreting the bits. Used by f16::from_bits(u16).
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        constexpr f16(Empty, u16 bits) noexcept : m_data(__half_raw{bits}) {}
        #elif defined(NOA_ENABLE_SYCL)
        constexpr f16(Empty, u16 bits) noexcept : m_data(std::bit_cast<native_type>(bits)) {}
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
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(__hfma(x.native(), y.native(), z.native()));),
                           (return f16(fma(static_cast<f16::arithmetic_type>(x),
                               static_cast<f16::arithmetic_type>(y),
                               static_cast<f16::arithmetic_type>(z)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::fma(x.native(), y.native(), z.native()));
        #else
        return f16(half_float::fma(x.native(), y.native(), z.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 cos(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hcos(x.native()));),
                           (return f16(cos(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::cos(x.native()));
        #else
        return f16(half_float::cos(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sin(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hsin(x.native()));),
                           (return f16(sin(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::sin(x.native()));
        #else
        return f16(half_float::sin(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 tan(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(tan(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::tan(x.native()));
        #else
        return f16(half_float::tan(x.native()));
        #endif
    }

    NOA_FHD void sincos(f16 x, f16* s, f16* c) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP) || defined(NOA_ENABLE_SYCL)
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
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(cosh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::cosh(x.native()));
        #else
        return f16(half_float::cosh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sinh(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(sinh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::sinh(x.native()));
        #else
        return f16(half_float::sinh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 tanh(f16 x) {
        #if defined(NOA_ENABLE_CUDA) && CUDART_VERSION >= 13000
        NOA_IF_TARGET_SM53((return f16(htanh(x.native()));),
                           (return f16(tanh(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(tanh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::tanh(x.native()));
        #else
        return f16(half_float::tanh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 acos(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(acos(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::acos(x.native()));
        #else
        return f16(half_float::acos(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 asin(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(asin(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::asin(x.native()));
        #else
        return f16(half_float::asin(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atan(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(atan(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::atan(x.native()));
        #else
        return f16(half_float::atan(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 acosh(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(acosh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::acosh(x.native()));
        #else
        return f16(half_float::acosh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 asinh(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(asinh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::asinh(x.native()));
        #else
        return f16(half_float::asinh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atanh(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(atanh(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::atanh(x.native()));
        #else
        return f16(half_float::atanh(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 atan2(f16 y, f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(atan2(static_cast<f16::arithmetic_type>(y), static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::atan2(y.native(), x.native()));
        #else
        return f16(half_float::atan2(y.native(), x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rad2deg(f16 x) {
        return f16(rad2deg(static_cast<f16::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD f16 deg2rad(f16 x) {
        return f16(deg2rad(static_cast<f16::arithmetic_type>(x)));
    }

    [[nodiscard]] NOA_FHD f16 pow(f16 x, f16 exp) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(pow(static_cast<f16::arithmetic_type>(x), static_cast<f16::arithmetic_type>(exp)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::pow(x.native(), exp.native()));
        #else
        return f16(half_float::pow(x.native(), exp.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 exp(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hexp(x.native()));),
                           (return f16(exp(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::exp(x.native()));
        #else
        return f16(half_float::exp(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hlog(x.native()));),
                           (return f16(log(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::log(x.native()));
        #else
        return f16(half_float::log(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log10(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hlog10(x.native()));),
                           (return f16(log10(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::log10(x.native()));
        #else
        return f16(half_float::log10(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 log1p(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(log1p(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::log1p(x.native()));
        #else
        return f16(half_float::log1p(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 hypot(f16 x, f16 y) {
        #if defined(NOA_ENABLE_SYCL)
        return f16(sycl::hypot(x.native(), y.native()));
        #else
        return f16(hypot(static_cast<f16::arithmetic_type>(x), static_cast<f16::arithmetic_type>(y)));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 sqrt(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hsqrt(x.native()));),
                           (return f16(sqrt(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::sqrt(x.native()));
        #else
        return f16(half_float::sqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rsqrt(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hrsqrt(x.native()));),
                           (return f16(rsqrt(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::rsqrt(x.native()));
        #else
        return f16(half_float::rsqrt(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 round(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        return f16(round(static_cast<f16::arithmetic_type>(x)));
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::round(x.native()));
        #else
        return f16(half_float::round(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 rint(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hrint(x.native()));),
                           (return f16(rint(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::rint(x.native()));
        #else
        return f16(half_float::rint(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 ceil(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hceil(x.native()));),
                           (return f16(ceil(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::ceil(x.native()));
        #else
        return f16(half_float::ceil(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 floor(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(hfloor(x.native()));),
                           (return f16(floor(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::floor(x.native()));
        #else
        return f16(half_float::floor(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 trunc(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(htrunc(x.native()));),
                           (return f16(trunc(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::trunc(x.native()));
        #else
        return f16(half_float::trunc(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD bool is_nan(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return __hisnan(x.native());),
                           (return is_nan(static_cast<f16::arithmetic_type>(x));))
        #elif defined(NOA_ENABLE_SYCL)
        return sycl::isnan(x.native());
        #else
        return half_float::isnan(x.native());
        #endif
    }

    [[nodiscard]] NOA_FHD bool is_inf(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return __hisinf(x.native()) != 0;),
                           (return is_inf(static_cast<f16::arithmetic_type>(x));))
        #elif defined(NOA_ENABLE_SYCL)
        return sycl::isinf(x.native());
        #else
        return half_float::isinf(x.native());
        #endif
    }

    [[nodiscard]] NOA_FHD bool is_finite(f16 x) {
        #if defined(NOA_ENABLE_SYCL)
        return sycl::isfinite(x.native());
        #else
        return !(is_inf(x) || is_nan(x));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 abs(f16 x) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM53((return f16(__habs(x.native()));),
                           (return f16(abs(static_cast<f16::arithmetic_type>(x)));))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::fabs(x.native()));
        #else
        return f16(half_float::abs(x.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 min(f16 x, f16 y) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM80((return f16(__hmin(x.native(), y.native()));),
                           (return x < y ? x : y;))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::fmin(x.native(), y.native()));
        #else
        return f16(half_float::fmin(x.native(), y.native()));
        #endif
    }

    [[nodiscard]] NOA_FHD f16 max(f16 x, f16 y) {
        #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
        NOA_IF_TARGET_SM80((return f16(__hmax(x.native(), y.native()));),
                           (return y < x ? x : y;))
        #elif defined(NOA_ENABLE_SYCL)
        return f16(sycl::fmax(x.native(), y.native()));
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
