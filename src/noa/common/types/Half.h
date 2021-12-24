#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/Math.h"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wbool-compare"
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
#if defined(__CUDA_ARCH__)
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

namespace noa {
    /// 16-bit precision float (IEEE-754-2008).
    /// \details This structure implements the datatype for storing half-precision floating-point numbers. Compared to
    ///          bfloat16, this type treads range for precision. There are 15361 representable numbers within the
    ///          interval [0.0, 1.0]. Its range is [6.10e-5, 6.55e4], so be careful with overflows.
    ///          This structure can be used on host code (using the "half" library from Christian Rau) and CUDA device
    ///          code (using the __half precision intrinsics from CUDA). Both implementations are using an unsigned
    ///          short as underlying type.
    /// \note For type-safety, they are no implicit constructors. This may be annoying sometimes, since it differs
    ///       from built-in floating-point types but it could potentially save us hours of debugging in the future.
    /// \note For device code, arithmetic operators and some math functions are only supported for devices with
    ///       compute capability >= 5.3 or 8.0. For devices with compute capability lower than that, higher precision
    ///       overloads are used internally (see HALF_ARITHMETIC_TYPE).
    class Half {
    public: // --- typedefs --- //
        #if defined(__CUDA_ARCH__)
        using native_t = __half;
        #else
        using native_t = half_float::half;
        #endif
        static_assert(std::is_standard_layout<native_t>::value);
        static_assert(std::is_nothrow_move_assignable<native_t>::value);
        static_assert(std::is_nothrow_move_constructible<native_t>::value);

    public: // --- Constructors --- //
        /// Default constructor, with initialization to 0.
        constexpr Half() = default;

        /// Conversion constructor.
        /// \details Any explicit initialization from a built-in type goes through that constructor.
        ///          No conversion warnings will be raised.
        /// \tparam T native_t, Half, float, double, (u)char, (u)short, (u)int, (u)long, (u)long long.
        /// \param x  Value to convert. Be careful with overflows.
        template<typename T>
        NOA_HD constexpr explicit Half(T x) : m_data(cast_<native_t>(x)) {}

        enum class Mode { BINARY };

        /// Conversion constructor.
        /// \details Reinterprets the \p bits as Half.
        NOA_HD constexpr Half(Mode, uint16_t bits) noexcept
        #if defined(__CUDA_ARCH__)
        : m_data(__ushort_as_half(bits)) {}
        #else
        // This function is not native to the half_float namespace. It was added to our version
        // to allow reinterpretation from bits to half_float::half in constexpr context.
                : m_data(half_float::reinterpret_as_half(bits)) {}
        #endif

    public:
        /// Returns a copy of the native type.
        /// On the host, it is half_float::half. On CUDA devices, it is __half.
        [[nodiscard]] NOA_HD constexpr native_t native() const noexcept { return m_data; }

    public: // --- Conversion to built-in types --- //
        NOA_HD explicit operator float() const {
            return cast_<float>(m_data);
        }
        NOA_HD explicit operator double() const {
            return cast_<double>(m_data);
        }
        NOA_HD explicit operator bool() const {
            return cast_<bool>(m_data);
        }
        NOA_HD explicit operator char() const {
            return cast_<char>(m_data);
        }
        NOA_HD explicit operator signed char() const {
            return cast_<signed char>(m_data);
        }
        NOA_HD explicit operator unsigned char() const {
            return cast_<unsigned char>(m_data);
        }
        NOA_HD explicit operator short() const {
            return cast_<short>(m_data);
        }
        NOA_HD explicit operator unsigned short() const {
            return cast_<unsigned short>(m_data);
        }
        NOA_HD explicit operator int() const {
            return cast_<int>(m_data);
        }
        NOA_HD explicit operator unsigned int() const {
            return cast_<unsigned int>(m_data);
        }
        NOA_HD explicit operator long() const {
            return cast_<long>(m_data);
        }
        NOA_HD explicit operator unsigned long() const {
            return cast_<unsigned long>(m_data);
        }
        NOA_HD explicit operator long long() const {
            return cast_<long long>(m_data);
        }
        NOA_HD explicit operator unsigned long long() const {
            return cast_<unsigned long long>(m_data);
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
            m_data = __half(static_cast<HALF_ARITHMETIC_TYPE>(m_data) + 1);
            #else
            ++m_data;
            #endif
            return *this;
        }
        NOA_HD Half& operator--() {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            m_data = __half(static_cast<HALF_ARITHMETIC_TYPE>(m_data) - 1);
            #else
            --m_data;
            #endif
            return *this;
        }

        Half operator++(int) {
            Half out(*this);
            ++(*this);
            return out;
        }
        Half operator--(int) {
            Half out(*this);
            --(*this);
            return out;
        }

        NOA_HD friend Half operator+(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<HALF_ARITHMETIC_TYPE>(lhs) + static_cast<HALF_ARITHMETIC_TYPE>(rhs));
            #else
            return Half(lhs.m_data + rhs.m_data);
            #endif
        }
        NOA_HD friend Half operator-(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<HALF_ARITHMETIC_TYPE>(lhs) - static_cast<HALF_ARITHMETIC_TYPE>(rhs));
            #else
            return Half(lhs.m_data - rhs.m_data);
            #endif
        }
        NOA_HD friend Half operator*(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<HALF_ARITHMETIC_TYPE>(lhs) * static_cast<HALF_ARITHMETIC_TYPE>(rhs));
            #else
            return Half(lhs.m_data * rhs.m_data);
            #endif
        }
        NOA_HD friend Half operator/(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(static_cast<HALF_ARITHMETIC_TYPE>(lhs) / static_cast<HALF_ARITHMETIC_TYPE>(rhs));
            #else
            return Half(lhs.m_data / rhs.m_data);
            #endif
        }

        NOA_HD friend Half operator+(Half lhs) {
            return lhs;
        }
        NOA_HD friend Half operator-(Half lhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return Half(-static_cast<HALF_ARITHMETIC_TYPE>(lhs));
            #else
            return Half(-lhs.m_data);
            #endif
        }

        NOA_HD friend bool operator==(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) == static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data == rhs.m_data;
            #endif
        }
        NOA_HD friend bool operator!=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) != static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data != rhs.m_data;
            #endif
        }
        NOA_HD friend bool operator>(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) > static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data > rhs.m_data;
            #endif
        }
        NOA_HD friend bool operator<(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) < static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data < rhs.m_data;
            #endif
        }
        NOA_HD friend bool operator>=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) >= static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data >= rhs.m_data;
            #endif
        }
        NOA_HD friend bool operator<=(Half lhs, Half rhs) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
            return static_cast<HALF_ARITHMETIC_TYPE>(lhs) <= static_cast<HALF_ARITHMETIC_TYPE>(rhs);
            #else
            return lhs.m_data <= rhs.m_data;
            #endif
        }

    public:
        NOA_HOST friend std::ostream& operator<<(std::ostream& os, Half half) {
            return os << cast_<float>(half.m_data);
        }

    private:
        // Cast to/from native_t. Most built-in type are supported.
        template<typename T, typename U>
        NOA_HD static T cast_(U arg) {
            #ifdef __CUDA_ARCH__
            if constexpr (std::is_same_v<T, U>) {
                return arg;
            } else if constexpr (std::is_same_v<T, native_t>) { // built-in -> native_t
                if constexpr (std::is_same_v<U, float>) {
                    return __float2half_rn(arg);
                } else if constexpr (std::is_same_v<U, double>) {
                    return __double2half(arg);
                } else if constexpr (std::is_same_v<U, signed char> ||
                                     std::is_same_v<U, char> ||
                                     std::is_same_v<U, bool>) {
                    return __short2half_rn(static_cast<short>(arg));
                } else if constexpr (std::is_same_v<U, unsigned char>) {
                    return __ushort2half_rn(static_cast<unsigned short>(arg));
                } else if constexpr (std::is_same_v<U, short>) {
                    return __short2half_rn(arg);
                } else if constexpr (std::is_same_v<U, ushort>) {
                    return __ushort2half_rn(arg);
                } else if constexpr (std::is_same_v<U, int> || (std::is_same_v<U, long> && sizeof(long) == 4)) {
                    return __int2half_rn(arg);
                } else if constexpr (std::is_same_v<U, uint> || (std::is_same_v<U, ulong> && sizeof(long) == 4)) {
                    return __uint2half_rn(arg);
                } else if constexpr (std::is_same_v<U, long long> || std::is_same_v<U, long>) {
                    return __ll2half_rn(arg);
                } else if constexpr (std::is_same_v<U, unsigned long long> || std::is_same_v<U, ulong>) {
                    return __ull2half_rn(arg);
                } else {
                    static_assert(noa::traits::always_false_v<T>);
                }
            } else if constexpr (std::is_same_v<U, native_t>) { // native_t -> built-in
                if constexpr (std::is_same_v<T, float>) {
                    return __half2float(arg);
                } else if constexpr (std::is_same_v<T, double>) {
                    return static_cast<double>(__half2float(arg));
                } else if constexpr (std::is_same_v<T, bool>) {
                    return static_cast<bool>(__half2short_rn(arg));
                } else if constexpr (std::is_same_v<T, signed char>) {
                    return static_cast<signed char>(__half2short_rn(arg));
                } else if constexpr (std::is_same_v<T, char>) {
                    return static_cast<char>(__half2short_rn(arg));
                } else if constexpr (std::is_same_v<T, unsigned char>) {
                    return static_cast<unsigned char>(__half2ushort_rn(arg));
                } else if constexpr (std::is_same_v<T, short>) {
                    return __half2short_rn(arg);
                } else if constexpr (std::is_same_v<T, ushort>) {
                    return __half2ushort_rn(arg);
                } else if constexpr (std::is_same_v<T, int> || (std::is_same_v<T, long> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2int_rn(arg));
                } else if constexpr (std::is_same_v<T, uint> || (std::is_same_v<T, ulong> && sizeof(long) == 4)) {
                    return static_cast<T>(__half2uint_rn(arg));
                } else if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, long>) {
                    return static_cast<T>(__half2ll_rn(arg));
                } else if constexpr (std::is_same_v<T, unsigned long long> || std::is_same_v<T, ulong>) {
                    return static_cast<T>(__half2ull_rn(arg));
                } else {
                    static_assert(noa::traits::always_false_v<T>);
                }
            } else {
                static_assert(noa::traits::always_false_v<T>);
            }
            return T(0); // unreachable
            #else
            if constexpr (std::is_same_v<T, U>) {
                return arg;
            } else if constexpr (std::is_same_v<T, native_t> || std::is_same_v<U, native_t>) {
                return half_float::half_cast<T>(arg);
            } else {
                static_assert(noa::traits::always_false_v<T>);
            }
            return T(0); // unreachable
            #endif
        }

    private:
        native_t m_data{};
    };

    using half_t = Half;

    template<>
    NOA_IH std::string string::typeName<half_t>() { return "half"; }

    template<>
    struct traits::proclaim_is_float<half_t> : std::true_type {};

    namespace math {
        template<typename T>
        struct Limits;

        template<>
        struct Limits<half_t> {
            NOA_FHD static constexpr half_t epsilon() {
                return {Half::Mode::BINARY, 0x1400};
            }

            NOA_FHD static constexpr half_t min() {
                return {Half::Mode::BINARY, 0x0400};
            }

            NOA_FHD static constexpr half_t max() {
                return {Half::Mode::BINARY, 0x7BFF};
            }

            NOA_FHD static constexpr half_t lowest() {
                return {Half::Mode::BINARY, 0xFBFF};
            }
        };

        NOA_FHD half_t fma(half_t x, half_t y, half_t z) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(__hfma(x.native(), y.native(), z.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(fma(static_cast<HALF_ARITHMETIC_TYPE>(x),
                            static_cast<HALF_ARITHMETIC_TYPE>(y),
                            static_cast<HALF_ARITHMETIC_TYPE>(z)));
            #else
            return Half(half_float::fma(x.native(), y.native(), z.native()));
            #endif
        }

        NOA_FHD half_t cos(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hcos(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(cos(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::cos(x.native()));
            #endif
        }

        NOA_FHD half_t sin(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hsin(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(sin(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::sin(x.native()));
            #endif
        }

        NOA_FHD void sincos(half_t x, half_t* s, half_t* c) {
            *s = sin(x); // use sincos instead?
            *c = cos(x);
        }

        NOA_FHD half_t tan(half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(tan(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::tan(x.native()));
            #endif
        }

        NOA_FHD half_t acos(half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(acos(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::acos(x.native()));
            #endif
        }

        NOA_FHD half_t asin(half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(asin(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::asin(x.native()));
            #endif
        }

        NOA_FHD half_t atan(half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(atan(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::atan(x.native()));
            #endif
        }

        NOA_FHD half_t atan2(half_t y, half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(atan2(static_cast<HALF_ARITHMETIC_TYPE>(y),
                              static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::atan2(y.native(), x.native()));
            #endif
        }

        NOA_FHD half_t toDeg(half_t x) {
            return Half(toDeg(static_cast<HALF_ARITHMETIC_TYPE>(x)));
        }

        NOA_FHD half_t toRad(half_t x) {
            return Half(toRad(static_cast<HALF_ARITHMETIC_TYPE>(x)));
        }

        NOA_FHD half_t exp(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hexp(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(exp(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::exp(x.native()));
            #endif
        }

        NOA_FHD half_t log(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hlog(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(log(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::log(x.native()));
            #endif
        }

        NOA_FHD half_t log10(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hlog10(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(log10(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::log10(x.native()));
            #endif
        }

        NOA_FHD half_t hypot(half_t x, half_t y) {
            return Half(hypot(static_cast<HALF_ARITHMETIC_TYPE>(x), static_cast<HALF_ARITHMETIC_TYPE>(y)));
        }

        NOA_FHD half_t sqrt(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hsqrt(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(sqrt(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::sqrt(x.native()));
            #endif
        }

        NOA_FHD half_t rsqrt(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hrsqrt(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(rsqrt(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::rsqrt(x.native()));
            #endif
        }

        NOA_FHD half_t round(half_t x) {
            #if defined(__CUDA_ARCH__)
            return Half(round(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::round(x.native()));
            #endif
        }

        NOA_FHD half_t rint(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hrint(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(rint(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::rint(x.native()));
            #endif
        }

        NOA_FHD half_t ceil(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hceil(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(ceil(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::ceil(x.native()));
            #endif
        }

        NOA_FHD half_t floor(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(hfloor(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(floor(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::floor(x.native()));
            #endif
        }

        NOA_FHD half_t trunc(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(htrunc(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(trunc(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::trunc(x.native()));
            #endif
        }

        NOA_FHD half_t isNaN(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(__hisnan(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(isNaN(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::isnan(x.native()));
            #endif
        }

        NOA_FHD half_t isInf(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(__hisinf(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(isInf(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::isinf(x.native()));
            #endif
        }

        NOA_FHD half_t abs(half_t x) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
            return Half(__habs(x.native()));
            #elif defined(__CUDA_ARCH__)
            return Half(abs(static_cast<HALF_ARITHMETIC_TYPE>(x)));
            #else
            return Half(half_float::abs(x.native()));
            #endif
        }

        NOA_FHD half_t min(half_t x, half_t y) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            return Half(__hmin(x.native(), y.native()));
            #elif defined(__CUDA_ARCH__)
            return x < y ? x : y;
            #else
            return Half(half_float::fmin(x.native(), y.native()));
            #endif
        }

        NOA_FHD half_t max(half_t x, half_t y) {
            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            return Half(__hmax(x.native(), y.native()));
            #elif defined(__CUDA_ARCH__)
            return y < x ? y : x;
            #else
            return Half(half_float::fmax(x.native(), y.native()));
            #endif
        }
    }
}

namespace fmt {
    template<>
    struct formatter<noa::Half> : formatter<float> {
        template<typename FormatContext>
        auto format(const noa::Half& vec, FormatContext& ctx) {
            return formatter<float>::format(static_cast<float>(vec), ctx);
        }
    };
}

/// Extensions to the C++ standard library.
namespace std {
    /// Numeric limits for half-precision floats.
    /// **See also:** Documentation for [std::numeric_limits](https://en.cppreference.com/w/cpp/types/numeric_limits)
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
            return {noa::Half::Mode::BINARY, 0x0400};
        }

        static constexpr noa::Half lowest() noexcept {
            return {noa::Half::Mode::BINARY, 0xFBFF};
        }

        static constexpr noa::Half max() noexcept {
            return {noa::Half::Mode::BINARY, 0x7BFF};
        }

        static constexpr noa::Half epsilon() noexcept {
            return {noa::Half::Mode::BINARY, 0x1400};
        }

        static constexpr noa::Half round_error() noexcept {
            return {noa::Half::Mode::BINARY, 0x3800}; // nearest
        }

        static constexpr noa::Half infinity() noexcept {
            return {noa::Half::Mode::BINARY, 0x7C00};
        }

        static constexpr noa::Half quiet_NaN() noexcept {
            return {noa::Half::Mode::BINARY, 0x7FFF};
        }

        static constexpr noa::Half signaling_NaN() noexcept {
            return {noa::Half::Mode::BINARY, 0x7DFF};
        }

        static constexpr noa::Half denorm_min() noexcept {
            return {noa::Half::Mode::BINARY, 0x0001};
        }
    };
}
