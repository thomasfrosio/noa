#pragma once

#include <cstdint>
#include <limits>
#include <cfloat> // FLT_EPSILON, DBL_EPSILON

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa::math {
    /// Some constants.
    template<typename T>
    struct Constants {
        static constexpr T PI = static_cast<T>(3.1415926535897932384626433832795);
        static constexpr T PI2 = static_cast<T>(6.283185307179586476925286766559);
        static constexpr T PIHALF = static_cast<T>(1.5707963267948966192313216916398);
    };

    /// Numeric limits.
    /// \note Use this type instead of \c std::numeric_limits to work on CUDA-device code
    ///       without the \c --expt-relaxed-constexpr flag on.
    template<typename T>
    struct Limits {
        NOA_FHD static constexpr T epsilon() {
            if constexpr (std::is_same_v<T, float>) {
                return FLT_EPSILON;
            } else if constexpr (std::is_same_v<T, double>) {
                return DBL_EPSILON;
            } else {
                static_assert(traits::always_false_v<T>);
            }
            return static_cast<T>(0); // unreachable
        }

        NOA_FHD static constexpr T min() {
            #ifdef __CUDA_ARCH__
            if constexpr (std::is_same_v<T, float>) {
                return FLT_MIN;
            } else if constexpr (std::is_same_v<T, double>) {
                return DBL_MIN;
            } else if constexpr (std::is_same_v<T, bool>) {
                return false;
            } else if constexpr (std::is_unsigned_v<T>) {
                return 0;
            } else if constexpr (std::is_same_v<T, signed char>) {
                return SCHAR_MIN;
            } else if constexpr (std::is_same_v<T, short>) {
                return SHRT_MIN;
            } else if constexpr (std::is_same_v<T, int>) {
                return INT_MIN;
            } else if constexpr (std::is_same_v<T, long>) {
                return LONG_MIN;
            } else if constexpr (std::is_same_v<T, long long>) {
                return LLONG_MIN;
            } else {
                static_assert(traits::always_false_v<T>);
            }
            return static_cast<T>(0); // unreachable
            #else
            return std::numeric_limits<T>::min();
            #endif
        }

        NOA_FHD static constexpr T max() {
            #ifdef __CUDA_ARCH__
            if constexpr (std::is_same_v<T, float>) {
                return FLT_MAX;
            } else if constexpr (std::is_same_v<T, double>) {
                return DBL_MAX;
            } else if constexpr (std::is_same_v<T, bool>) {
                return true;
            } else if constexpr (std::is_same_v<T, unsigned char>) {
                return UCHAR_MAX;
            } else if constexpr (std::is_same_v<T, unsigned short>) {
                return USHRT_MAX;
            } else if constexpr (std::is_same_v<T, unsigned int>) {
                return UINT_MAX;
            } else if constexpr (std::is_same_v<T, unsigned long>) {
                return ULONG_MAX;
            } else if constexpr (std::is_same_v<T, unsigned long long>) {
                return ULLONG_MAX;
            } else if constexpr (std::is_same_v<T, signed char>) {
                return SCHAR_MAX;
            } else if constexpr (std::is_same_v<T, short>) {
                return SHRT_MAX;
            } else if constexpr (std::is_same_v<T, int>) {
                return INT_MAX;
            } else if constexpr (std::is_same_v<T, long>) {
                return LONG_MAX;
            } else if constexpr (std::is_same_v<T, long long>) {
                return LLONG_MAX;
            } else {
                static_assert(traits::always_false_v<T>);
            }
            return static_cast<T>(0); // unreachable
            #else
            return std::numeric_limits<T>::max();
            #endif
        }

        NOA_FHD static constexpr T lowest() {
            #ifdef __CUDA_ARCH__
            if constexpr (traits::is_float_v<T>) {
                return -max();
            } else {
                return min();
            }
            return static_cast<T>(0); // unreachable
            #else
            return std::numeric_limits<T>::lowest();
            #endif
        }
    };
}
