#pragma once

#include <cstdint>
#include <limits>
#include <cfloat> // FLT_EPSILON, DBL_EPSILON

#include "noa/core/Definitions.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::math {
    /// Some constants.
    template<typename Real>
    struct Constant {
        static_assert(noa::traits::is_real_v<Real>);
        static constexpr Real PI = static_cast<Real>(3.1415926535897932384626433832795);
        static constexpr Real PLANCK = static_cast<Real>(6.62607015e-34); // J.Hz-1
        static constexpr Real SPEED_OF_LIGHT = static_cast<Real>(299792458.0); // m.s
        static constexpr Real ELECTRON_MASS = static_cast<Real>(9.1093837015e-31); // kg
        static constexpr Real ELEMENTARY_CHARGE = static_cast<Real>(1.602176634e-19); // Coulombs
    };

    /// Numeric limits.
    /// \note Use this type instead of std::numeric_limits to work on CUDA-device code
    ///       without the --expt-relaxed-constexpr flag on.
    template<typename T>
    struct Limits {
        NOA_FHD static constexpr T epsilon() {
            if constexpr (std::is_same_v<T, float>) {
                return FLT_EPSILON;
            } else if constexpr (std::is_same_v<T, double>) {
                return DBL_EPSILON;
            } else {
                static_assert(noa::traits::always_false_v<T>);
            }
            return T{0}; // unreachable
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
                static_assert(noa::traits::always_false_v<T>);
            }
            return T{0}; // unreachable
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
                static_assert(noa::traits::always_false_v<T>);
            }
            return T{0}; // unreachable
            #else
            return std::numeric_limits<T>::max();
            #endif
        }

        NOA_FHD static constexpr T lowest() {
            #ifdef __CUDA_ARCH__
            if constexpr (noa::traits::is_real_v<T>) {
                return -max();
            } else {
                return min();
            }
            return T{0}; // unreachable
            #else
            return std::numeric_limits<T>::lowest();
            #endif
        }
    };
}
