#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

namespace noa {
    template<typename T>
    requires nt::is_real_v<T>
    struct Constant {
        static constexpr T PI = static_cast<T>(3.1415926535897932384626433832795);
        static constexpr T PLANCK = static_cast<T>(6.62607015e-34); // J.Hz-1
        static constexpr T SPEED_OF_LIGHT = static_cast<T>(299792458.0); // m.s
        static constexpr T ELECTRON_MASS = static_cast<T>(9.1093837015e-31); // kg
        static constexpr T ELEMENTARY_CHARGE = static_cast<T>(1.602176634e-19); // Coulombs
    };
}
