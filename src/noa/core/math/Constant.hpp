#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

namespace noa {
    template<typename Real>
    struct Constant {
        static_assert(nt::is_real_v<Real>);
        static constexpr Real PI = static_cast<Real>(3.1415926535897932384626433832795);
        static constexpr Real PLANCK = static_cast<Real>(6.62607015e-34); // J.Hz-1
        static constexpr Real SPEED_OF_LIGHT = static_cast<Real>(299792458.0); // m.s
        static constexpr Real ELECTRON_MASS = static_cast<Real>(9.1093837015e-31); // kg
        static constexpr Real ELEMENTARY_CHARGE = static_cast<Real>(1.602176634e-19); // Coulombs
    };
}
