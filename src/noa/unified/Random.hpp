#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/math/Distribution.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Array.hpp"
#include <random>

namespace noa {
    /// Randomizes an array with uniform random values.
    /// \param[in] distribution A value distribution, such as Uniform, Normal, LogNormal or Poisson.
    /// \param[out] output      Array to randomize.
    template<typename Distribution, typename Output>
    requires (nt::is_distribution_v<Distribution> and nt::is_varray_v<Output>)
    void randomize(const Distribution& distribution, const Output& output) {
        ewise({}, output, Randomizer(distribution, std::random_device{}()));
    }

    /// Returns an array initialized with random values.
    template<typename T, typename Distribution>
    requires (nt::is_distribution_v<Distribution> and nt::is_numeric_v<T>)
    [[nodiscard]] Array<T> random(const Distribution& distribution, const Shape4<i64>& shape, ArrayOption option = {}) {
        Array<T> out(shape, option);
        randomize(distribution, out);
        return out;
    }

    /// Returns an array initialized with random values.
    template<typename T, typename Distribution>
    requires (nt::is_distribution_v<Distribution> and nt::is_numeric_v<T>)
    [[nodiscard]] Array<T> random(const Distribution& distribution, i64 n_elements, ArrayOption option = {}) {
        Array<T> out(n_elements, option);
        randomize(distribution, out);
        return out;
    }
}
#endif
