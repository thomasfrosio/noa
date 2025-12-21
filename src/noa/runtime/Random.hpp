#pragma once

#include <random>
#include "noa/runtime/core/Random.hpp"
#include "noa/runtime/Ewise.hpp"
#include "noa/runtime/Array.hpp"

namespace noa {
    /// Randomizes an array with uniform random values.
    /// \param[in] distribution A value distribution: Uniform, Normal, LogNormal or Poisson.
    /// \param[out] output      Array to randomize.
    template<nt::distribution Distribution, nt::writable_varray_decay Output>
    void randomize(const Distribution& distribution, Output&& output) {
        ewise({}, std::forward<Output>(output), Randomizer(distribution, std::random_device{}()));
    }

    /// Returns an array initialized with random values.
    template<typename T = void, nt::distribution Distribution>
    [[nodiscard]] auto random(const Distribution& distribution, const Shape4& shape, ArrayOption option = {}) {
        using value_t = std::conditional_t<std::is_void_v<T>, nt::value_type_t<Distribution>, T>;
        Array<value_t> out(shape, option);
        randomize(distribution, out);
        return out;
    }

    /// Returns an array initialized with random values.
    template<typename T = void, nt::distribution Distribution>
    [[nodiscard]] auto random(const Distribution& distribution, isize n_elements, ArrayOption option = {}) {
        using value_t = std::conditional_t<std::is_void_v<T>, nt::value_type_t<Distribution>, T>;
        Array<value_t> out(n_elements, option);
        randomize(distribution, out);
        return out;
    }
}
