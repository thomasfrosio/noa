#pragma once

#include "noa/core/signal/Windows.hpp"
#include "noa/unified/Array.hpp"

namespace noa::signal {
    /// Computes the gaussian (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param options      Window options.
    template<typename T>
    auto window_gaussian(isize n_elements, f64 stddev, WindowOptions options = {}) -> Array<T> {
        auto output = Array<T>(options.half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_gaussian(output.data(), n_elements, stddev, {options.normalize, options.half_window});
        return output;
    }

    /// Computes the blackman (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param options      Window options.
    template<typename T>
    auto window_blackman(isize n_elements, WindowOptions options = {}) -> Array<T> {
        auto output = Array<T>(options.half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_blackman(output.data(), n_elements, {options.normalize, options.half_window});
        return output;
    }

    /// Computes the sinc (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param options      Window options.
    template<typename T>
    auto window_sinc(isize n_elements, f64 constant, WindowOptions options = {}) -> Array<T> {
        auto output = Array<T>(options.half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_sinc(output.data(), n_elements, constant, {options.normalize, options.half_window});
        return output;
    }
}
