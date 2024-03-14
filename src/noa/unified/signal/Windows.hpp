#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/signal/Windows.hpp"
#include "noa/unified/Array.hpp"

namespace noa::signal {
    /// Computes the gaussian (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param stddev       Standard deviation of the gaussian.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, the output array only has \c (n_elements-1)//2+1 n_elements.
    template<typename T>
    Array<T> window_gaussian(i64 n_elements, f64 stddev, bool normalize = false, bool half_window = false) {
        Array<T> output(half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_gaussian(output.data(), n_elements, stddev, normalize, half_window);
        return output;
    }

    /// Computes the blackman (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, the output array only has \c (n_elements-1)//2+1 n_elements.
    template<typename T>
    Array<T> window_blackman(i64 n_elements, bool normalize = false, bool half_window = false) {
        Array<T> output(half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_blackman(output.data(), n_elements, normalize, half_window);
        return output;
    }

    /// Computes the sinc (half-)window.
    /// \param n_elements   Number of elements in the full-window.
    /// \param constant     Additional constant factor. \c sin(constant*pi*x)/(pi*x) is computed.
    /// \param normalize    Whether to normalize the sum of the full-window to 1.
    /// \param half_window  Whether to only compute the second half of the window.
    ///                     If true, the output array only has \c (n_elements-1)//2+1 n_elements.
    template<typename T>
    Array<T> window_sinc(i64 n_elements, f64 constant, bool normalize = false, bool half_window = false) {
        Array<T> output(half_window ? (n_elements - 1) / 2 + 1 : n_elements);
        window_sinc(output.data(), n_elements, constant, normalize, half_window);
        return output;
    }
}
