#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/signal/Windows.hpp"
#include "noa/unified/Array.hpp"

namespace noa::signal {
    /// Returns a Gaussian symmetric window.
    /// \tparam T           Any floating-point type.
    /// \param elements     Number of elements in the window.
    /// \param stddev       Standard deviation, sigma.
    /// \param normalize    Whether the window should normalized to have a total sum of 1.
    /// \return CPU row contiguous vector with the gaussian window.
    template<typename T>
    Array<T> gaussian_window(i64 elements, f64 stddev, bool normalize = false) {
        Array<T> output(elements);
        gaussian_window(output.data(), elements, static_cast<T>(stddev), normalize);
        return output;
    }
}
