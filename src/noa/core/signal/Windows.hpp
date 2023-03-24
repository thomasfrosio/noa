#pragma once

#include <numeric>

#include "noa/core/Types.hpp"
#include "noa/core/Math.hpp"

namespace noa::signal {
    /// Returns a Gaussian symmetric window.
    /// \tparam T           Any floating-point type.
    /// \param[out] output  Output window.
    /// \param elements     Number of elements in the window.
    /// \param stddev       Standard deviation, sigma.
    /// \param normalize    Whether the window should normalized to have a total sum of 1.
    template<typename T>
    void gaussian(const T* output, i64 elements, T stddev, bool normalize = false) {
        if (elements <= 1) {
            if (elements)
                *output = T{1};
            return;
        }

        const auto half = static_cast<T>(elements) / T{2};
        const T sig2 = 2 * stddev * stddev;
        for (i64 i = 0; i < elements; ++i, ++output) {
            const T n = static_cast<T>(i) - half;
            *output = noa::math::exp(-(n * n) / sig2);
        }

        if (normalize) {
            T sum = std::accumulate(output, output + elements);
            for (i64 i = 0; i < elements; ++i, ++output)
                *output /= sum;
        }
    }
}
