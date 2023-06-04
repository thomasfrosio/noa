#pragma once

#include <numeric>

#include "noa/core/Types.hpp"
#include "noa/core/Math.hpp"

namespace noa::signal {
    // Returns a Gaussian symmetric window.
    template<typename T>
    void gaussian_window(T* output, i64 elements, T stddev, bool normalize = false) {
        if (elements <= 1) {
            if (elements)
                *output = T{1};
            return;
        }

        auto half = static_cast<T>(elements / 2);
        if (!(elements % 2))
            half += T{0.5}; // if even, the window should still be symmetric

        const T sig2 = 2 * stddev * stddev;
        for (i64 i = 0; i < elements; ++i) {
            const T n = static_cast<T>(i) - half;
            output[i] = noa::math::exp(-(n * n) / sig2);
        }

        if (normalize) {
            const T sum = std::accumulate(output, output + elements, T{0});
            for (i64 i = 0; i < elements; ++i)
                output[i] /= sum;
        }
    }
}
