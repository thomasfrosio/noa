#pragma once

#include "noa/base/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Span.hpp"

namespace noa::fft {
    /// Sign of the exponent in the formula that defines the c2c Fourier transform.
    /// Either FORWARD (-1) for the forward/direct transform, or BACKWARD (+1) for the backward/inverse transform.
    enum class Sign : i32 {
        FORWARD = -1,
        BACKWARD = 1
    };

    /// Normalization mode. Indicates which direction of the forward/backward pair of transforms is scaled and
    /// with what normalization factor. ORTHO scales each transform with 1/sqrt(N).
    enum class Norm {
        FORWARD,
        ORTHO,
        BACKWARD,
        NONE
    };

    namespace details {
        /// Returns [batches, shape_dhw, rank].
        template<typename T>
        [[nodiscard]] constexpr auto ranked_shape_info(const Shape<T, 4>& shape) noexcept {
            return noa::make_tuple(
                shape[0],
                shape.template pop_front<1>(),
                shape.template rank<i32>()
            );
        }
    }
}
