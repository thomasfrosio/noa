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

    /// Transform the BDHW shape to match the rank for the transform.
    /// Note that the width is simply forwarded, so the real or logical shape can be used.
    ///
    /// If rank= 1: (B,D,H,W) reshaped to (BDH,1,1,W). Encodes BDH 1d transforms of shape W.
    /// If rank= 2: (B,D,H,W) reshaped to (BD, 1,H,W). Encodes BD 2d transforms of shape HW.
    /// If rank= 3: (B,D,H,W) reshaped to (B,  D,H,W). Encodes B 3d transforms of shape DHW.
    /// If rank=-1:
    ///     Shape 111W or B11W sets rank 1,
    ///     Shape 11HW or B1HW sets rank 2,
    ///     Shape 1DHW or BDHW sets rank 3,
    ///     Other shapes throw an error.
    template<typename T>
    constexpr auto ranked_shape(const Shape<T, 4>& shape, i32& rank) {
        const auto& [b, d, h, w] = shape;
        if (rank == -1) {
            if (d == 1 and h == 1)
                rank = 1;
            else if (d == 1 and h > 1)
                rank = 2;
            else if (d > 1 and h > 1)
                rank = 3;
            else
                panic("Cannot deduce rank from shape={}", shape);
            return shape;
        }
        if (rank == 1)
            return Shape4{b * d * h, 1, 1, w};
        if (rank == 2)
            return Shape4{b * d, 1, h, w};
        check(rank == 3, "Invalid rank. Should be -1, 1, 2, or 3, got {}", rank);
        return shape;
    }

    /// Returns [batches, shape_dhw, rank].
    template<typename T>
    [[nodiscard]] constexpr auto ranked_shape_info(const Shape<T, 4>& shape) noexcept {
        return noa::make_tuple(
            shape.template pop_back<3>(),
            shape.template pop_front<1>(),
            shape.rank()
        );
    }

    namespace details {
        template<typename T, typename U>
        void prepare_ranked_spans(Span<T, 4>& input, Span<U, 4>& output, Shape<isize, 4>& logical_shape, i32& rank) {
            logical_shape = ranked_shape(logical_shape, rank);
            input = input.reshape(logical_shape.set<3>(input.shape()[3]));
            output = output.reshape(logical_shape.set<3>(output.shape()[3]));
        }
    }
}
