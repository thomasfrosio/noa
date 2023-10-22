#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

namespace noa::geometry {
    /// BSpline prefilter utility.
    /// \details Cubic B-spline curves are not constrained to pass through the data, i.e. data points are simply
    /// referred to as control points. As such, these curves are not really interpolating. For instance, a ratio of 0
    /// puts only 2/3 of the total weight on \a v1 and 1/3 on \a v0 and \a v2. One solution to this is to
    /// counter-weight in anticipation of a cubic B-spline function to force the function to go through the original
    /// data points. Combined with this filter, this function performs an actual interpolation of the data.
    ///
    /// \see This is adapted from https://github.com/DannyRuijters/CubicInterpolationCUDA
    ///      Copyright (c) 2008-2013, Danny Ruijters. All rights reserved.
    ///      http://www.dannyruijters.nl/cubicinterpolation/
    ///
    /// \note Compared to Danny's implementation:
    ///  - steps were switched to number of elements and strides were added.
    ///  - const was added when appropriate.
    ///  - In-place filtering was added.
    ///  - Support for double precision and complex types.
    template<typename Value, typename Int>
    class BSplinePrefilter1d {
    public:
        static_assert(nt::is_real_or_complex_v<Value> && nt::is_int_v<Int>);
        using value_type = Value;
        using real_type = nt::value_type_t<value_type>;
        using sint_type = std::make_signed<Int>;

        // math::sqrt(3.0f)-2.0f; pole for cubic b-spline
        static constexpr real_type POLE = static_cast<real_type>(-0.2679491924311228);
        static constexpr real_type LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

    public:
        NOA_HD static constexpr void filter_inplace(value_type* output, sint_type strides, sint_type size) {
            // causal initialization and recursion
            value_type* c = output;
            value_type previous_c;  //cache the previously calculated c rather than look it up again (faster!)
            *c = previous_c = LAMBDA * initial_causal_coefficient_(c, strides, size);
            for (sint_type n = 1; n < size; n++) {
                c += strides;
                *c = previous_c = LAMBDA * *c + POLE * previous_c;
            }

            // anticausal initialization and recursion
            *c = previous_c = initial_anticausal_coefficient_(c);
            for (sint_type n = size - 2; 0 <= n; n--) {
                c -= strides;
                *c = previous_c = POLE * (previous_c - *c);
            }
        }

        NOA_HD static constexpr void filter(
                const value_type* input, sint_type input_stride,
                value_type* output, sint_type output_stride,
                sint_type size
        ) {
            // causal initialization and recursion
            value_type* c = output;
            value_type previous_c;  // cache the previously calculated c rather than look it up again (faster!)
            *c = previous_c = LAMBDA * initial_causal_coefficient_(input, input_stride, size);
            for (sint_type n = 1; n < size; n++) {
                input += input_stride;
                c += output_stride;
                *c = previous_c = LAMBDA * *input + POLE * previous_c;
            }

            // anticausal initialization and recursion
            *c = previous_c = initial_anticausal_coefficient_(c);
            for (sint_type n = size - 2; 0 <= n; n--) {
                c -= output_stride;
                *c = previous_c = POLE * (previous_c - *c);
            }
        }

    private:
        [[nodiscard]] NOA_HD static constexpr value_type initial_causal_coefficient_(
                const value_type* c, sint_type stride, sint_type size
        ) {
            const sint_type horizon = min(sint_type{12}, size);

            // this initialization corresponds to clamping boundaries accelerated loop
            real_type zn = POLE;
            value_type sum = *c;
            for (sint_type n = 0; n < horizon; n++) {
                sum += zn * *c;
                zn *= POLE;
                c += stride;
            }
            return sum;
        }

        [[nodiscard]] NOA_HD static constexpr value_type initial_anticausal_coefficient_(
                const value_type* c
        ) {
            // this initialization corresponds to clamping boundaries
            return ((POLE / (POLE - 1)) * (*c));
        }
    };
}
