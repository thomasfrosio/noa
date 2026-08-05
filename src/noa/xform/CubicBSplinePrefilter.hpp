#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::xform::details {
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
    template<nt::real_or_complex Value, nt::integer Int>
    class BSplinePrefilter1d {
    public:
        using value_type = Value;
        using real_type = nt::value_type_t<value_type>;
        using sint_type = std::make_signed_t<Int>;

        // sqrt(3.0f)-2.0f; pole for cubic b-spline
        static constexpr real_type POLE = static_cast<real_type>(-0.2679491924311228);
        static constexpr real_type LAMBDA = (1 - POLE) * (1 - 1 / POLE); // compute the overall gain

    public:
        NOA_HD static constexpr void filter_inplace(value_type* output, sint_type strides, sint_type size) {
            // causal initialization and recursion
            value_type* c = output;
            value_type previous_c; // cache the previously calculated c rather than look it up again (faster!)
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
        [[nodiscard]] NOA_HD static constexpr auto initial_causal_coefficient_(
            const value_type* c, sint_type stride, sint_type size
        ) -> value_type {
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

        [[nodiscard]] NOA_HD static constexpr auto initial_anticausal_coefficient_(
            const value_type* c
        ) -> value_type {
            // this initialization corresponds to clamping boundaries
            return (POLE / (POLE - 1)) * (*c);
        }
    };

    template<usize N, nt::readable_nd<N> Input, nt::writable_nd<N> Output>
    struct BSplinePrefilter {
        Input input;
        Output output;
        isize width;

        constexpr auto operator()(const Vec<isize, N - 1>& indices) const {
            const auto input_1d = input[indices];
            const auto output_1d = output[indices];
            BSplinePrefilter1d<nt::value_type_t<Output>, isize>::filter(
                input_1d.get(), input_1d.template stride<0>(),
                output_1d.get(), output_1d.template stride<0>(), width
            );
        }
    };

    template<usize N, nt::writable_nd<N> Output>
    struct BSplinePrefilterInplace {
        Output output;
        isize width;

        constexpr auto operator()(const Vec<isize, N - 1>& indices) const {
            const auto output_1d = output[indices];
            BSplinePrefilter1d<nt::value_type_t<Output>, isize>::filter_inplace(
                output_1d.get(), output_1d.template stride<0>(), width
            );
        }
    };

    template<typename T, usize N, typename... A>
        void cubic_bspline_prefilter_width(
        const Span<const T, N, isize>& input,
        const Span<T, N, isize>& output,
        Device device,
        A&&... arrays
    ) {
        constexpr auto OPTIONS = noa::IwiseOptions{
            .gpu_block_shape = {1, 1, 128},
            .gpu_optimize_block_shape = true,
        };
        auto width = output.shape()[N - 1];
        noa::iwise<OPTIONS>(
            output.shape().pop_back(), device,
            BSplinePrefilter<N, Accessor<const T, N, isize>, Accessor<T, N, isize>>{
                .input = input.accessor(),
                .output = output.accessor(),
                .width = width,
            }, std::forward<A>(arrays)...);
    }

    template<typename T, usize N, typename... A>
    void cubic_bspline_prefilter_width_inplace(
        const Span<T, N, isize>& output,
        Device device,
        A&&... arrays
    ) {
        constexpr auto OPTIONS = noa::IwiseOptions{
            .gpu_block_shape = {1, 1, 128}, // FIXME
            .gpu_optimize_block_shape = true,
        };
        auto width = output.shape()[N - 1];
        noa::iwise<OPTIONS>(output.shape().pop_back(), device,
            BSplinePrefilterInplace<N, Accessor<T, N, isize>>{
                .output = output.accessor(),
                .width = width,
            }, std::forward<A>(arrays)...);
    }
}

namespace noa::xform {
    struct CubicBSplinePrefilterOptions {
        /// Runtime rank.
        /// It is ignored if the rank is specified at compile time.
        /// If 0, the rank is deduced with Shape::rank_checked.
        usize rank{};
    };

    /// Applies a prefilter to the input so that the cubic B-spline values will pass through the sample data.
    /// \details
    ///     Without prefiltering, cubic B-spline filtering results in smoothened images. This is because the cubic
    ///     B-spline filtering yields a function that does not pass through its coefficients, i.e., it is not an
    ///     interpolating spline. To end up with cubic B-spline interpolated data that passes through the original
    ///     samples, we need to prefilter the input.
    /// \tparam RANK:       Rank of the data. If 0, uses the runtime options.rank.
    /// \param[in] input    Input array of f32, f64, c32, or c64.
    /// \param[out] output  Output array. Can be equal to the input.
    ///
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/ for more details.
    template<usize RANK = 0,
             nt::readable_array_decay_of_almost_any<f32, f64, c32, c64> Input,
             nt::writable_array_decay_of_any<nt::mutable_value_type_t<Input>> Output>
        requires (nt::array_decay_with_same_nd<Input, Output> and nt::array_size_v<Output> >= RANK)
    void cubic_bspline_prefilter(Input&& input, Output&& output, const CubicBSplinePrefilterOptions& options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        // If runtime rank, reshape to BDHW.
        constexpr usize RUNTIME_RANK = RANK == 0;
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize R = RUNTIME_RANK ? 3 : RANK;
        constexpr usize B = RUNTIME_RANK ? 1 : N - R;
        constexpr usize BR = B + R;
        const auto rank = RUNTIME_RANK ? output.shape().rank_checked(options.rank) : R;
        auto output_br = output.span().template as_nd<BR>();
        auto input_br = input.span().as_const().template as_nd<BR>();

        // Broadcast and reorder to rightmost.
        input_br = noa::broadcast(input_br, output_br.shape());
        const auto output_strides_r = output_br.strides().template pop_front<B>();
        const auto output_shape_r = output_br.shape().template pop_front<B>();
        const auto rightmost_order = output_strides_r.rightmost_order(output_shape_r);
        if (rightmost_order != Vec<isize, R>::arange()) {
            const auto order = (rightmost_order + B).push_front(Vec<isize, B>::arange());
            input_br = input_br.permute(order);
            output_br = output_br.permute(order);
        } // FIXME reorder B axes?

        // FIXME For width in CUDA, a single thread to go through the each line. Obviously, this leeds to poor performance.
        if (input.data() == output.data()) {
            details::cubic_bspline_prefilter_width_inplace(
                output_br, device,
                std::forward<Input>(input), std::forward<Output>(output)
            );
        } else {
            details::cubic_bspline_prefilter_width(
                input_br, output_br, device,
                std::forward<Input>(input), std::forward<Output>(output)
            );
        }
        if constexpr (BR >= 2) {
            if (rank >= 2) {
                // Move the height to the rightmost position.
                auto permutation = Vec<usize, BR>::arange().swap(BR - 2, BR - 1);
                auto output_br_wh = output_br.permute(permutation);
                details::cubic_bspline_prefilter_width_inplace(
                    output_br_wh, device,
                    std::forward<Input>(input), std::forward<Output>(output)
                );
            }
        }
        if constexpr (BR >= 3) {
            if (rank >= 3) {
                // Move the depth to the rightmost position.
                auto permutation = Vec<usize, BR>::arange().exclude_axis(BR - 3).push_back(BR - 3);
                auto output_br_hwd = output_br.permute(permutation);
                details::cubic_bspline_prefilter_width_inplace(
                    output_br_hwd, device,
                    std::forward<Input>(input), std::forward<Output>(output)
                );
            }
        }
    }
    template<typename Input, typename Output>
    void cubic_bspline_prefilter_1d(Input&& input, Output&& output, const CubicBSplinePrefilterOptions& options = {}) {
        cubic_bspline_prefilter<1>(std::forward<Input>(input), std::forward<Output>(output), options);
    }
    template<typename Input, typename Output>
    void cubic_bspline_prefilter_2d(Input&& input, Output&& output, const CubicBSplinePrefilterOptions& options = {}) {
        cubic_bspline_prefilter<2>(std::forward<Input>(input), std::forward<Output>(output), options);
    }
    template<typename Input, typename Output>
    void cubic_bspline_prefilter_3d(Input&& input, Output&& output, const CubicBSplinePrefilterOptions& options = {}) {
        cubic_bspline_prefilter<3>(std::forward<Input>(input), std::forward<Output>(output), options);
    }
}
