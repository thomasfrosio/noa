#pragma once

#include "noa/runtime/Iwise.hpp"
#include "noa/xform/core/Interp.hpp"

namespace noa::xform::details {
    template<usize B, usize R,
             nt::real Coord,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output>
    class FourierInterpolationCorrection {
        using output_value_type = nt::value_type_t<Output>;
        using input_value_type = nt::value_type_t<Input>;
        static_assert(nt::real<input_value_type, output_value_type>);

    public:
        template<typename T>
        constexpr FourierInterpolationCorrection(
            const Input& input,
            const Output& output,
            const Shape<T, R>& shape,
            const Vec<Coord, R>& center,
            Interp interp
        ) :
            m_input(input),
            m_output(output),
            m_shape(Vec<Coord, R>::from_vec(shape.vec)),
            m_center(center),
            m_interp(interp) {}

        template<nt::integer T>
        NOA_HD void operator()(const Vec<T, B + R>& batched_indices) const noexcept {
            auto normalized_coordinates = batched_indices.template pop_front<B>().template as<Coord>();
            normalized_coordinates -= m_center;
            normalized_coordinates /= m_shape;

            constexpr Coord PI = Constant<Coord>::PI;
            const auto response = static_cast<input_value_type>([&] {
                const auto pi_coords = PI * normalized_coordinates;
                const auto sinc = noa::product(noa::sinc(pi_coords));
                switch (m_interp) {
                    case Interp::NEAREST:
                        return sinc;
                    case Interp::LINEAR:
                        return sinc * sinc;
                    case Interp::CUBIC_BSPLINE: {
                        auto sinc2 = sinc * sinc;
                        return sinc2 * sinc2;
                    }
                    default:
                        return input_value_type{1};
                }
            }());

            m_output[batched_indices] = static_cast<output_value_type>(m_input[batched_indices] / response);
        }

    private:
        Input m_input;
        Output m_output;
        Vec<Coord, R> m_shape;
        Vec<Coord, R> m_center;
        Interp m_interp;
    };
}

namespace noa::xform {
    template<usize R>
    struct FourierInterpolationCorrectionOption {
        /// Rank of the inverse Fourier transforms.
        /// See Shape::rank_checked for more details.
        usize rank{};

        /// Method used for the Fourier interpolation.
        /// Interp::NEAREST, Interp::LINEAR, and Interp::CUBIC_BSPLINE are supported.
        Interp interp;

        /// Center of the correction function.
        /// This should match the real-space geometric center around which the Fourier interpolation happened.
        /// For each axis, default to the center of the input/output (N//2, integer division).
        Vec<f64, R> center{Vec<f64, R>::from_value(std::numeric_limits<f64>::max())};

        /// Logical shape of the Fourier transform where the interpolation happened.
        /// For each axis, default to the input/output shape.
        Shape<isize, R> shape{};
    };

    /// Corrects for the interpolation kernel applied to Fourier transforms.
    /// \details
    ///     When interpolating Fourier transforms, we effectively convolve the input Fourier components with
    ///     an interpolation kernel. As such, the resulting iFT of the interpolated output is the product of the
    ///     final wanted output and the iFT of the interpolation kernel. This function corrects for the effect of
    ///     the interpolation kernel in real-space. This correction can be applied on the interpolated output,
    ///     as well as the input about to be Fourier transformed and interpolated.
    /// \tparam RANK:
    ///     Rank of the arrays.
    ///     If zero, use the runtime options.rank.
    /// \param[in] input:
    ///     Inverse Fourier transform(s).
    ///     Automatically broadcast to the output shape.
    /// \param[out] output:
    ///     Corrected output. Can be equal to the input.
    /// \param[in] options:
    ///     Correction options.
    template<usize RANK = 0,
             nt::array_decay_of_almost_any<f32, f64> Input,
             nt::array_decay_of_any<f32, f64> Output>
        requires (nt::array_decay_with_same_nd<Input, Output> and nt::array_size_v<Output> >= RANK)
    void fourier_interpolation_correction(
        Input&& input,
        Output&& output,
        const FourierInterpolationCorrectionOption<RANK>& options
    ) {
        check(
            options.interp.is_almost_any(Interp::NEAREST, Interp::LINEAR, Interp::CUBIC_BSPLINE),
            "{} is currently not supported", options.interp
        );

        constexpr usize N = nt::array_size_v<Input>;
        constexpr bool RUNTIME_RANK = RANK == 0;
        constexpr usize R = RUNTIME_RANK ? 3 : RANK;
        constexpr usize B = RUNTIME_RANK ? 1 : N - RANK;
        constexpr usize BR = B + R;
        const usize rank = RUNTIME_RANK ? output.shape().rank_checked(options.rank) : R;

        const auto input_br = input.span().reshape(input.shape().template as_nd<BR>().ranked(rank));
        const auto output_br = output.span().reshape(output.shape().template as_nd<BR>().ranked(rank));

        auto input_strides_br = input_br.strides();
        check(noa::broadcast(input_br.shape(), input_strides_br, output_br.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input_br.shape(), output_br.shape());

        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = std::conditional_t<nt::any_of<f64, input_value_t, output_value_t>, f64, f32>;
        auto input_accessor = Accessor<const input_value_t, BR, isize>(input_br.get(), input_strides_br);
        auto output_accessor = Accessor<output_value_t, BR, isize>(output_br.get(), output_br.strides());
        using input_accessor_t = decltype(input_accessor);
        using output_accessor_t = decltype(output_accessor);

        auto shape_r = output_br.shape().template pop_front<B>();

        // Get the center of the geometric transform.
        // If not specified, use the default center of N//2 (integer division)
        auto center_r = options.center.template as<coord_t>();
        for (usize i{}; i < R; ++i)
            if (center_r[i] == std::numeric_limits<coord_t>::max())
                center_r[i] = static_cast<coord_t>(shape_r[i] / 2);

        // Get the shape of the Fourier transform.
        // If not specified, use the shape of the input/output.
        for (usize i{}; i < R; ++i)
            if (options.shape[i] > 0)
                shape_r[i] = options.shape[i];

        const auto op = details::FourierInterpolationCorrection<B, R, coord_t, input_accessor_t, output_accessor_t>(
            input_accessor, output_accessor, shape_r, center_r, options.interp.erase_fast());
        iwise(output_br.shape(), output.device(), op, {}, NOA_FWD(input), NOA_FWD(output));
    }

    template<typename Input, typename Output>
    void fourier_interpolation_correction_2d(
        Input&& input,
        Output&& output,
        const FourierInterpolationCorrectionOption<2>& options
    ) {
        fourier_interpolation_correction<2>(NOA_FWD(input), NOA_FWD(output), options);
    }
    template<typename Input, typename Output>
    void fourier_interpolation_correction_3d(
        Input&& input,
        Output&& output,
        const FourierInterpolationCorrectionOption<3>& options
    ) {
        fourier_interpolation_correction<3>(NOA_FWD(input), NOA_FWD(output), options);
    }
}
