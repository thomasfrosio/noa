#pragma once

#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/xform/core/Polar.hpp"
#include "noa/xform/Traits.hpp"
#include "noa/xform/Utils.hpp"

namespace noa::xform::details {
    /// iwise operator to compute 2d cartesian->polar transformation(s).
    template<usize B,
             nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_nd<2> Interpolator,
             nt::readable_nd<B + 2> Input,
             nt::writable_nd<B + 2> Output>
    class Polar2Cartesian {
    public:
        using index_type = Index;
        using input_type = Input;
        using interpolator_type = Interpolator;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec<coord_type, 2>;
        using shape2_type = Shape<index_type, 2>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

    public:
        Polar2Cartesian(
            const input_type& polar,
            const interpolator_type& interpolator,
            const shape2_type& polar_shape,
            const output_type& cartesian,
            const coord2_type& cartesian_center,
            const Linspace<coord_type>& radius_range,
            const Linspace<coord_type>& angle_range
        ) :
            m_polar(polar),
            m_interpolator(interpolator),
            m_cartesian(cartesian),
            m_center(cartesian_center),
            m_start_angle(angle_range.start),
            m_start_radius(radius_range.start)
        {
            NOA_ASSERT(radius_range[1] - radius_range[0] >= 0);
            m_step_angle = angle_range.for_size(polar_shape[0]).step;
            m_step_radius = radius_range.for_size(polar_shape[1]).step;
        }

        NOA_HD constexpr void operator()(const Vec<index_type, B + 2>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            auto cartesian_coordinate = indices.template as<coord_type>();
            cartesian_coordinate -= m_center;

            const coord_type phi = cartesian2phi(cartesian_coordinate);
            const coord_type rho = cartesian2rho(cartesian_coordinate);
            const coord2_type polar_coordinate{
                (phi - m_start_angle) / m_step_angle,
                (rho - m_start_radius) / m_step_radius
            };

            auto value = m_interpolator.get(m_polar[batches], polar_coordinate);
            m_cartesian[batched_indices] = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_polar;
        interpolator_type m_interpolator;
        output_type m_cartesian;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };

    /// 3d iwise operator to compute 2d polar->cartesian transformation(s).
    template<usize B,
             nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_nd<2> Interpolator,
             nt::readable_nd<B + 2> Input,
             nt::writable_nd<B + 2> Output>
    class Cartesian2Polar {
    public:
        using index_type = Index;
        using input_type = Input;
        using interpolator_type = Interpolator;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec<coord_type, 2>;
        using shape2_type = Shape<index_type, 2>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

    public:
        Cartesian2Polar(
            const input_type& cartesian,
            const interpolator_type& interpolator,
            const output_type& polar,
            const shape2_type& polar_shape,
            const coord2_type& cartesian_center,
            const Linspace<coord_type>& radius_range,
            const Linspace<coord_type>& angle_range
        ) :
            m_cartesian(cartesian),
            m_interpolator(interpolator),
            m_polar(polar),
            m_center(cartesian_center),
            m_start_angle(angle_range.start),
            m_start_radius(radius_range.start)
        {
            NOA_ASSERT(radius_range.stop - radius_range.start >= 0);

            // We could use polar2phi() and polar2rho() in the loop, but instead, precompute the step here.
            m_step_angle = angle_range.for_size(polar_shape[0]).step;
            m_step_radius = radius_range.for_size(polar_shape[1]).step;
        }

        NOA_HD constexpr void operator()(const Vec<index_type, B + 2>& batched_indices) const {
            const auto& [batches, indices] = batched_indices.template split<B>();
            const auto polar_coordinate = indices.template as<coord_type>();
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_radius + m_start_radius;

            coord2_type cartesian_coordinate = sincos(phi);
            cartesian_coordinate *= rho;
            cartesian_coordinate += m_center;

            auto value = m_interpolator.get(m_cartesian[batches], cartesian_coordinate);
            m_polar[batched_indices] = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_cartesian;
        interpolator_type m_interpolator;
        output_type m_polar;
        coord2_type m_center;
        coord_type m_step_angle;
        coord_type m_step_radius;
        coord_type m_start_angle;
        coord_type m_start_radius;
    };

    template<usize N>
    void set_polar_window_range_to_default(
        const Shape<isize, N>& cartesian_shape,
        const Vec<f64, 2>& cartesian_center,
        Linspace<f64>& radius_range,
        Linspace<f64>& angle_range
    ) {
        if (radius_range.start == 0 and radius_range.stop == 0)
            radius_range.stop = min(cartesian_shape.vec.filter(N - 2, N - 1).template as<f64>() - cartesian_center);
        if (angle_range.start == 0 and angle_range.stop == 0)
            angle_range.stop = 2 * Constant<f64>::PI;
    }

    template<typename Input, typename Output>
    void polar_check_parameters(const Input& input, const Output& output) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");

        // Check batch axes are compatible.
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - 2;
        auto input_shape_b = input.shape().template pop_back<2>();
        auto output_shape_b = output.shape().template pop_back<2>();
        if constexpr (B >= 1) {
            for (usize i{}; i < B; ++i)
                input_shape_b[i] = input_shape_b[i] == 1 ? output_shape_b[i] : input_shape_b[i];
            check(input_shape_b == output_shape_b,
                  "The batch axes are not compatible, input:batches={}, output:batches={}",
                  input_shape_b, output_shape_b);
        }

        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, but got input:device={} and output:device={}",
              input.device(), output.device());
        check(nd::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, otherwise a data-race might occur. Got output output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::array<Input>) {
            check(not are_overlapped(input, output),
                  "Input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not are_overlapped(input.cpu(), output),
                  "The input and output arrays should not overlap");
            check(input.border() == Border::ZERO,
                  "The input border mode should be {}, but got {}",
                  Border::ZERO, input.border());
        }
    }

    template<bool CARTESIAN_TO_POLAR, bool IS_GPU = false,
             typename Index, typename Input, typename Output, typename Options>
    void launch_cartesian_polar(
        Input&& input,
        Output&& output,
        const Vec<f64, 2>& cartesian_center,
        const Options& options
    ) {
        constexpr usize N = nt::array_size_v<Output>;
        constexpr usize B = N - 2;
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, Index>;
        const auto output_span = output.span().template as_index<Index>();
        const auto output_accessor = output_accessor_t(output_span.get(), output_span.strides());

        auto launch_iwise = [&](auto interp) {
            auto result = prepare_interpolation_inputs<2, interp(), Border::ZERO, IS_GPU, Index, coord_t, false>(input);
            using interpolator_t = decltype(result)::interpolator_type;
            using accessor_t = decltype(result)::accessor_type;

            auto op = [&]{
                if constexpr (CARTESIAN_TO_POLAR) {
                    const auto polar_shape_r = output_span.shape().template pop_front<B>();
                    return Cartesian2Polar<B, Index, coord_t, interpolator_t, accessor_t, output_accessor_t>(
                        result.accessor, result.interpolator,
                        output_accessor, polar_shape_r, cartesian_center.as<coord_t>(),
                        rho_range, phi_range);
                } else {
                    const auto polar_shape_r = input.shape().template pop_front<B>().template as<Index>();
                    return Polar2Cartesian<B, Index, coord_t, interpolator_t, accessor_t, output_accessor_t>(
                        result.accessor, result.interpolator, polar_shape_r,
                        output_accessor, cartesian_center.as<coord_t>(),
                        rho_range, phi_range);
                }
            }();
            return iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_span.shape(), output.device(), op,
               std::forward<Input>(input), std::forward<Output>(output));
        };

        Interp interp = options.interp;
        if constexpr (nt::texture_decay<Input>)
            interp = input.interp();
        switch (interp) {
            case Interp::NEAREST:            return launch_iwise(WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_iwise(WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_iwise(WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_iwise(WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_iwise(WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }

    template<typename Input, typename Output>
    concept polar_transformable =
        nt::readable_array_or_texture_rd_decay<Input, 2> and nt::writable_array_decay<Output> and
        nt::array_size_v<Input> == nt::array_size_v<Output> and nt::array_size_v<Output> >= 2 and
        nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>;
}

namespace noa::xform {
    struct PolarTransformOptions {
        /// Rho radius range of the bounding circle to transform, in pixels.
        /// Rho maps to the width dimension of the polar array.
        /// Defaults to the largest in-bound circle.
        /// The computed linspace range is rho_range.for_size(polar_width).
        Linspace<f64> rho_range{0, 0, false};

        /// Phi angle range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array.
        /// Defaults to the entire circle, i.e. [0, 2pi).
        /// The computed linspace range is phi_range.for_size(polar_height).
        Linspace<f64> phi_range{0, 0, false};

        /// Interpolation method used to interpolate the values onto the new grid.
        /// Out-of-bounds elements are set to zero.
        /// This is ignored if the input is a texture.
        Interp interp{Interp::LINEAR};
    };

    /// Transforms 2D array(s) from cartesian to polar coordinates.
    /// \param[in] cartesian:
    ///     ((Bi..,)Hi,Wi) Input 2D cartesian array|texture to interpolate onto the new coordinate system.
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] polar:
    ///     ((Bo..,)Ho,Wo) Output 2D transformed array(s) on the polar grid.
    /// \param cartesian_center:
    ///     HW transformation center.
    /// \param options:
    ///     Transformation options.
    template<typename Input, typename Output>
        requires details::polar_transformable<Input, Output>
    void cartesian2polar(
        Input&& cartesian,
        Output&& polar,
        const Vec<f64, 2>& cartesian_center,
        PolarTransformOptions options = {}
    ) {
        details::polar_check_parameters(cartesian, polar);
        details::set_polar_window_range_to_default(
            cartesian.shape(), cartesian_center,
            options.rho_range, options.phi_range);

        if (polar.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(cartesian, cartesian.shape()) and
                      nd::is_accessor_access_safe<i32>(polar, polar.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_cartesian_polar<true, true, i32>(
                    std::forward<Input>(cartesian),
                    std::forward<Output>(polar),
                    cartesian_center, options);
            }
            #else
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            details::launch_cartesian_polar<true, false, isize>(
                std::forward<Input>(cartesian),
                std::forward<Output>(polar),
                cartesian_center, options);
        }
    }

    /// Transforms 2D array(s) from polar to cartesian coordinates.
    /// \param[in] polar:
    ///     ((Bi..,)Hi,Wi) Input 2D polar array|texture to interpolate onto the new coordinate system.
    ///     The batch axes are broadcast to the output batch axes.
    /// \param[out] cartesian:
    ///     ((Bo..,)Ho,Wo) Output 2D transformed array(s) on the cartesian grid.
    /// \param cartesian_center:
    ///     HW transformation center.
    /// \param options:
    ///     Transformation options.
    template<typename Input, typename Output>
        requires details::polar_transformable<Input, Output>
    void polar2cartesian(
        Input&& polar,
        Output&& cartesian,
        const Vec<f64, 2>& cartesian_center,
        PolarTransformOptions options = {}
    ) {
        details::polar_check_parameters(polar, cartesian);
        details::set_polar_window_range_to_default(
            cartesian.shape(), cartesian_center,
            options.rho_range, options.phi_range);

        if (cartesian.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(nd::is_accessor_access_safe<i32>(cartesian, cartesian.shape()) and
                      nd::is_accessor_access_safe<i32>(polar, polar.shape()),
                      "isize indexing not instantiated for GPU devices");
                details::launch_cartesian_polar<false, true, i32>(
                    std::forward<Input>(cartesian),
                    std::forward<Output>(polar),
                    cartesian_center, options);
            }
            #else
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            details::launch_cartesian_polar<false, false, isize>(
                std::forward<Input>(polar),
                std::forward<Output>(cartesian),
                cartesian_center, options);
        }
    }
}
