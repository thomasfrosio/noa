#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/geometry/PolarTransform.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::geometry::guts {
    inline void set_polar_window_range_to_default(
        const Shape4<i64>& cartesian_shape,
        const Vec2<f64>& cartesian_center,
        Vec2<f64>& radius_range,
        Vec2<f64>& angle_range
    ) {
        if (all(radius_range == Vec2<f64>{}))
            radius_range = {0., min(cartesian_shape.filter(2, 3).vec.as<f64>() - cartesian_center)};
        if (all(angle_range == Vec2<f64>{}))
            angle_range = {0., 2 * Constant<f64>::PI};
    }

    inline void set_frequency_range_to_default(
        const Shape4<i64>& shape,
        Vec2<f64>& fftfreq_range
    ) {
        if (all(fftfreq_range == Vec2<f64>{})) {
            // default value
            // Find smallest non-empty dimension.
            i64 min_size = shape[1] > 1 ? shape[1] : std::numeric_limits<i64>::max();
            min_size = shape[2] > 1 ? std::min(min_size, shape[2]) : min_size;
            min_size = shape[3] > 1 ? std::min(min_size, shape[3]) : min_size;

            // Get the max normalized frequency (if odd, it's not 0.5).
            fftfreq_range = {0, noa::fft::highest_normalized_frequency<f64>(min_size)};
        }
    }

    inline void set_polar_window_range_to_default(
        const Shape4<i64>& cartesian_shape,
        Vec2<f64>& fftfreq_range,
        Vec2<f64>& angle_range
    ) {
        set_frequency_range_to_default(cartesian_shape, fftfreq_range);
        if (all(angle_range == Vec2<f64>{}))
            angle_range = {0., Constant<f64>::PI};
    }

    template<typename Input, typename Output>
    void polar_check_parameters(const Input& input, const Output& output) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape()[1] == 1 and output.shape()[1] == 1, "3d arrays are not supported");
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The input batch size ({}) is not compatible with the output batch size ({})",
              input.shape()[0], output.shape()[0]);

        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={} and output:device={}", input.device(), output.device());

        check(ni::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, "
              "otherwise a data-race might occur. Got output output:strides={} and output:shape={}",
              output.strides(), output.shape());

        if constexpr (nt::varray<Input>) {
            check(not ni::are_overlapped(input, output),
                  "Input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
            check(input.border() == Border::ZERO,
                  "The input border mode should be {}, but got {}", Border::ZERO, input.border());
        }
    }

    template<bool CARTESIAN_TO_POLAR, bool IS_GPU = false,
             typename Index, typename Input, typename Output, typename Options>
    void launch_cartesian_polar(
        Input&& input,
        Output&& output,
        const Vec2<f64>& cartesian_center,
        const Options& options
    ) {
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, 2, 3).template as<Index>());
        const auto output_shape = output.shape().filter(0, 2, 3).template as<Index>();

        auto launch_iwise = [&](auto interp) {
            auto interpolator = ng::to_interpolator<2, interp(), Border::ZERO, Index, coord_t, IS_GPU>(input);
            auto op = [&]{
                if constexpr (CARTESIAN_TO_POLAR) {
                    return Cartesian2Polar<Index, coord_t, decltype(interpolator), output_accessor_t>(
                        interpolator,
                        output_accessor, output_shape.pop_front(), cartesian_center.as<coord_t>(),
                        rho_range, options.rho_endpoint,
                        phi_range, options.phi_endpoint);
                } else {
                    return Polar2Cartesian<Index, coord_t, decltype(interpolator), output_accessor_t>(
                        interpolator, input.shape().filter(2, 3).template as<Index>(),
                        output_accessor, cartesian_center.as<coord_t>(),
                        rho_range, options.rho_endpoint,
                        phi_range, options.phi_endpoint);
                }
            }();
            return iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(output_shape, output.device(), op, std::forward<Input>(input), std::forward<Output>(output));
        };

        Interp interp = options.interp;
        if constexpr (nt::texture_decay<Input>)
            interp = input.interp();
        switch (interp) {
            case Interp::NEAREST:            return launch_iwise(ng::WrapInterp<Interp::NEAREST>{});
            case Interp::NEAREST_FAST:       return launch_iwise(ng::WrapInterp<Interp::NEAREST_FAST>{});
            case Interp::LINEAR:             return launch_iwise(ng::WrapInterp<Interp::LINEAR>{});
            case Interp::LINEAR_FAST:        return launch_iwise(ng::WrapInterp<Interp::LINEAR_FAST>{});
            case Interp::CUBIC:              return launch_iwise(ng::WrapInterp<Interp::CUBIC>{});
            case Interp::CUBIC_FAST:         return launch_iwise(ng::WrapInterp<Interp::CUBIC_FAST>{});
            case Interp::CUBIC_BSPLINE:      return launch_iwise(ng::WrapInterp<Interp::CUBIC_BSPLINE>{});
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise(ng::WrapInterp<Interp::CUBIC_BSPLINE_FAST>{});
            case Interp::LANCZOS4:           return launch_iwise(ng::WrapInterp<Interp::LANCZOS4>{});
            case Interp::LANCZOS6:           return launch_iwise(ng::WrapInterp<Interp::LANCZOS6>{});
            case Interp::LANCZOS8:           return launch_iwise(ng::WrapInterp<Interp::LANCZOS8>{});
            case Interp::LANCZOS4_FAST:      return launch_iwise(ng::WrapInterp<Interp::LANCZOS4_FAST>{});
            case Interp::LANCZOS6_FAST:      return launch_iwise(ng::WrapInterp<Interp::LANCZOS6_FAST>{});
            case Interp::LANCZOS8_FAST:      return launch_iwise(ng::WrapInterp<Interp::LANCZOS8_FAST>{});
        }
    }
}

namespace noa::geometry {
    struct PolarTransformOptions {
        /// Rho radius [start,end] range of the bounding circle to transform, in pixels.
        /// Rho maps to the width dimension of the polar array.
        /// Defaults to the largest in-bound circle.
        Vec2<f64> rho_range{};

        /// Whether the rho_range's end should be included in the range.
        /// The computed linspace range is Linspace{rho_range[0], rho_range[1], rho_endpoint}.for_size(polar_width).
        bool rho_endpoint{};

        /// Phi angle [start,end) range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array.
        /// Defaults to the entire circle, i.e. [0, 2pi).
        Vec2<f64> phi_range{};

        /// Whether the phi_range's end should be included in the range.
        /// The computed linspace range is Linspace{phi_range[0], phi_range[1], phi_endpoint}.for_size(polar_height).
        bool phi_endpoint{};

        /// Interpolation method used to interpolate the values onto the new grid.
        /// Out-of-bounds elements are set to zero.
        /// This is ignored if the input is a texture.
        Interp interp{Interp::LINEAR};
    };

    /// Transforms 2d array(s) from cartesian to polar coordinates.
    /// \param[in] cartesian        Input 2d cartesian array|texture to interpolate onto the new coordinate system.
    /// \param[out] polar           Transformed 2d array on the polar grid.
    /// \param cartesian_center     HW transformation center.
    /// \param options              Transformation options.
    template<nt::varray_or_texture_decay Input,
             nt::writable_varray_decay Output>
    requires nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>
    void cartesian2polar(
        Input&& cartesian,
        Output&& polar,
        const Vec2<f64>& cartesian_center,
        PolarTransformOptions options = {}
    ) {
        guts::polar_check_parameters(cartesian, polar);
        guts::set_polar_window_range_to_default(
            cartesian.shape(), cartesian_center,
            options.rho_range, options.phi_range);

        if (polar.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(cartesian, cartesian.shape()) and
                      ng::is_accessor_access_safe<i32>(polar, polar.shape()),
                      "i64 indexing not instantiated for GPU devices");
                guts::launch_cartesian_polar<true, true, i32>(
                    std::forward<Input>(cartesian),
                    std::forward<Output>(polar),
                    cartesian_center, options);
            }
            #else
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            guts::launch_cartesian_polar<true, false, i64>(
                std::forward<Input>(cartesian),
                std::forward<Output>(polar),
                cartesian_center, options);
        }
    }

    /// Transforms 2d array(s) from polar to cartesian coordinates.
    /// \param[in] polar            Input 2d polar array|texture to interpolate onto the new coordinate system.
    /// \param[out] cartesian       Transformed 2d array on the cartesian grid.
    /// \param cartesian_center     HW transformation center.
    /// \param options              Transformation options.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    requires nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>
    void polar2cartesian(
        Input&& polar,
        Output&& cartesian,
        const Vec2<f64>& cartesian_center,
        PolarTransformOptions options = {}
    ) {
        guts::polar_check_parameters(polar, cartesian);
        guts::set_polar_window_range_to_default(
            cartesian.shape(), cartesian_center,
            options.rho_range, options.phi_range);

        if (cartesian.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(cartesian, cartesian.shape()) and
                      ng::is_accessor_access_safe<i32>(polar, polar.shape()),
                      "i64 indexing not instantiated for GPU devices");
                guts::launch_cartesian_polar<false, true, i32>(
                    std::forward<Input>(cartesian),
                    std::forward<Output>(polar),
                    cartesian_center, options);
            }
            #else
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            guts::launch_cartesian_polar<false, false, i64>(
                std::forward<Input>(polar),
                std::forward<Output>(cartesian),
                cartesian_center, options);
        }
    }
}
