#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/Linspace.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/PolarTransform.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Texture.hpp"

namespace noa::geometry::guts {
    inline void set_frequency_range_to_default(
            const Shape4<i64>& shape,
            Vec2<f64>& frequency_range
    ) {
        if (all(frequency_range == Vec2<f64>{})) { // default value
            // Find smallest non-empty dimension.
            i64 min_size = shape[1] > 1 ? shape[1] : std::numeric_limits<i64>::max();
            min_size = shape[2] > 1 ? std::min(min_size, shape[2]) : min_size;
            min_size = shape[3] > 1 ? std::min(min_size, shape[3]) : min_size;

            // Get the max normalized frequency (if odd, it's not 0.5).
            frequency_range = {0, noa::fft::highest_normalized_frequency<f64>(min_size)};
        }
    }

    inline void set_polar_window_range_to_default(
            const Shape4<i64>& cartesian_shape,
            Vec2<f64>& frequency_range,
            Vec2<f64>& angle_range
    ) {
        set_frequency_range_to_default(cartesian_shape, frequency_range);
        if (all(angle_range == Vec2<f64>{}))
            angle_range = {0., Constant<f64>::PI};
    }

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

    template<typename Input, typename Output,
             typename InputValue = nt::value_type_t<Input>,
             typename OutputValue = nt::value_type_t<Output>>
    constexpr bool is_valid_polar_transform_v =
            nt::is_varray_of_mutable_v<Output> and
            (nt::is_varray_v<Input> or nt::is_texture_v<Input>) and
            (nt::are_real_v<InputValue, OutputValue> or
             nt::are_complex_v<InputValue, OutputValue> or
             (nt::is_complex_v<InputValue> and nt::is_real_v<OutputValue>));

    template<typename Input, typename Output>
    void polar_check_parameters(const Input& input, const Output& output) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape()[1] == 1 and output.shape()[1] == 1, "3D arrays are not supported");
        check(input.shape()[0] == 1 or input.shape()[0] == output.shape()[0],
              "The number of batches in the input array ({}) is not compatible with the number of "
              "batches in the output array ({})",
              input.shape()[0], output.shape()[0]);

        check(input.device() == output.device(),
              "The input and output arrays must be on the same device, "
              "but got input:{} and output:{}", input.device(), output.device());

        check(ni::are_elements_unique(output.strides(), output.shape()),
              "The elements in the output should not overlap in memory, "
              "otherwise a data-race might occur. Got output strides:{} and shape:{}",
              output.strides(), output.shape());

        if constexpr (nt::is_varray_v<Input>) {
            check(not ni::are_overlapped(input, output),
                  "Input and output arrays should not overlap");
        } else {
            check(input.device().is_gpu() or not ni::are_overlapped(input.view(), output),
                  "The input and output arrays should not overlap");
        }
    }

    template<typename Index, typename Input, typename Output, typename Options>
    void launch_cartesian2polar(
            const Input& cartesian,
            const Output& polar,
            const Vec2<f64>& cartesian_center,
            const Options& options
    ) {
        const auto device = polar.device();
        auto cartesian_strides = cartesian.strides().template as<Index>();
        auto cartesian_shape = cartesian.shape().template as<Index>();
        auto polar_strides = polar.strides().template as<Index>();
        auto polar_shape = polar.shape().template as<Index>();

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;

        // Cast to coordinate type.
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using input_accessor_t = AccessorRestrict<const nt::mutable_value_type_t<Input>, 3, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto input_accessor = input_accessor_t(cartesian.get(), cartesian_strides.filter(0, 2, 3));
        const auto output_accessor = output_accessor_t(polar.get(), polar_strides.filter(0, 2, 3));
        const auto cartesian_shape_2d = cartesian_shape.filter(2, 3);
        const auto polar_shape_2d = polar_shape.filter(0, 2, 3);

        switch (options.interpolation_mode) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::NEAREST, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, cartesian_shape_2d),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::LINEAR_FAST:
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::LINEAR, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, cartesian_shape_2d),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::COSINE_FAST:
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::COSINE, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, cartesian_shape_2d),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, cartesian_shape_2d),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::CUBIC_BSPLINE_FAST:
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC_BSPLINE, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, cartesian_shape_2d),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
        }
    }

    template<typename Index, typename Input, typename Output, typename Options>
    void launch_polar2cartesian(
            const Input& polar,
            const Output& cartesian,
            const Vec2<f64>& cartesian_center,
            const Options& options
    ) {
        const auto device = cartesian.device();
        auto cartesian_strides = cartesian.strides().template as<Index>();
        auto cartesian_shape = cartesian.shape().template as<Index>();
        auto polar_strides = polar.strides().template as<Index>();
        auto polar_shape = polar.shape().template as<Index>();

        // Broadcast the input to every output batch.
        if (polar_shape[0] == 1)
            polar_strides[0] = 0;

        // Cast to coordinate type.
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using input_accessor_t = AccessorRestrict<const nt::mutable_value_type_t<Input>, 3, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto input_accessor = input_accessor_t(polar.get(), polar_strides.filter(0, 2, 3));
        const auto output_accessor = output_accessor_t(cartesian.get(), cartesian_strides.filter(0, 2, 3));
        const auto polar_shape_2d = polar_shape.filter(2, 3);
        const auto cartesian_shape_2d = cartesian_shape.filter(0, 2, 3);

        switch (options.interpolation_mode) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::NEAREST, input_accessor_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, polar_shape_2d), polar_shape,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::LINEAR_FAST:
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::LINEAR, input_accessor_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, polar_shape_2d), polar_shape,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::COSINE_FAST:
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::COSINE, input_accessor_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, polar_shape_2d), polar_shape,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC, input_accessor_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, polar_shape_2d), polar_shape,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::CUBIC_BSPLINE_FAST:
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC_BSPLINE, input_accessor_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, polar_shape_2d), polar_shape,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
        }
    }

    template<typename Index, typename Value, typename Output, typename Options>
    void launch_cartesian2polar_texture(
            const Texture<Value>& cartesian,
            const Output& polar,
            const Vec2<f64>& cartesian_center,
            const Options& options
    ) {
#ifdef NOA_ENABLE_CUDA
        const auto device = polar.device();
        auto polar_strides = polar.strides().template as<Index>();
        auto polar_shape = polar.shape().template as<Index>();

        // Cast to coordinate type.
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto output_accessor = output_accessor_t(polar, polar_strides.filter(0, 2, 3));
        const auto polar_shape_2d = polar_shape.filter(0, 2, 3);

        using noa::cuda::geometry::Interpolator2d;
        cudaTextureObject_t cuda_texture = cartesian.cuda()->texture;

        switch (cartesian.interp_mode()) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Interp::NEAREST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Interp::LINEAR, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Interp::COSINE, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Interp::CUBIC, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::LINEAR_FAST: {
                using interpolator_t = Interpolator2d<Interp::LINEAR_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::COSINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::COSINE_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
            case Interp::CUBIC_BSPLINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Cartesian2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture),
                                     output_accessor, polar_shape, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             cartesian, polar);
            }
        }
#else
        panic("No GPU backend detected");
#endif
    }

    template<typename Index, typename Value, typename Output, typename Options>
    void launch_polar2cartesian_texture(
            const Texture<Value>& polar,
            const Output& cartesian,
            const Vec2<f64>& cartesian_center,
            const Options& options
    ) {
#ifdef NOA_ENABLE_CUDA
        const auto device = cartesian.device();
        auto cartesian_strides = cartesian.strides().template as<Index>();
        auto cartesian_shape = cartesian.shape().template as<Index>();

        // Cast to coordinate type.
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto output_accessor = output_accessor_t(cartesian, cartesian_strides.filter(0, 2, 3));
        const auto cartesian_shape_2d = cartesian_shape.filter(0, 2, 3);
        const auto polar_shape_2d = polar.shape().filter(2, 3).template as<Index>();

        using noa::cuda::geometry::Interpolator2d;
        cudaTextureObject_t cuda_texture = polar.cuda()->texture;

        switch (cartesian.interp_mode()) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Interp::NEAREST, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Interp::LINEAR, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Interp::COSINE, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Interp::CUBIC, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::LINEAR_FAST: {
                using interpolator_t = Interpolator2d<Interp::LINEAR_FAST, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::COSINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::COSINE_FAST, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
            case Interp::CUBIC_BSPLINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE_FAST, Value, false, true, coord_t>;
                return iwise(cartesian_shape_2d, device,
                             Polar2Cartesian<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), polar_shape_2d,
                                     output_accessor, cartesian_center,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             polar, cartesian);
            }
        }
#else
        panic("No GPU backend detected");
#endif
    }
}

namespace noa::geometry {
    struct PolarTransformOptions {
        /// Rho radius [start,end] range of the bounding circle to transform, in pixels.
        /// Rho maps to the width dimension of the polar array.
        /// Defaults to the largest in-bound circle.
        Vec2<f64> rho_range{};

        /// Whether the rho_range's end should be included in the range.
        /// The computed linspace range is linspace(rho_range[0], rho_range[1], polar_width, rho_endpoint).
        bool rho_endpoint{};

        /// Phi angle [start,end) range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array.
        /// Defaults to the entire circle, i.e. [0, 2pi).
        Vec2<f64> phi_range{};

        /// Whether the phi_range's end should be included in the range.
        /// The computed linspace range is linspace(phi_range[0], phi_range[1], polar_height, phi_endpoint).
        bool phi_endpoint{};

        /// Interpolation method used to interpolate the values onto the new grid.
        /// Out-of-bounds elements are set to zero.
        /// This is unused if a texture is passed to the function.
        Interp interpolation_mode{Interp::LINEAR};
    };

    /// Transforms 2d array(s) from cartesian to polar coordinates.
    /// \param[in] cartesian        Input 2d cartesian array|texture to interpolate onto the new coordinate system.
    /// \param[out] polar           Transformed 2d array on the polar grid.
    /// \param cartesian_center     HW transformation center.
    /// \param options              Transformation options.
    template<typename Input, typename Output>
    requires guts::is_valid_polar_transform_v<Input, Output>
    void cartesian2polar(
            const Input& cartesian,
            const Output& polar,
            const Vec2<f64>& cartesian_center,
            const PolarTransformOptions& options = {}
    ) {
        guts::polar_check_parameters(cartesian, polar);
        guts::set_polar_window_range_to_default(
                cartesian.shape(), cartesian_center,
                options.rho_range, options.phi_range);

        if (polar.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(cartesian, cartesian.shape()) and
                  ng::is_accessor_access_safe<i32>(polar, polar.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            if constexpr (nt::is_texture_v<Input>)
                guts::launch_cartesian2polar_texture<i32>(cartesian, polar, cartesian_center, options);
            else
                guts::launch_cartesian2polar<i32>(cartesian, polar, cartesian_center, options);
        } else {
            guts::launch_cartesian2polar<i64>(cartesian, polar, cartesian_center, options);
        }
    }

    /// Transforms 2d array(s) from polar to cartesian coordinates.
    /// \param[in] polar            Input 2d polar array|texture to interpolate onto the new coordinate system.
    /// \param[out] cartesian       Transformed 2d array on the cartesian grid.
    /// \param cartesian_center     HW transformation center.
    /// \param options              Transformation options.
    template<typename Input, typename Output>
    requires guts::is_valid_polar_transform_v<Input, Output>
    void polar2cartesian(
            const Input& polar,
            const Output& cartesian,
            const Vec2<f64>& cartesian_center,
            PolarTransformOptions options = {}
    ) {
        guts::polar_check_parameters(polar, cartesian);
        guts::set_polar_window_range_to_default(
                cartesian.shape(), cartesian_center,
                options.rho_range, options.phi_range);

        if (cartesian.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(polar, polar.shape()) and
                  ng::is_accessor_access_safe<i32>(cartesian, cartesian.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            if constexpr (nt::is_texture_v<Input>)
                guts::launch_polar2cartesian_texture<i32>(polar, cartesian, cartesian_center, options);
            else
                guts::launch_polar2cartesian<i32>(polar, cartesian, cartesian_center, options);
        } else {
            guts::launch_polar2cartesian<i64>(polar, cartesian, cartesian_center, options);
        }
    }
}
