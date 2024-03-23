#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/Linspace.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/geometry/FourierPolar.hpp"

#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Interpolator.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/Texture.hpp"
#include "noa/unified/geometry/PolarTransform.hpp"

namespace noa::geometry::guts {
    template<typename Index, typename Input, typename Output, typename Options>
    void launch_spectrum2polar(
            const Input& spectrum,
            Shape4<Index> spectrum_shape,
            const Output& polar,
            const Options& options
    ) {
        const auto device = polar.device();
        auto spectrum_strides = spectrum.strides().template as<Index>();
        auto polar_strides = polar.strides().template as<Index>();
        auto polar_shape = polar.shape().template as<Index>();

        // Broadcast the input to every output batch.
        if (spectrum_shape[0] == 1)
            spectrum_strides[0] = 0;

        // Cast to coordinate type.
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using input_accessor_t = AccessorRestrict<const nt::mutable_value_type_t<Input>, 3, Index>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto input_accessor = input_accessor_t(spectrum.get(), spectrum_strides.filter(0, 2, 3));
        const auto output_accessor = output_accessor_t(polar.get(), polar_strides.filter(0, 2, 3));
        const auto spectrum_shape_2d = spectrum_shape.filter(2, 3);
        const auto polar_shape_2d = polar_shape.filter(0, 2, 3);

        switch (options.interpolation_mode) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::NEAREST, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, spectrum_shape_2d), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::LINEAR_FAST:
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::LINEAR, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, spectrum_shape_2d), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::COSINE_FAST:
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::COSINE, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, spectrum_shape_2d), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, spectrum_shape_2d), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::CUBIC_BSPLINE_FAST:
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Border::ZERO, Interp::CUBIC_BSPLINE, input_accessor_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(input_accessor, spectrum_shape_2d), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
        }
    }

    template<typename Index, typename Value, typename Output, typename Options>
    void launch_spectrum2polar_texture(
            const Texture<Value>& spectrum,
            const Shape4<i64>& spectrum_shape,
            const Output& polar,
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
        cudaTextureObject_t cuda_texture = spectrum.cuda()->texture;

        switch (spectrum.interp_mode()) {
            case Interp::NEAREST: {
                using interpolator_t = Interpolator2d<Interp::NEAREST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::LINEAR: {
                using interpolator_t = Interpolator2d<Interp::LINEAR, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::COSINE: {
                using interpolator_t = Interpolator2d<Interp::COSINE, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::CUBIC: {
                using interpolator_t = Interpolator2d<Interp::CUBIC, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::CUBIC_BSPLINE: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::LINEAR_FAST: {
                using interpolator_t = Interpolator2d<Interp::LINEAR_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::COSINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::COSINE_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
            case Interp::CUBIC_BSPLINE_FAST: {
                using interpolator_t = Interpolator2d<Interp::CUBIC_BSPLINE_FAST, Value, false, true, coord_t>;
                return iwise(polar_shape_2d, device,
                             Spectrum2Polar<Index, coord_t, interpolator_t, output_accessor_t>(
                                     interpolator_t(cuda_texture), spectrum_shape,
                                     output_accessor, polar_shape,
                                     rho_range, options.rho_endpoint,
                                     phi_range, options.phi_endpoint),
                             spectrum, polar);
            }
        }
#else
        panic("No GPU backend detected");
#endif
    }
}

namespace noa::geometry {
    struct PolarTransformSpectrumOptions {
        /// Rho frequency [start,end] range of the bounding shells to transform, in cycle/pixels.
        /// Rho maps to the width dimension of the polar array.
        /// Defaults to the [0, v], where v is the highest normalized frequency of min(height,width).
        Vec2<f64> rho_range{};

        /// Whether the rho_range's end should be included in the range.
        /// The computed linspace range is linspace(rho_range[0], rho_range[1], polar_width, rho_endpoint).
        bool rho_endpoint{true};

        /// Phi angle [start,end) range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array.
        /// While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
        /// this range can include the entire unit circle, e.g. [-pi, pi]. Defaults to [0, pi).
        Vec2<f64> phi_range{};

        /// Whether the phi_range's end should be included in the range.
        /// The computed linspace range is linspace(phi_range[0], phi_range[1], polar_height, phi_endpoint).
        bool phi_endpoint{};

        /// Interpolation method used to interpolate the values onto the new grid.
        /// Cubic interpolations are not supported. Out-of-bounds elements are set to zero.
        /// This is unused if a texture is passed to the function.
        Interp interpolation_mode{Interp::LINEAR};
    };

// TODO Add polar2spectrum

    /// Transforms 2d dft(s) to polar coordinates.
    /// \tparam REMAP           Only HC2FC is currently supported. The output is denoted as "FC" (full-centered)
    ///                         to emphasize that it has a full shape (equals to \p polar_shape) and can map the
    ///                         entire angular range (e.g. 0 to 2PI).
    /// \param[in] spectrum     Centered 2d rfft to interpolate onto the new coordinate system.
    /// \param spectrum_shape   BDHW logical shape of \p spectrum.
    /// \param[out] polar       Transformed 2d array on the polar grid.
    ///                         If real and \p spectrum is complex, the power spectrum is computed.
    /// \param options          Transformation options.
    template<noa::fft::RemapInterface REMAP, typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64> &&
              nt::is_varray_of_any_v<Output, f32, f64, c32, c64> &&
              (nt::are_almost_same_value_type_v<Input, Output> ||
               nt::are_almost_same_value_type_v<Input, nt::value_type_t<Output>>) &&
              REMAP.remap == noa::fft::Remap::HC2FC)
    void spectrum2polar(
            const Input& spectrum,
            const Shape4<i64>& spectrum_shape,
            const Output& polar,
            PolarTransformSpectrumOptions options
    ) {
        guts::polar_check_parameters(spectrum, spectrum_shape, polar);
        guts::set_polar_window_range_to_default(spectrum_shape, options.rho_range, options.phi_range);

        check(all(spectrum.shape() == spectrum_shape.rfft()),
              "The rfft with shape {} doesn't match the logical shape {}",
              spectrum.shape(), spectrum_shape);

        if (polar.device().is_gpu()) {
            check(ng::is_accessor_access_safe<i32>(spectrum, spectrum.shape()) and
                  ng::is_accessor_access_safe<i32>(polar, polar.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            if constexpr (nt::is_texture_v<Input>)
                guts::launch_spectrum2polar_texture<i32>(spectrum, spectrum_shape.as<i32>(), polar, options);
            else
                guts::launch_spectrum2polar(spectrum, spectrum_shape.as<i32>(), polar, options);
        } else {
            guts::launch_spectrum2polar(spectrum, spectrum_shape.as<i64>(), polar, options);
        }
    }
}
