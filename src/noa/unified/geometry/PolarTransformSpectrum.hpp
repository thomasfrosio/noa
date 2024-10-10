#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/geometry/PolarTransformSpectrum.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/geometry/PolarTransform.hpp"

namespace noa::geometry::guts {
    template<Remap REMAP, bool IS_GPU = false, typename Index, typename Input, typename Output, typename Options>
    void launch_spectrum2polar(
        Input&& spectrum,
        const Shape4<Index>& spectrum_shape,
        Output&& polar,
        const Options& options
    ) {
        using coord_t = nt::value_type_twice_t<Output>;
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto output_accessor = output_accessor_t(polar.get(), polar.strides().filter(0, 2, 3).template as<Index>());

        auto launch_iwise = [&]<Interp INTERP> {
            auto interpolator = ng::to_interpolator_spectrum<2, REMAP, INTERP, coord_t, IS_GPU>(
                spectrum, spectrum_shape);

            auto polar_shape = polar.shape().filter(0, 2, 3).template as<Index>();
            auto op = Spectrum2Polar<Index, coord_t, decltype(interpolator), output_accessor_t>(
                interpolator, spectrum_shape.filter(2, 3),
                output_accessor, polar_shape.pop_front(),
                rho_range, options.rho_endpoint,
                phi_range, options.phi_endpoint);

            return iwise<{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(polar_shape, polar.device(), op,
               std::forward<Input>(spectrum), std::forward<Output>(polar));
        };

        Interp interp = options.interp;
        if constexpr (nt::texture_decay<Input>)
            interp = spectrum.interp();
        switch (interp) {
            case Interp::NEAREST:            return launch_iwise.template operator()<Interp::NEAREST>();
            case Interp::NEAREST_FAST:       return launch_iwise.template operator()<Interp::NEAREST_FAST>();
            case Interp::LINEAR:             return launch_iwise.template operator()<Interp::LINEAR>();
            case Interp::LINEAR_FAST:        return launch_iwise.template operator()<Interp::LINEAR_FAST>();
            case Interp::CUBIC:              return launch_iwise.template operator()<Interp::CUBIC>();
            case Interp::CUBIC_FAST:         return launch_iwise.template operator()<Interp::CUBIC_FAST>();
            case Interp::CUBIC_BSPLINE:      return launch_iwise.template operator()<Interp::CUBIC_BSPLINE>();
            case Interp::CUBIC_BSPLINE_FAST: return launch_iwise.template operator()<Interp::CUBIC_BSPLINE_FAST>();
            case Interp::LANCZOS4:           return launch_iwise.template operator()<Interp::LANCZOS4>();
            case Interp::LANCZOS6:           return launch_iwise.template operator()<Interp::LANCZOS6>();
            case Interp::LANCZOS8:           return launch_iwise.template operator()<Interp::LANCZOS8>();
            case Interp::LANCZOS4_FAST:      return launch_iwise.template operator()<Interp::LANCZOS4_FAST>();
            case Interp::LANCZOS6_FAST:      return launch_iwise.template operator()<Interp::LANCZOS6_FAST>();
            case Interp::LANCZOS8_FAST:      return launch_iwise.template operator()<Interp::LANCZOS8_FAST>();
        }
    }
}

namespace noa::geometry {
    struct PolarTransformSpectrumOptions {
        /// Rho frequency [start,end] range of the bounding shells to transform, in cycle/pixels (fftfreq).
        /// Rho maps to the width dimension of the polar array.
        /// Defaults to the [0, v], where v is the highest normalized frequency of min(height,width).
        Vec2<f64> rho_range{};

        /// Whether the rho_range's end should be included in the range.
        /// The computed linspace range is Linspace{rho_range[0], rho_range[1], rho_endpoint}.for_size(polar_width).
        bool rho_endpoint{true};

        /// Phi angle [start,end) range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array.
        /// While the range naturally included in the non-redundant centered FFT is [-pi/2, pi/2],
        /// this range can include the entire unit circle, e.g. [-pi, pi]. Defaults to [0, pi).
        Vec2<f64> phi_range{};

        /// Whether the phi_range's end should be included in the range.
        /// The computed linspace range is Linspace{phi_range[0], phi_range[1], phi_endpoint}.for_size(polar_height).
        bool phi_endpoint{};

        /// Interpolation method used to interpolate the values onto the new grid.
        /// Out-of-bounds elements are set to zero.
        /// This is unused if a texture is passed to the transform function.
        Interp interp{Interp::LINEAR};
    };

    // TODO Add polar2spectrum ?

    /// Transforms 2d DFT(s) to polar coordinates.
    /// \tparam REMAP           Every input layout is supported (see InterpolateSpectrum).
    ///                         The output is denoted as "FC" (full-centered) to emphasize that it has a full shape
    ///                         (equals to polar_shape) and can map the entire angular range (e.g. 0 to 2PI).
    /// \param[in] spectrum     Centered 2d (r)FFT to interpolate onto the new coordinate system.
    /// \param spectrum_shape   BDHW logical shape of spectrum.
    /// \param[out] polar       Transformed 2d array on the polar grid.
    ///                         If real, and spectrum is complex, the power spectrum is computed.
    /// \param options          Transformation options.
    template<Remap REMAP,
             nt::varray_or_texture_decay Input,
             nt::writable_varray_decay Output>
    requires (REMAP.is_xx2fc() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    void spectrum2polar(
        Input&& spectrum,
        const Shape4<i64>& spectrum_shape,
        Output&& polar,
        PolarTransformSpectrumOptions options = {}
    ) {
        guts::polar_check_parameters(spectrum, polar);
        guts::set_polar_window_range_to_default(spectrum_shape, options.rho_range, options.phi_range);

        check(all(spectrum.shape() == (REMAP.is_hx2xx() ? spectrum_shape.rfft() : spectrum_shape)),
              "The logical shape {} does not match the spectrum shape. Got spectrum:shape={}, REMAP={}",
              spectrum_shape, spectrum.shape(), REMAP);

        if (polar.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                check(ng::is_accessor_access_safe<i32>(spectrum, spectrum.shape()) and
                      ng::is_accessor_access_safe<i32>(polar, polar.shape()),
                      "i64 indexing not instantiated for GPU devices");
                guts::launch_spectrum2polar<REMAP, true>(
                        std::forward<Input>(spectrum), spectrum_shape.as<i32>(),
                        std::forward<Output>(polar), options);
            }
            #else
            std::terminate(); // unreachable
            #endif
        } else {
            guts::launch_spectrum2polar<REMAP>(
                std::forward<Input>(spectrum), spectrum_shape.as<i64>(),
                std::forward<Output>(polar), options);
        }
    }
}
