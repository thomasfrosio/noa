#pragma once

#include "noa/runtime/core/Enums.hpp"
#include "noa/runtime/core/Iwise.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/xform/core/Interpolation.hpp"
#include "noa/xform/PolarTransform.hpp"

namespace noa::xform::details {
    /// 3d iwise operator to compute the spectrum->polar transformation of 2d (r)FFT(s).
    template<nt::sinteger Index,
             nt::any_of<f32, f64> Coord,
             nt::interpolator_spectrum_nd<2> Input,
             nt::writable_nd<3> Output>
    class Spectrum2Polar {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        using coord_type = Coord;
        using coord2_type = Vec<coord_type, 2>;
        using shape2_type = Shape<Index, 2>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

    public:
        constexpr Spectrum2Polar(
            const input_type& spectrum,
            const shape2_type& spectrum_shape,
            const Linspace<coord_type>& spectrum_fftfreq,
            const output_type& polar,
            const shape2_type& polar_shape,
            Linspace<coord_type> rho,
            const Linspace<coord_type>& phi
        ) :
            m_spectrum(spectrum),
            m_polar(polar)
        {
            coord_type spectrum_stop{};
            for (usize i{}; i < 2; ++i) {
                const auto max_sample_size = spectrum_shape[i] / 2 + 1;
                const auto highest_fftfreq = nf::highest_fftfreq<coord_type>(spectrum_shape[i]);

                auto linspace = spectrum_fftfreq;
                if (linspace.stop <= 0) // default to highest fftfreq
                    linspace.stop = highest_fftfreq;
                if (not linspace.endpoint) // convert to endpoint=true
                    linspace.stop -= linspace.for_size(max_sample_size).step;

                spectrum_stop = std::max(spectrum_stop, linspace.stop);

                const auto polar2spectrum = highest_fftfreq / linspace.stop;
                m_scale[i] = polar2spectrum * static_cast<coord_type>(spectrum_shape[i]);
            }

            // Polar start defaults to the input start, which is always 0.
            if (rho.start < 0)
                rho.start = 0;
            m_rho_start = rho.start;

            // Polar stop defaults to the input stop.
            if (rho.stop <= 0)
                rho.stop = spectrum_stop;
            m_rho_step = rho.for_size(polar_shape[1]).step;

            m_phi_start = phi.start;
            m_phi_step = phi.for_size(polar_shape[0]).step;
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_phi_step + m_phi_start;
            const coord_type rho = polar_coordinate[1] * m_rho_step + m_rho_start;
            const coord2_type frequency = (rho * sincos(phi)) * m_scale;
            auto value = m_spectrum.interpolate_spectrum_at(frequency, batch);
            m_polar(batch, y, x) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_spectrum;
        output_type m_polar;
        coord2_type m_scale;
        coord_type m_rho_start;
        coord_type m_rho_step;
        coord_type m_phi_start;
        coord_type m_phi_step;
    };

    template<nf::Layout REMAP, bool IS_GPU = false, typename Index, typename Input, typename Output, typename Options>
    void launch_spectrum2polar(
        Input&& spectrum,
        const Shape<Index, 4>& spectrum_shape,
        Output&& polar,
        const Options& options
    ) {
        using input_real_type = nt::mutable_value_type_twice_t<Input>;
        using output_real_type = nt::value_type_twice_t<Output>;
        using coord_t = nt::largest_type_t<f32, input_real_type, output_real_type>;

        auto spectrum_range = options.spectrum_fftfreq.template as<coord_t>();
        auto rho_range = options.rho_range.template as<coord_t>();
        auto phi_range = options.phi_range.template as<coord_t>();

        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 3, Index>;
        const auto output_accessor = output_accessor_t(polar.get(), polar.strides().filter(0, 2, 3).template as<Index>());

        auto launch_iwise = [&](auto interp) {
            auto interpolator = to_interpolator_spectrum<2, REMAP, interp(), coord_t, IS_GPU>(
                spectrum, spectrum_shape
            );
            auto polar_shape = polar.shape().filter(0, 2, 3).template as<Index>();
            auto op = Spectrum2Polar<Index, coord_t, decltype(interpolator), output_accessor_t>(
                interpolator, spectrum_shape.filter(2, 3), spectrum_range,
                output_accessor, polar_shape.pop_front(),
                rho_range, phi_range
            );
            return iwise<IwiseOptions{
                .generate_cpu = not IS_GPU,
                .generate_gpu = IS_GPU,
            }>(polar_shape, polar.device(), op,
               std::forward<Input>(spectrum), std::forward<Output>(polar));
        };

        Interp interp = options.interp;
        if constexpr (nt::texture_decay<Input>)
            interp = spectrum.interp();
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
}

namespace noa::xform {
    struct PolarTransformSpectrumOptions {
        /// Frequency range of the cartesian input, along the cartesian axes, in cycle/pixels (fftfreq).
        /// If the stop is negative or zero, it is set to the highest frequencies for each dimension, i.e.,
        /// the entire rfft/fft range is selected. For even dimensions, this is equivalent to {0, 0.5}.
        /// Note that the start should be 0; otherwise an error will be thrown.
        Linspace<f64> spectrum_fftfreq{.start = 0., .stop = -1., .endpoint = true};

        /// Rho range of the bounding shells to transform, in cycle/pixels (fftfreq).
        /// Rho maps to the width dimension of the polar array. A negative value (or zero for the stop)
        /// defaults to the corresponding value of the spectrum fftfreq range. If the spectrum fftfreq is itself
        /// defaulted, this would be equal to [0, max(noa::fft::highest_fftfreq(input_shape))].
        Linspace<f64> rho_range{.start = 0., .stop = -1., .endpoint = true};

        /// Phi angle range increasing in the counterclockwise orientation, in radians.
        /// Phi maps to the height dimension of the polar array. While the range naturally included in the
        /// HC-layout FFT is [-pi/2, pi/2], this range can include the entire unit circle, e.g. [-pi, pi].
        /// Defaults to [0, pi).
        Linspace<f64> phi_range{.start = 0., .stop = Constant<f64>::PI, .endpoint = false};

        /// Interpolation method used to interpolate the values onto the new polar grid.
        /// Out-of-bounds elements are set to zero.
        /// This is unused if a texture is passed to the transform function.
        Interp interp{Interp::LINEAR};
    };

    // TODO Add polar2spectrum ?

    /// Transforms 2d DFT(s) to polar coordinates.
    /// \tparam REMAP           Every input layout is supported (see InterpolateSpectrum).
    ///                         The output is denoted as "FC" (full-centered) to emphasize that it has a full shape
    ///                         (equals to polar_shape) and can map the entire angular range (e.g. 0 to 2PI).
    /// \param[in] spectrum     2d (r)FFT to interpolate onto the polar coordinate system.
    /// \param spectrum_shape   BDHW logical shape of spectrum.
    /// \param[out] polar       Transformed 2d array on the polar grid.
    ///                         If real, and spectrum is complex, the power spectrum is computed.
    /// \param options          Transformation options.
    template<nf::Layout REMAP,
             nt::varray_or_texture_decay Input,
             nt::writable_varray_decay Output>
    requires (REMAP.is_xx2fc() and nt::spectrum_types<nt::value_type_t<Input>, nt::value_type_t<Output>>)
    void spectrum2polar(
        Input&& spectrum,
        const Shape4& spectrum_shape,
        Output&& polar,
        PolarTransformSpectrumOptions options = {}
    ) {
        details::polar_check_parameters(spectrum, polar);

        check(spectrum.shape() == (REMAP.is_hx2xx() ? spectrum_shape.rfft() : spectrum_shape),
              "The logical shape {} does not match the spectrum shape. Got spectrum:shape={}, REMAP={}",
              spectrum_shape, spectrum.shape(), REMAP);
        check(allclose(options.spectrum_fftfreq.start, 0.),
              "For multidimensional cases, the starting fftfreq should be 0, but got {}",
              options.spectrum_fftfreq.start);

        if (polar.device().is_gpu()) {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (nt::texture_decay<Input> and not nt::any_of<nt::value_type_t<Input>, f32, c32>) {
                std::terminate(); // unreachable
            } else {
                details::launch_spectrum2polar<REMAP, true>(
                    std::forward<Input>(spectrum), spectrum_shape.as<isize>(),
                    std::forward<Output>(polar), options
                );
            }
            #else
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            details::launch_spectrum2polar<REMAP>(
                std::forward<Input>(spectrum), spectrum_shape.as<isize>(),
                std::forward<Output>(polar), options
            );
        }
    }
}
