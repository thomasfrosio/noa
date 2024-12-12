#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Interpolation.hpp"
#include "noa/unified/geometry/PolarTransform.hpp"

namespace noa::geometry::guts {
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
        using coord2_type = Vec2<coord_type>;
        using shape2_type = Shape2<Index>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type>);

    public:
        constexpr Spectrum2Polar(
            const input_type& spectrum,
            const shape2_type& spectrum_shape,
            const output_type& polar,
            const shape2_type& polar_shape,
            coord2_type fftfreq_range,
            bool fftfreq_endpoint,
            const coord2_type& angle_range,
            bool angle_endpoint
        ) :
            m_spectrum(spectrum),
            m_polar(polar),
            m_start_angle(angle_range[0]),
            m_start_fftfreq(fftfreq_range[0])
        {
            NOA_ASSERT(fftfreq_range[1] - fftfreq_range[0] >= 0);
            m_step_angle = Linspace{angle_range[0], angle_range[1], angle_endpoint}.for_size(polar_shape[0]).step;
            m_step_fftfreq = Linspace{fftfreq_range[0], fftfreq_range[1], fftfreq_endpoint}.for_size(polar_shape[1]).step;

            // Scale the frequency range to the polar dimension [0,width).
            m_scale = coord2_type::from_vec(spectrum_shape.vec);
        }

        NOA_HD constexpr void operator()(index_type batch, index_type y, index_type x) const {
            const auto polar_coordinate = coord2_type::from_values(y, x);
            const coord_type phi = polar_coordinate[0] * m_step_angle + m_start_angle;
            const coord_type rho = polar_coordinate[1] * m_step_fftfreq + m_start_fftfreq;
            const coord2_type frequency = (rho * sincos(phi)) * m_scale;
            auto value = m_spectrum.interpolate_spectrum_at(frequency, batch);
            m_polar(batch, y, x) = cast_or_abs_squared<output_value_type>(value);
        }

    private:
        input_type m_spectrum;
        output_type m_polar;
        coord2_type m_scale;
        coord_type m_step_angle;
        coord_type m_start_angle;
        coord_type m_step_fftfreq;
        coord_type m_start_fftfreq;
    };

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

        auto launch_iwise = [&](auto interp) {
            auto interpolator = ng::to_interpolator_spectrum<2, REMAP, interp(), coord_t, IS_GPU>(
                spectrum, spectrum_shape);

            auto polar_shape = polar.shape().filter(0, 2, 3).template as<Index>();
            auto op = Spectrum2Polar<Index, coord_t, decltype(interpolator), output_accessor_t>(
                interpolator, spectrum_shape.filter(2, 3),
                output_accessor, polar_shape.pop_front(),
                rho_range, options.rho_endpoint,
                phi_range, options.phi_endpoint);

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

    inline void set_spectrum2polar_defaults(
        const Shape4<i64>& cartesian_shape,
        Vec2<f64>& rho_range,
        Vec2<f64>& angle_range
    ) {
        if (rho_range[0] <= 0) {
            // Find highest fftfreq. If any dimension is even sized, this is 0.5.
            rho_range[1] = std::max(noa::fft::highest_fftfreq<f64>(cartesian_shape[1]), rho_range[1]);
            rho_range[1] = std::max(noa::fft::highest_fftfreq<f64>(cartesian_shape[2]), rho_range[1]);
            rho_range[1] = std::max(noa::fft::highest_fftfreq<f64>(cartesian_shape[3]), rho_range[1]);
        }
        if (vall(IsZero{}, angle_range))
            angle_range = {0., Constant<f64>::PI};
    }
}

namespace noa::geometry {
    struct PolarTransformSpectrumOptions {
        /// Rho [start, end] range of the bounding shells to transform, in cycle/pixels (fftfreq).
        /// Rho maps to the width dimension of the polar array.
        /// A negative or zero end-frequency defaults the highest fftfreq along the cartesian axes.
        Vec2<f64> rho_range{0, -1};

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
        guts::set_spectrum2polar_defaults(spectrum_shape, options.rho_range, options.phi_range);

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
            panic_no_gpu_backend(); // unreachable
            #endif
        } else {
            guts::launch_spectrum2polar<REMAP>(
                std::forward<Input>(spectrum), spectrum_shape.as<i64>(),
                std::forward<Output>(polar), options);
        }
    }
}
