#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/geometry/Polar.hpp"
#include "noa/core/fft/Frequency.hpp"

namespace noa::signal {
    /// Calculate the relativistic electron wavelength in meters given an acceleration potential in volts.
    /// \see https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory
    [[nodiscard]] NOA_IHD f64 relativistic_electron_wavelength(f64 energy) noexcept {
        constexpr auto h = Constant<f64>::PLANCK;
        constexpr auto c = Constant<f64>::SPEED_OF_LIGHT;
        constexpr auto m0 = Constant<f64>::ELECTRON_MASS;
        constexpr auto e = Constant<f64>::ELEMENTARY_CHARGE;
        const auto V = energy;
        const auto eV = e * V;

        constexpr auto numerator = h * c;
        const auto denominator = sqrt(eV * (2 * m0 * c * c + eV));
        return numerator / denominator;
    }

    // TODO Anisotropic b-factor
    // TODO Pixel angle?
    template<nt::any_of<f32, f64> Real>
    class CTFAnisotropic;

    template<nt::any_of<f32, f64> T>
    class CTFIsotropic {
    public:
        using value_type = T;
        struct Parameters {
            value_type pixel_size;
            value_type defocus;
            value_type voltage;
            value_type amplitude;
            value_type cs;
            value_type phase_shift;
            value_type bfactor;
            value_type scale;
        };

    public: // constructors
        constexpr CTFIsotropic() = default;

        /// Create an isotropic CTF.
        /// \param pixel_size   Pixel size in A/p.
        /// \param defocus      Defocus in μm, positive is underfocused.
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        /// \param scale        Post-processing scaling-factor.
        constexpr CTFIsotropic(
            value_type pixel_size,
            value_type defocus,
            value_type voltage,
            value_type amplitude,
            value_type cs,
            value_type phase_shift,
            value_type bfactor,
            value_type scale
        ) :
            m_pixel_size(pixel_size),
            m_defocus_angstroms(-defocus * static_cast<value_type>(1e4)), // micrometers -> angstroms
            m_phase_shift(phase_shift),
            m_cs_angstroms(cs * static_cast<value_type>(1e7)), // mm -> angstroms
            m_voltage_volts(voltage * static_cast<value_type>(1e3)), // kV -> V
            m_amplitude(amplitude),
            m_scale(scale)
        {
            set_bfactor(bfactor);
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

        constexpr explicit CTFIsotropic(Parameters parameters) : CTFIsotropic(
            parameters.pixel_size,
            parameters.defocus,
            parameters.voltage,
            parameters.amplitude,
            parameters.cs,
            parameters.phase_shift,
            parameters.bfactor,
            parameters.scale
        ) {}

        constexpr explicit CTFIsotropic(
            const CTFAnisotropic<value_type>& ctf_anisotropic
        ) : CTFIsotropic(
            mean(ctf_anisotropic.pixel_size()), // average pixel size
            ctf_anisotropic.defocus().value, // of course, ignore the astigmatism
            ctf_anisotropic.voltage(),
            ctf_anisotropic.amplitude(),
            ctf_anisotropic.cs(),
            ctf_anisotropic.phase_shift(),
            ctf_anisotropic.bfactor(),
            ctf_anisotropic.scale()
        ) {}

    public: // getters
        [[nodiscard]] constexpr value_type pixel_size() const { return m_pixel_size; }
        [[nodiscard]] constexpr value_type sampling_rate() const { return pixel_size(); }
        [[nodiscard]] constexpr value_type phase_shift() const { return m_phase_shift; }
        [[nodiscard]] constexpr value_type amplitude() const { return m_amplitude; }
        [[nodiscard]] constexpr value_type scale() const { return m_scale; }
        [[nodiscard]] constexpr value_type bfactor() const {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] constexpr value_type defocus() const {
            return -m_defocus_angstroms * static_cast<value_type>(1e-4); // angstrom -> micrometers
        }
        [[nodiscard]] constexpr value_type cs() const {
            return m_cs_angstroms * static_cast<value_type>(1e-7); // angstrom -> millimeters
        }
        [[nodiscard]] constexpr value_type voltage() const {
            return m_voltage_volts * static_cast<value_type>(1e-3); // volts -> kilovolts
        }

    public: // setters
        constexpr void set_pixel_size(value_type pixel_size) { m_pixel_size = pixel_size; }
        constexpr void set_sampling_rate(value_type sampling_rate) { set_pixel_size(sampling_rate); }
        constexpr void set_phase_shift(value_type phase_shift) { m_phase_shift = phase_shift; }
        constexpr void set_scale(value_type scale) { m_scale = scale; }
        constexpr void set_bfactor(value_type bfactor) {
            m_bfactor_forth = bfactor / 4;
        }
        constexpr void set_defocus(value_type defocus) {
            m_defocus_angstroms = -defocus * static_cast<value_type>(1e4);
        }
        constexpr void set_cs(value_type cs) {
            m_cs_angstroms = cs * * static_cast<value_type>(1e7);
        }
        constexpr void set_voltage(value_type voltage) {
            m_voltage_volts = voltage * static_cast<value_type>(1e3);
            set_lambda_and_cs_();
        }
        constexpr void set_amplitude(value_type amplitude) {
            m_amplitude = amplitude;
            set_amplitude_fraction_();
        }

    public:
        [[nodiscard]] constexpr value_type phase_at(nt::real auto fftfreq) const {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            return phase;
        }

        [[nodiscard]] constexpr value_type value_at(nt::real auto fftfreq) const {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            auto ctf = -sin(phase);
            if (m_bfactor_forth != 0)
                ctf *= exp(m_bfactor_forth * r2);
            return ctf * m_scale;
        }

    private:
        NOA_HD void set_lambda_and_cs_() {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = PI * lambda;
            m_k2 = PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda;
        }

        constexpr void set_amplitude_fraction_() {
            m_k3 = atan(m_amplitude / sqrt(1 - m_amplitude * m_amplitude));
        }

    private:
        value_type m_pixel_size{};
        value_type m_defocus_angstroms{};
        value_type m_phase_shift{};
        value_type m_cs_angstroms{};
        value_type m_voltage_volts{};
        value_type m_amplitude{};
        value_type m_bfactor_forth{};
        value_type m_scale{};

        value_type m_k1;
        value_type m_k2;
        value_type m_k3;
    };

    /// Astigmatic defocus parameters.
    /// \param value        Defocus in μm, positive is underfocused.
    /// \param astigmatism  Amount of astigmatism in μm.
    /// \param angle        Angle of astigmatism in radians.
    ///                     Positive is counterclockwise when looking at the origin.
    /// \note With \p angle=0, \c height=v, \c width=u:
    ///     \c defocus_u=value+astigmatism
    ///     \c defocus_v=value-astigmatism
    template<nt::any_of<f32, f64> Real>
    struct DefocusAstigmatic {
        Real value;
        Real astigmatism;
        Real angle;
    };

    template<nt::any_of<f32, f64> T>
    class CTFAnisotropic {
    public:
        using value_type = T;
        using pixel_size_type = Vec2<value_type>;
        using defocus_type = DefocusAstigmatic<value_type>;
        struct Parameters {
            pixel_size_type pixel_size;
            defocus_type defocus;
            value_type voltage;
            value_type amplitude;
            value_type cs;
            value_type phase_shift;
            value_type bfactor;
            value_type scale;
        };

    public: // constructors
        constexpr CTFAnisotropic() = default;

        /// Create an anisotropic CTF.
        /// \param pixel_size   HW pixel size in A/p.
        /// \param defocus      Astigmatic defocus.
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        /// \param scale        Post-processing scaling-factor.
        constexpr CTFAnisotropic(
            pixel_size_type pixel_size,
            defocus_type defocus,
            value_type voltage,
            value_type amplitude,
            value_type cs,
            value_type phase_shift,
            value_type bfactor,
            value_type scale
        ) :
            m_pixel_size(pixel_size),
            m_phase_shift(phase_shift),
            m_cs_angstroms(cs * static_cast<value_type>(1e7)), // mm -> angstroms
            m_voltage_volts(voltage * static_cast<value_type>(1e3)), // kV -> V
            m_amplitude(amplitude),
            m_scale(scale)
        {
            set_defocus(defocus);
            set_bfactor(bfactor);
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

        constexpr explicit CTFAnisotropic(const Parameters& parameters) : CTFAnisotropic(
            parameters.pixel_size,
            parameters.defocus,
            parameters.voltage,
            parameters.amplitude,
            parameters.cs,
            parameters.phase_shift,
            parameters.bfactor,
            parameters.scale
        ) {}

        constexpr explicit CTFAnisotropic(
            const CTFIsotropic<value_type>& ctf_isotropic,
            value_type defocus_astigmatic_value = 0,
            value_type defocus_astigmatic_angle = 0
        ) :
            CTFAnisotropic(
                pixel_size_type::filled_with(ctf_isotropic.pixel_size()),
                defocus_type{ctf_isotropic.defocus(), defocus_astigmatic_value, defocus_astigmatic_angle},
                ctf_isotropic.voltage(),
                ctf_isotropic.amplitude(),
                ctf_isotropic.cs(),
                ctf_isotropic.phase_shift(),
                ctf_isotropic.bfactor(),
                ctf_isotropic.scale()) {}

    public: // getters
        [[nodiscard]] constexpr pixel_size_type pixel_size() const { return m_pixel_size; }
        [[nodiscard]] constexpr pixel_size_type sampling_rate() const { return pixel_size(); }
        [[nodiscard]] constexpr value_type phase_shift() const { return m_phase_shift; }
        [[nodiscard]] constexpr value_type amplitude() const { return m_amplitude; }
        [[nodiscard]] constexpr value_type scale() const { return m_scale; }
        [[nodiscard]] constexpr value_type bfactor() const {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] constexpr defocus_type defocus() const {
            return {-m_defocus_angstroms.value * static_cast<value_type>(1e-4), // angstroms -> micrometers
                    -m_defocus_angstroms.astigmatism * static_cast<value_type>(1e-4), // angstroms -> micrometers
                    m_defocus_angstroms.angle};
        }
        [[nodiscard]] constexpr value_type cs() const {
            return m_cs_angstroms * static_cast<value_type>(1e-7); // angstroms -> millimeters
        }
        [[nodiscard]] constexpr value_type voltage() const {
            return m_voltage_volts * static_cast<value_type>(1e-3); // volts -> kilovolts
        }

    public: // setters
        constexpr void set_pixel_size(pixel_size_type pixel_size) { m_pixel_size = pixel_size; }
        constexpr void set_sampling_rate(pixel_size_type sampling_rate) { set_pixel_size(sampling_rate); }
        constexpr void set_phase_shift(value_type phase_shift) { m_phase_shift = phase_shift; }
        constexpr void set_scale(value_type scale) { m_scale = scale; }
        constexpr void set_bfactor(value_type bfactor) {
            m_bfactor_forth = bfactor / 4;
        }
        constexpr void set_defocus(defocus_type defocus) {
            m_defocus_angstroms.value = -defocus.value * static_cast<value_type>(1e4); // micrometers -> angstroms
            m_defocus_angstroms.astigmatism = -defocus.astigmatism * static_cast<value_type>(1e4); // micrometers -> angstroms
            m_defocus_angstroms.angle = defocus.angle;
        }
        constexpr void set_cs(value_type cs) {
            m_cs_angstroms = cs * * static_cast<value_type>(1e7);
        }
        constexpr void set_voltage(value_type voltage) {
            m_voltage_volts = voltage * static_cast<value_type>(1e3);
            set_lambda_and_cs_();
        }
        constexpr void set_amplitude(value_type amplitude) {
            m_amplitude = amplitude;
            set_amplitude_fraction_();
        }

    public:
        template<nt::real Coord>
        [[nodiscard]] constexpr value_type phase_at(Vec2<Coord> fftfreq) const {
            const auto scaled_fftfreq = fftfreq.template as<value_type>() / m_pixel_size;
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_fftfreq);
            const auto rho = noa::geometry::cartesian2rho(scaled_fftfreq);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;

            const auto phase = m_k1 * r2 * phi2defocus_(phi) + m_k2 * r4 - m_phase_shift - m_k3;
            return phase;
        }

        template<nt::real Coord>
        [[nodiscard]] constexpr value_type value_at(Vec2<Coord> fftfreq) const {
            const auto scaled_fftfreq = fftfreq.template as<value_type>() / m_pixel_size;
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_fftfreq);
            const auto rho = noa::geometry::cartesian2rho(scaled_fftfreq);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;

            const auto phase = m_k1 * r2 * phi2defocus_(phi) + m_k2 * r4 - m_phase_shift - m_k3;
            auto ctf = -sin(phase);
            if (m_bfactor_forth != 0)
                ctf *= exp(m_bfactor_forth * r2);
            return ctf * m_scale;
        }

        template<nt::real Coord>
        [[nodiscard]] constexpr value_type isotropic_fftfreq(Vec2<Coord> fftfreq_2d) const {
            // Correct for the anisotropic pixel size directly in cartesian space.
            const auto scaled_fftfreq_2d = fftfreq_2d.template as<value_type>() / m_pixel_size;

            // Given rho, compute a scaling factor s(phi) such that rho_corrected = rho * s(phi),
            // rho_corrected being the anisotropy-corrected magnitude.
            // from doi:10.1016/j.ultramic.2014.01.009.
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_fftfreq_2d);
            const auto rho = noa::geometry::cartesian2rho(scaled_fftfreq_2d);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto l1 = m_lambda_angstroms;
            const auto l2 = l1 * l1;
            const auto l4 = l2 * l2;
            const auto c1 = m_cs_angstroms;
            const auto c2 = c1 * c1;
            const auto d1 = abs(m_defocus_angstroms.value);
            const auto d2 = d1 * d1;
            const auto da = phi2defocus_(phi);

            const auto t0 = abs(d2 + c2 * l4 * r4 + 2 * da * c1 * l2 * r2);
            const auto t1 = abs((d1 - sqrt(t0)) / (c1 * l2));
            const auto rho_corrected = sqrt(t1);

            // Scale back to fftfreq using average/isotropic spacing.
            return rho_corrected * mean(m_pixel_size);
        }

    private:
        [[nodiscard]] constexpr value_type phi2defocus_(value_type phi) const {
            const auto ellipse_angle = (phi - m_defocus_angstroms.angle);
            return m_defocus_angstroms.value + m_defocus_angstroms.astigmatism * cos(2 * ellipse_angle);
        }

        NOA_HD void set_lambda_and_cs_() {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            m_lambda_angstroms = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = PI * m_lambda_angstroms;
            m_k2 = PI * 0.5 * m_cs_angstroms * m_lambda_angstroms * m_lambda_angstroms * m_lambda_angstroms;
        }

        constexpr void set_amplitude_fraction_() {
            m_k3 = atan(m_amplitude / sqrt(1 - m_amplitude * m_amplitude));
        }

    private:
        pixel_size_type m_pixel_size{};
        defocus_type m_defocus_angstroms{};
        value_type m_phase_shift{};
        value_type m_cs_angstroms{};
        value_type m_voltage_volts{};
        value_type m_amplitude{};
        value_type m_bfactor_forth{};
        value_type m_scale{};

        value_type m_lambda_angstroms;
        value_type m_k1;
        value_type m_k2;
        value_type m_k3;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_ctf<noa::signal::CTFIsotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf<noa::signal::CTFAnisotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf_isotropic<noa::signal::CTFIsotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf_anisotropic<noa::signal::CTFAnisotropic<T>> : std::true_type {};
}

namespace noa::signal::guts {
    /// Index-wise operator, to compute/apply CTFs to {1|2|3}d DFTs.
    /// \details If the input is valid, input*ctf->output is computed for the full fftfreq range.
    ///          Note that if the input is complex and the output is real, abs(input*ctf)^2->output
    ///          is computed instead. If the input is empty, ctf->output is computed, for a user-defined
    ///          frequency range.
    template<Remap REMAP, size_t N,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Index,
             nt::readable_nd_or_empty<N + 1> Input,
             nt::writable_nd<N + 1> Output,
             nt::batched_parameter CTFParameter>
    class CTF {
    public:
        static_assert(REMAP.is_any(Remap::H2H, Remap::HC2HC, Remap::HC2H, Remap::H2HC,
                                   Remap::F2F, Remap::FC2FC, Remap::FC2F, Remap::F2FC));

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type> or
                      (nt::empty<input_type> and nt::real_or_complex<output_value_type>));

        using ctf_parameter_type = CTFParameter;
        using ctf_type = nt::value_type_t<ctf_parameter_type>;
        static_assert(nt::ctf<ctf_type>);

        static constexpr bool HAS_INPUT = not nt::empty<input_type>;
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_ISOTROPIC = nt::ctf_isotropic<ctf_type>;

        using index_type = Index;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<nt::empty<input_type>, coord_type, Empty>;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, N>;
        using frequency_range_type = Vec2<coord_type>;

    public:
        constexpr CTF(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const ctf_parameter_type& ctf,
            bool ctf_abs,
            bool ctf_squared
        ) requires HAS_INPUT :
            m_ctf(ctf),
            m_output(output),
            m_shape(shape.template pop_back<IS_RFFT>()),
            m_frequency_step(coord_type{1} / coord_nd_type::from_vec(shape.vec)),
            m_input(input),
            m_ctf_abs(ctf_abs),
            m_ctf_squared(ctf_squared) {}

        constexpr CTF(
            const output_type& output,
            const shape_nd_type& shape,
            const ctf_parameter_type& ctf,
            bool ctf_abs,
            bool ctf_squared,
            const frequency_range_type& frequency_range,
            bool frequency_range_endpoint
        ) requires (not HAS_INPUT) :
            m_ctf(ctf),
            m_output(output),
            m_shape(shape.template pop_back<IS_RFFT>()),
            m_frequency_start(frequency_range[0]),
            m_ctf_abs(ctf_abs),
            m_ctf_squared(ctf_squared)
        {
            // If frequency.end is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            for (size_t i{}; i < N; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_end =
                    frequency_range[1] < 0 ?
                    noa::fft::highest_normalized_frequency<coord_type>(shape[i]) :
                    frequency_range[1];
                m_frequency_step[i] = Linspace{
                    .start = frequency_range[0],
                    .stop = frequency_end,
                    .endpoint = frequency_range_endpoint
                }.for_size(max_sample_size).step;
            }
        }

    public:
        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD void operator()(
            index_type batch,
            I... output_indices
        ) const {
            auto frequency = noa::fft::index2frequency<IS_DST_CENTERED, IS_RFFT>(Vec{output_indices...}, m_shape);
            auto fftfreq = coord_nd_type::from_vec(frequency) * m_frequency_step;
            if constexpr (not HAS_INPUT)
                fftfreq += m_frequency_start;

            auto ctf = m_ctf[batch].value_at([&] {
                if constexpr (N == 1)
                    return fftfreq[0];
                else if constexpr ((N == 2 and IS_ISOTROPIC) or N == 3)
                    return norm(fftfreq);
                else // N == 2 anisotropic
                    return fftfreq;
            }());
            if (m_ctf_abs)
                ctf = abs(ctf);
            if (m_ctf_squared)
                ctf *= ctf;

            if constexpr (HAS_INPUT) {
                const auto input_indices = noa::fft::remap_indices<REMAP, true>(Vec{output_indices...}, m_shape);
                m_output(batch, output_indices...) = cast_or_abs_squared<output_value_type>(
                    m_input(input_indices.push_front(batch)) * static_cast<input_real_type>(ctf));
            } else {
                m_output(batch, output_indices...) = static_cast<output_value_type>(ctf);
            }
        }

    private:
        ctf_parameter_type m_ctf;
        output_type m_output;
        shape_type m_shape;
        coord_nd_type m_frequency_step;
        NOA_NO_UNIQUE_ADDRESS input_type m_input{};
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_frequency_start{};
        bool m_ctf_abs;
        bool m_ctf_squared;
    };
}
