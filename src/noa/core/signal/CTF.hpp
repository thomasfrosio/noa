#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/Linspace.hpp"
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
    template<typename Real>
    class CTFAnisotropic;

    template<typename Real>
    class CTFIsotropic {
    public:
        static_assert(std::is_floating_point_v<Real>);
        using value_type = Real;

    public: // constructors
        NOA_HD constexpr CTFIsotropic() = default;

        /// Create an isotropic CTF.
        /// \param pixel_size   Pixel size in A/p.
        /// \param defocus      Defocus in μm, positive is underfocused.
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        /// \param scale        Post-processing scaling-factor.
        NOA_HD constexpr CTFIsotropic(
                value_type pixel_size,
                value_type defocus,
                value_type voltage,
                value_type amplitude,
                value_type cs,
                value_type phase_shift,
                value_type bfactor,
                value_type scale
        ) noexcept:
                m_pixel_size(pixel_size),
                m_defocus_angstroms(-defocus * static_cast<Real>(1e4)), // micrometers -> angstroms
                m_phase_shift(phase_shift),
                m_cs_angstroms(cs * static_cast<Real>(1e7)), // mm -> angstroms
                m_voltage_volts(voltage * static_cast<Real>(1e3)), // kV -> V
                m_amplitude(amplitude),
                m_scale(scale) {
            set_bfactor(bfactor);
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

        NOA_HD constexpr explicit CTFIsotropic(
                const CTFAnisotropic<Real>& ctf_anisotropic
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
        [[nodiscard]] NOA_HD constexpr value_type pixel_size() const noexcept { return m_pixel_size; }
        [[nodiscard]] NOA_HD constexpr value_type phase_shift() const noexcept { return m_phase_shift; }
        [[nodiscard]] NOA_HD constexpr value_type amplitude() const noexcept { return m_amplitude; }
        [[nodiscard]] NOA_HD constexpr value_type scale() const noexcept { return m_scale; }
        [[nodiscard]] NOA_HD constexpr value_type bfactor() const noexcept {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] NOA_HD constexpr value_type defocus() const noexcept {
            return -m_defocus_angstroms * static_cast<Real>(1e-4); // angstrom -> micrometers
        }
        [[nodiscard]] NOA_HD constexpr value_type cs() const noexcept {
            return m_cs_angstroms * static_cast<Real>(1e-7); // angstrom -> millimeters
        }
        [[nodiscard]] NOA_HD constexpr value_type voltage() const noexcept {
            return m_voltage_volts * static_cast<Real>(1e-3); // volts -> kilovolts
        }

    public: // setters
        NOA_HD constexpr void set_pixel_size(value_type pixel_size) noexcept { m_pixel_size = pixel_size; }
        NOA_HD constexpr void set_phase_shift(value_type phase_shift) noexcept { m_phase_shift = phase_shift; }
        NOA_HD constexpr void set_scale(value_type scale) noexcept { m_scale = scale; }
        NOA_HD constexpr void set_bfactor(value_type bfactor) noexcept {
            m_bfactor_forth = bfactor / 4;
        }
        NOA_HD constexpr void set_defocus(value_type defocus) noexcept {
            m_defocus_angstroms = -defocus * static_cast<Real>(1e4);
        }
        NOA_HD constexpr void set_cs(value_type cs) noexcept {
            m_cs_angstroms = cs * * static_cast<Real>(1e7);
        }
        NOA_HD constexpr void set_voltage(value_type voltage) noexcept {
            m_voltage_volts = voltage * static_cast<Real>(1e3);
            set_lambda_and_cs_();
        }
        NOA_HD constexpr void set_amplitude(value_type amplitude) noexcept {
            m_amplitude = amplitude;
            set_amplitude_fraction_();
        }

    public:
        [[nodiscard]] NOA_HD constexpr value_type phase_at(std::floating_point auto fftfreq) const noexcept {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            return phase;
        }

        [[nodiscard]] NOA_HD constexpr value_type value_at(std::floating_point auto fftfreq) const noexcept {
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
        NOA_HD void set_lambda_and_cs_() noexcept {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = PI * lambda;
            m_k2 = PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda;
        }

        NOA_HD constexpr void set_amplitude_fraction_() noexcept {
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
    template<typename Real>
    struct DefocusAstigmatic {
        Real value;
        Real astigmatism;
        Real angle;
    };

    template<typename Real>
    class CTFAnisotropic {
    public:
        static_assert(std::is_floating_point_v<Real>);
        using value_type = Real;
        using pixel_size_type = Vec2<value_type>;
        using defocus_type = DefocusAstigmatic<value_type>;

    public: // constructors
        NOA_HD constexpr CTFAnisotropic() = default;

        /// Create an anisotropic CTF.
        /// \param pixel_size   HW pixel size in A/p.
        /// \param defocus      Astigmatic defocus.
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        /// \param scale        Post-processing scaling-factor.
        NOA_HD constexpr CTFAnisotropic(
                pixel_size_type pixel_size,
                defocus_type defocus,
                value_type voltage,
                value_type amplitude,
                value_type cs,
                value_type phase_shift,
                value_type bfactor,
                value_type scale
        ) noexcept:
                m_pixel_size(pixel_size),
                m_phase_shift(phase_shift),
                m_cs_angstroms(cs * static_cast<Real>(1e7)), // mm -> angstroms
                m_voltage_volts(voltage * static_cast<Real>(1e3)), // kV -> V
                m_amplitude(amplitude),
                m_scale(scale) {
            set_defocus(defocus);
            set_bfactor(bfactor);
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

        NOA_HD constexpr explicit CTFAnisotropic(
                const CTFIsotropic<Real>& ctf_isotropic,
                value_type defocus_astigmatic_value = 0,
                value_type defocus_astigmatic_angle = 0
        ) noexcept:
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
        [[nodiscard]] NOA_HD constexpr pixel_size_type pixel_size() const noexcept { return m_pixel_size; }
        [[nodiscard]] NOA_HD constexpr value_type phase_shift() const noexcept { return m_phase_shift; }
        [[nodiscard]] NOA_HD constexpr value_type amplitude() const noexcept { return m_amplitude; }
        [[nodiscard]] NOA_HD constexpr value_type scale() const noexcept { return m_scale; }
        [[nodiscard]] NOA_HD constexpr value_type bfactor() const noexcept {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] NOA_HD constexpr defocus_type defocus() const noexcept {
            return {-m_defocus_angstroms.value * static_cast<Real>(1e-4), // angstroms -> micrometers
                    -m_defocus_angstroms.astigmatism * static_cast<Real>(1e-4), // angstroms -> micrometers
                    m_defocus_angstroms.angle};
        }
        [[nodiscard]] NOA_HD constexpr value_type cs() const noexcept {
            return m_cs_angstroms * static_cast<Real>(1e-7); // angstroms -> millimeters
        }
        [[nodiscard]] NOA_HD constexpr value_type voltage() const noexcept {
            return m_voltage_volts * static_cast<Real>(1e-3); // volts -> kilovolts
        }

    public: // setters
        NOA_HD constexpr void set_pixel_size(pixel_size_type pixel_size) noexcept { m_pixel_size = pixel_size; }
        NOA_HD constexpr void set_phase_shift(value_type phase_shift) noexcept { m_phase_shift = phase_shift; }
        NOA_HD constexpr void set_scale(value_type scale) noexcept { m_scale = scale; }
        NOA_HD constexpr void set_bfactor(value_type bfactor) noexcept {
            m_bfactor_forth = bfactor / 4;
        }
        NOA_HD constexpr void set_defocus(defocus_type defocus) noexcept {
            m_defocus_angstroms.value = -defocus.value * static_cast<Real>(1e4); // micrometers -> angstroms
            m_defocus_angstroms.astigmatism = -defocus.astigmatism * static_cast<Real>(1e4); // micrometers -> angstroms
            m_defocus_angstroms.angle = defocus.angle;
        }
        NOA_HD constexpr void set_cs(value_type cs) noexcept {
            m_cs_angstroms = cs * * static_cast<Real>(1e7);
        }
        NOA_HD constexpr void set_voltage(value_type voltage) noexcept {
            m_voltage_volts = voltage * static_cast<Real>(1e3);
            set_lambda_and_cs_();
        }
        NOA_HD constexpr void set_amplitude(value_type amplitude) noexcept {
            m_amplitude = amplitude;
            set_amplitude_fraction_();
        }

    public:
        template<std::floating_point Coord>
        [[nodiscard]] NOA_HD constexpr value_type phase_at(Vec2<Coord> fftfreq) const noexcept {
            const auto scaled_fftfreq = fftfreq.template as<value_type>() / m_pixel_size;
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_fftfreq);
            const auto rho = noa::geometry::cartesian2rho(scaled_fftfreq);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;

            const auto phase = m_k1 * r2 * phi2defocus_(phi) + m_k2 * r4 - m_phase_shift - m_k3;
            return phase;
        }

        template<std::floating_point Coord>
        [[nodiscard]] NOA_HD constexpr value_type value_at(Vec2<Coord> fftfreq) const noexcept {
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

        template<std::floating_point Coord>
        [[nodiscard]] NOA_HD constexpr value_type isotropic_fftfreq(Vec2<Coord> fftfreq_2d) const noexcept {
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
        [[nodiscard]] NOA_HD constexpr value_type phi2defocus_(value_type phi) const noexcept {
            const auto ellipse_angle = (phi - m_defocus_angstroms.angle);
            return m_defocus_angstroms.value + m_defocus_angstroms.astigmatism * cos(2 * ellipse_angle);
        }

        NOA_HD void set_lambda_and_cs_() noexcept {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            m_lambda_angstroms = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = PI * m_lambda_angstroms;
            m_k2 = PI * 0.5 * m_cs_angstroms * m_lambda_angstroms * m_lambda_angstroms * m_lambda_angstroms;
        }

        NOA_HD constexpr void set_amplitude_fraction_() noexcept {
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

namespace noa::signal {
    /// Index-wise operator, to compute/apply CTFs to {1|2|3}d DFTs.
    /// \details If the input is valid, input*ctf->output is computed for the full fftfreq range.
    ///          Note that if the input is complex and the output is real, abs(input*ctf)^2->output
    ///          is computed instead. If the input is empty, ctf->output is computed, for a user-defined
    ///          frequency range.
    ///
    /// \tparam REMAP           H(C)2H(C) or F(C)2F(C).
    /// \tparam NDIM            1d, 2d, or 3d.
    /// \tparam Coord           Coordinate type. f32 or f64.
    /// \tparam Index           Index/Integer type. i32 or i64.
    /// \tparam InputAccessor   Accessor<const {f32|f64|c32|c64},NDIM+1> (+1 for the batch), or Empty.
    /// \tparam OutputAccessor  Accessor<{f32|f64|c32|c64},NDIM+1> (+1 for the batch).
    /// \tparam CTFAccessor     AccessorRestrictContiguous<const CTF,1> or CTF, where CTF is satisfies nt::is_ctf.
    template<noa::fft::Remap REMAP, size_t NDIM, typename Coord, typename Index,
             typename InputAccessor, typename OutputAccessor, typename CTFAccessor>
    class CTF {
    public:
        static constexpr bool IS_VALID_CTF =
                nt::is_ctf_v<CTFAccessor> or
                (nt::is_accessor_1d_restrict_contiguous_v<CTFAccessor> and
                 std::is_const_v<nt::value_type_t<CTFAccessor>> and
                 nt::is_ctf_v<nt::value_type_t<CTFAccessor>>);

        static constexpr bool IS_VALID_REMAP =
                REMAP == noa::fft::Remap::H2H or REMAP == noa::fft::Remap::HC2HC or
                REMAP == noa::fft::Remap::HC2H or REMAP == noa::fft::Remap::H2HC or
                REMAP == noa::fft::Remap::F2F or REMAP == noa::fft::Remap::FC2FC or
                REMAP == noa::fft::Remap::FC2F or REMAP == noa::fft::Remap::F2FC;
        static constexpr bool IS_VALID_COORD_INDEX =
                nt::is_sint_v<Index> and nt::is_any_v<Coord, f32, f64>;

        using output_value_t = nt::value_type_t<OutputAccessor>;
        static constexpr bool IS_VALID_OUTPUT =
                nt::is_accessor_nd_v<OutputAccessor, NDIM + 1> and
                !std::is_const_v<output_value_t> and
                nt::is_real_or_complex_v<output_value_t>;

        using input_value_t = nt::value_type_t<InputAccessor>;
        static constexpr bool IS_VALID_INPUT =
                std::is_empty_v<InputAccessor> or
                (nt::is_accessor_nd_v<InputAccessor, NDIM + 1> and
                 std::is_const_v<input_value_t> and
                 (nt::are_complex_v<output_value_t, input_value_t> or
                  (nt::is_real_v<output_value_t> and nt::is_real_or_complex_v<input_value_t>)));

        static constexpr bool IS_VALID =
                IS_VALID_CTF and IS_VALID_REMAP and IS_VALID_COORD_INDEX and IS_VALID_OUTPUT and IS_VALID_INPUT;

        using input_type = InputAccessor;
        using input_or_empty_type = InputAccessor;
        using output_type = OutputAccessor;
        using ctf_type = CTFAccessor;
        using input_value_type = nt::mutable_value_type_t<InputAccessor>; // ignore const
        using output_value_type = nt::value_type_t<OutputAccessor>;
        using ctf_value_type = nt::value_type_twice_t<ctf_type>;
        using input_real_type = nt::value_type_t<input_value_type>;

        using index_type = Index;
        using coord_type = Coord;
        using coord_or_empty_type = std::conditional_t<std::is_empty_v<input_or_empty_type>, coord_type, Empty>;
        using shape_nd_type = Shape<index_type, NDIM>;
        using coord_nd_type = Vec<coord_type, NDIM>;
        using frequency_range_type = Vec2<coord_type>;

        static constexpr bool HAS_INPUT = !std::is_empty_v<input_or_empty_type>;
        static constexpr bool IS_RFFT = static_cast<u8>(REMAP) & noa::fft::Layout::DST_HALF; // or SRC_HALF
        static constexpr bool IS_DST_CENTERED = static_cast<u8>(REMAP) & noa::fft::Layout::DST_CENTERED;
        static constexpr bool IS_ISOTROPIC =
                nt::is_ctf_isotropic_v<ctf_type> or
                (nt::is_accessor_v<ctf_type> and nt::is_ctf_isotropic_v<nt::value_type_t<ctf_type>>);

    public:
        CTF(
                const input_type& input,
                const output_type& output,
                const shape_nd_type& shape,
                const ctf_type& ctf,
                bool ctf_abs,
                bool ctf_squared
        ) noexcept requires HAS_INPUT :
                m_ctf(ctf),
                m_input(input),
                m_output(output),
                m_shape(shape),
                m_frequency_step(coord_type{1} / coord_nd_type::from_vec(shape.vec)),
                m_ctf_abs(ctf_abs),
                m_ctf_squared(ctf_squared) {}

        CTF(const output_type& output,
              const shape_nd_type& shape,
              const ctf_type& ctf,
              bool ctf_abs,
              bool ctf_squared,
              const frequency_range_type& frequency_range,
              bool frequency_range_endpoint
        ) noexcept requires (not HAS_INPUT) :
                m_ctf(ctf),
                m_output(output),
                m_shape(shape),
                m_frequency_start(frequency_range[0]),
                m_ctf_abs(ctf_abs),
                m_ctf_squared(ctf_squared) {
            // If frequency-end is negative, defaults to the highest frequency.
            // In this case, and if the frequency-start is 0, this results in the full frequency range.
            for (size_t i = 0; i < NDIM; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_end =
                        frequency_range[1] < 0 ?
                        noa::fft::highest_normalized_frequency<coord_type>(shape[i]) :
                        frequency_range[1];
                m_frequency_step[i] = Linspace<coord_type, index_type>::from_range(
                        frequency_range[0], frequency_end, max_sample_size, frequency_range_endpoint).step;
            }
        }

    public:
        NOA_HD void operator()(index_type batch, index_type ox) const noexcept requires (NDIM == 1) {
            auto fftfreq = static_cast<coord_type>(
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[0]));
            fftfreq *= m_frequency_step[0];
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            const auto ctf = get_ctf_value_(fftfreq, batch);

            if constexpr (HAS_INPUT) {
                // Get the input index corresponding to this output index.
                // In the mode, there's no user defined range, so a remap is enough to get the input index.
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[0]);
                m_output(batch, ox) = get_input_value_and_apply_ctf_(ctf, batch, ix);
            } else {
                m_output(batch, ox) = static_cast<output_value_type>(ctf);
            }
        }

        NOA_HD void operator()(index_type batch, index_type oy, index_type ox) const noexcept requires (NDIM == 2) {
            auto fftfreq = coord_nd_type::from_values(
                    noa::fft::index2frequency<IS_DST_CENTERED>(oy, m_shape[0]),
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[1]));
            fftfreq *= m_frequency_step;
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            ctf_value_type ctf;
            if constexpr (IS_ISOTROPIC)
                ctf = get_ctf_value_(norm(fftfreq), batch);
            else // anisotropic
                ctf = get_ctf_value_(fftfreq, batch);

            if constexpr (HAS_INPUT) {
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[0]);
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[1]);
                m_output(batch, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iy, ix);
            } else {
                m_output(batch, oy, ox) = static_cast<output_value_type>(ctf);
            }
        }

        NOA_HD void operator()(index_type batch, index_type oz, index_type oy, index_type ox) const noexcept requires (NDIM == 3) {
            auto fftfreq = coord_nd_type::from_values(
                    noa::fft::index2frequency<IS_DST_CENTERED>(oz, m_shape[0]),
                    noa::fft::index2frequency<IS_DST_CENTERED>(oy, m_shape[1]),
                    IS_RFFT ? ox : noa::fft::index2frequency<IS_DST_CENTERED>(ox, m_shape[2]));
            fftfreq *= m_frequency_step;
            if constexpr (!HAS_INPUT)
                fftfreq += m_frequency_start;

            const auto ctf = get_ctf_value_(norm(fftfreq), batch);

            if constexpr (HAS_INPUT) {
                const auto iz = noa::fft::remap_index<REMAP, true>(oz, m_shape[0]);
                const auto iy = noa::fft::remap_index<REMAP, true>(oy, m_shape[1]);
                const auto ix = IS_RFFT ? ox : noa::fft::remap_index<REMAP, true>(ox, m_shape[2]);
                m_output(batch, oz, oy, ox) = get_input_value_and_apply_ctf_(ctf, batch, iz, iy, ix);
            } else {
                m_output(batch, oz, oy, ox) = static_cast<output_value_type>(ctf);
            }
        }

    private:
        template<typename T>
        NOA_HD constexpr auto get_ctf_value_(T frequency, index_type batch) const noexcept {
            ctf_value_type ctf;
            if constexpr (nt::is_accessor_1d_restrict_contiguous_v<ctf_type>) {
                ctf = m_ctf[batch].value_at(frequency);
            } else {
                ctf = m_ctf.value_at(frequency);
            }
            if (m_ctf_abs)
                ctf = abs(ctf);
            if (m_ctf_squared)
                ctf *= ctf;
            return ctf;
        }

        template<typename... Indexes>
        NOA_HD constexpr auto get_input_value_and_apply_ctf_(ctf_value_type ctf, Indexes... indexes) const noexcept {
            const auto value = m_input(indexes...) * static_cast<input_real_type>(ctf);
            if constexpr (nt::is_complex_v<input_value_type> and nt::is_real_v<output_value_type>)
                return static_cast<output_value_type>(abs_squared(value));
            else
                return static_cast<output_value_type>(value);
        }

    private:
        ctf_type m_ctf;
        NOA_NO_UNIQUE_ADDRESS input_or_empty_type m_input{};
        output_type m_output;
        shape_nd_type m_shape;
        coord_nd_type m_frequency_step;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_frequency_start{};
        bool m_ctf_abs;
        bool m_ctf_squared;
    };
}
