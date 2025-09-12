#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"
#include "noa/core/geometry/Polar.hpp"
#include "noa/core/types/Vec.hpp"

namespace noa::signal {
    /// Calculate the relativistic electron wavelength in meters given an acceleration potential in volts.
    /// \see https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory
    [[nodiscard]] NOA_IHD auto relativistic_electron_wavelength(f64 energy) noexcept -> f64 {
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

            [[nodiscard]] constexpr auto to_ctf() noexcept { return CTFIsotropic(*this); }
        };

    public: // constructors
        constexpr CTFIsotropic() noexcept = default;

        /// Create an isotropic CTF.
        /// \param pixel_size   Pixel size in A/p.
        /// \param defocus      Defocus in μm, positive is underfocused.
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        /// \param scale        Post-processing scaling-factor.
        constexpr explicit CTFIsotropic(
            value_type pixel_size,
            value_type defocus,
            value_type voltage,
            value_type amplitude,
            value_type cs,
            value_type phase_shift,
            value_type bfactor,
            value_type scale
        ) noexcept :
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

        constexpr explicit CTFIsotropic(const Parameters& parameters) noexcept : CTFIsotropic(
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
        ) noexcept : CTFIsotropic(
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
        [[nodiscard]] constexpr auto pixel_size() const -> value_type { return m_pixel_size; }
        [[nodiscard]] constexpr auto sampling_rate() const -> value_type { return pixel_size(); }
        [[nodiscard]] constexpr auto phase_shift() const -> value_type { return m_phase_shift; }
        [[nodiscard]] constexpr auto amplitude() const -> value_type { return m_amplitude; }
        [[nodiscard]] constexpr auto scale() const -> value_type { return m_scale; }
        [[nodiscard]] constexpr auto bfactor() const -> value_type {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] constexpr auto defocus() const -> value_type {
            return -m_defocus_angstroms * static_cast<value_type>(1e-4); // angstrom -> micrometers
        }
        [[nodiscard]] constexpr auto cs() const -> value_type {
            return m_cs_angstroms * static_cast<value_type>(1e-7); // angstrom -> millimeters
        }
        [[nodiscard]] constexpr auto voltage() const -> value_type {
            return m_voltage_volts * static_cast<value_type>(1e-3); // volts -> kilovolts
        }
        [[nodiscard]] constexpr auto wavelength() const -> value_type {
            return m_k1 / Constant<value_type>::PI; // k1 = wavelength * PI, in angstrom
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

        // Access private members for the as<U>() function.
        friend CTFIsotropic<f32>;
        friend CTFIsotropic<f64>;

        template<nt::any_of<f32, f64> U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            if constexpr (std::same_as<U, value_type>) {
                return *this;
            } else {
                CTFIsotropic<U> out;
                out.m_pixel_size = static_cast<U>(m_pixel_size);
                out.m_defocus_angstroms = static_cast<U>(m_defocus_angstroms);
                out.m_phase_shift = static_cast<U>(m_phase_shift);
                out.m_cs_angstroms = static_cast<U>(m_cs_angstroms);
                out.m_voltage_volts = static_cast<U>(m_voltage_volts);
                out.m_amplitude = static_cast<U>(m_amplitude);
                out.m_bfactor_forth = static_cast<U>(m_bfactor_forth);
                out.m_scale = static_cast<U>(m_scale);
                out.m_k1 = static_cast<U>(m_k1);
                out.m_k2 = static_cast<U>(m_k2);
                out.m_k3 = static_cast<U>(m_k3);
                return out;
            }
        }

    public:
        [[nodiscard]] constexpr auto phase_at(nt::real auto fftfreq) const -> value_type {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            return phase;
        }

        [[nodiscard]] constexpr auto value_at(nt::real auto fftfreq) const -> value_type {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            auto ctf = -sin(phase);
            if (m_bfactor_forth != 0)
                ctf *= exp(m_bfactor_forth * r2);
            return ctf * m_scale;
        }

        [[nodiscard]] constexpr auto envelope_at(nt::real auto fftfreq) const -> value_type {
            if (m_bfactor_forth == 0)
                return m_scale;
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            return exp(m_bfactor_forth * r2) * m_scale;
        }

        [[nodiscard]] constexpr auto fftfreq_at(nt::real auto phase) const -> value_type {
            // Floating-point precision errors are a real thing, switch everything to double precision.
            const auto a = static_cast<f64>(m_k2);
            const auto b = static_cast<f64>(m_k1) * static_cast<f64>(m_defocus_angstroms);
            const auto c = -static_cast<f64>(phase) - static_cast<f64>(m_phase_shift) - static_cast<f64>(m_k3);
            const auto d = b * b - 4 * a * c;

            const auto numerator = -b - sqrt(d);
            const auto denominator = 2 * a;
            auto fftfreq = sqrt(abs(numerator / denominator));
            fftfreq *= static_cast<f64>(m_pixel_size);

            return static_cast<value_type>(fftfreq);
        }

    private:
        NOA_HD void set_lambda_and_cs_() {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = static_cast<value_type>(PI * lambda);
            m_k2 = static_cast<value_type>(PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda);
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
        using pixel_size_type = Vec<value_type, 2>;
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

            [[nodiscard]] constexpr auto to_ctf() noexcept { return CTFAnisotropic(*this); }
        };

        [[nodiscard]] static constexpr auto from_isotropic_ctf(
            const CTFIsotropic<value_type>& ctf_isotropic,
            value_type defocus_astigmatic_value = 0,
            value_type defocus_astigmatic_angle = 0
        ) -> CTFAnisotropic {
            return CTFAnisotropic(ctf_isotropic, defocus_astigmatic_value, defocus_astigmatic_angle);
        }

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
        constexpr explicit CTFAnisotropic(
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
        [[nodiscard]] constexpr auto pixel_size() const -> pixel_size_type { return m_pixel_size; }
        [[nodiscard]] constexpr auto sampling_rate() const -> pixel_size_type { return pixel_size(); }
        [[nodiscard]] constexpr auto phase_shift() const -> value_type { return m_phase_shift; }
        [[nodiscard]] constexpr auto amplitude() const -> value_type { return m_amplitude; }
        [[nodiscard]] constexpr auto scale() const -> value_type { return m_scale; }
        [[nodiscard]] constexpr auto bfactor() const -> value_type {
            return m_bfactor_forth * 4;
        }
        [[nodiscard]] constexpr auto defocus() const -> defocus_type {
            return {-m_defocus_angstroms.value * static_cast<value_type>(1e-4), // angstroms -> micrometers
                    -m_defocus_angstroms.astigmatism * static_cast<value_type>(1e-4), // angstroms -> micrometers
                    m_defocus_angstroms.angle};
        }
        [[nodiscard]] constexpr auto cs() const -> value_type {
            return m_cs_angstroms * static_cast<value_type>(1e-7); // angstroms -> millimeters
        }
        [[nodiscard]] constexpr auto voltage() const -> value_type {
            return m_voltage_volts * static_cast<value_type>(1e-3); // volts -> kilovolts
        }
        [[nodiscard]] constexpr auto wavelength() const -> value_type {
            return m_k1 / Constant<value_type>::PI; // k1 = wavelength * PI, in angstrom
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

        // Access private members for the as<U>() function.
        friend CTFAnisotropic<f32>;
        friend CTFAnisotropic<f64>;

        template<nt::any_of<f32, f64> U>
        [[nodiscard]] NOA_HD constexpr auto as() const noexcept {
            if constexpr (std::same_as<U, value_type>) {
                return *this;
            } else {
                CTFAnisotropic<U> out;
                out.m_pixel_size = m_pixel_size.template as<U>();
                out.m_defocus_angstroms = {
                    static_cast<U>(m_defocus_angstroms.value),
                    static_cast<U>(m_defocus_angstroms.astigmatism),
                    static_cast<U>(m_defocus_angstroms.angle)
                };
                out.m_phase_shift = static_cast<U>(m_phase_shift);
                out.m_cs_angstroms = static_cast<U>(m_cs_angstroms);
                out.m_voltage_volts = static_cast<U>(m_voltage_volts);
                out.m_amplitude = static_cast<U>(m_amplitude);
                out.m_bfactor_forth = static_cast<U>(m_bfactor_forth);
                out.m_scale = static_cast<U>(m_scale);
                out.m_k1 = static_cast<U>(m_k1);
                out.m_k2 = static_cast<U>(m_k2);
                out.m_k3 = static_cast<U>(m_k3);
                return out;
            }
        }

    public:
        template<nt::real Coord>
        [[nodiscard]] constexpr auto phase_at(const Vec<Coord, 2>& fftfreq) const -> value_type {
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
        [[nodiscard]] constexpr auto value_at(const Vec<Coord, 2>& fftfreq) const -> value_type {
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
        [[nodiscard]] constexpr auto envelope_at(const Vec<Coord, 2>& fftfreq) const -> value_type {
            if (m_bfactor_forth == 0)
                return m_scale;
            const auto scaled_fftfreq = fftfreq.template as<value_type>() / m_pixel_size;
            const auto rho = noa::geometry::cartesian2rho(scaled_fftfreq);
            const auto r1 = rho;
            const auto r2 = r1 * r1;
            return exp(m_bfactor_forth * r2) * m_scale;
        }

        template<nt::real Coord>
        [[nodiscard]] constexpr auto isotropic_fftfreq(const Vec2<Coord>& fftfreq_2d) const -> value_type {
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
            const auto l1 = wavelength();
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

        [[nodiscard]] constexpr auto defocus_at(nt::real auto phi) const -> value_type {
            return -phi2defocus_(static_cast<value_type>(phi)) * static_cast<value_type>(1e-4);
        }

    private:
        [[nodiscard]] constexpr auto phi2defocus_(value_type phi) const -> value_type {
            const auto ellipse_angle = (phi - m_defocus_angstroms.angle);
            return m_defocus_angstroms.value + m_defocus_angstroms.astigmatism * cos(2 * ellipse_angle);
        }

        NOA_HD void set_lambda_and_cs_() {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = Constant<f64>::PI;
            m_k1 = static_cast<value_type>(PI * lambda);
            m_k2 = static_cast<value_type>(PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda);
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
