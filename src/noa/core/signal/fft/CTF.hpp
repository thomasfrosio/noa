#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/geometry/Polar.hpp"

namespace noa::signal {
    /// Calculate the relativistic electron wavelength in meters given an acceleration potential in volts.
    /// \see https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory
    [[nodiscard]] NOA_IHD f64 relativistic_electron_wavelength(f64 energy) noexcept {
        constexpr auto h = noa::math::Constant<f64>::PLANCK;
        constexpr auto c = noa::math::Constant<f64>::SPEED_OF_LIGHT;
        constexpr auto m0 = noa::math::Constant<f64>::ELECTRON_MASS;
        constexpr auto e = noa::math::Constant<f64>::ELEMENTARY_CHARGE;
        const auto V = energy;
        const auto eV = e * V;

        constexpr auto numerator = h * c;
        const auto denominator = noa::math::sqrt(eV * (2 * m0 * c * c + eV));
        return numerator / denominator;
    }
}

namespace noa::signal::fft {
    // TODO Anisotropic b-factor
    // TODO Pixel angle?

    template<typename Real>
    class CTFIsotropic {
    public:
        static_assert(std::is_floating_point_v<Real>);
        using value_type = Real;

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
        NOA_HD constexpr CTFIsotropic(
                value_type pixel_size,
                value_type defocus,
                value_type voltage,
                value_type amplitude,
                value_type cs,
                value_type phase_shift,
                value_type bfactor) noexcept
                : m_pixel_size(pixel_size),
                  m_defocus_angstroms(-defocus * static_cast<Real>(1e4)), // micrometers -> angstroms
                  m_phase_shift(phase_shift),
                  m_cs_angstroms(cs * static_cast<Real>(1e7)), // mm -> angstroms
                  m_voltage_volts(voltage * static_cast<Real>(1e3)), // kV -> V
                  m_amplitude(amplitude),
                  m_bfactor(bfactor) {
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

    public: // getters
        [[nodiscard]] NOA_HD constexpr value_type pixel_size() const noexcept { return m_pixel_size; }
        [[nodiscard]] NOA_HD constexpr value_type phase_shift() const noexcept { return m_phase_shift; }
        [[nodiscard]] NOA_HD constexpr value_type amplitude() const noexcept { return m_amplitude; }
        [[nodiscard]] NOA_HD constexpr value_type bfactor() const noexcept { return m_bfactor; }
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
        NOA_HD constexpr void set_bfactor(value_type bfactor) noexcept { m_bfactor = bfactor; }

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
        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type phase_at(Coord frequency) const noexcept {
            const auto r1 = static_cast<value_type>(frequency) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            return phase;
        }

        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type value_at(Coord frequency) const noexcept {
            const auto r1 = static_cast<value_type>(frequency) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            auto ctf = -noa::math::sin(phase);
            if (m_bfactor != 0)
                ctf *= noa::math::exp((m_bfactor / 4) * r2);
            return ctf;
        }

    private:
        NOA_HD void set_lambda_and_cs_() noexcept {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = noa::math::Constant<f64>::PI;
            m_k1 = PI * lambda;
            m_k2 = PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda;
        }

        NOA_HD constexpr void set_amplitude_fraction_() noexcept {
            m_k3 = noa::math::atan(m_amplitude / noa::math::sqrt(1 - m_amplitude * m_amplitude));
        }

    private:
        value_type m_pixel_size{};
        value_type m_defocus_angstroms{};
        value_type m_phase_shift{};
        value_type m_cs_angstroms{};
        value_type m_voltage_volts{};
        value_type m_amplitude{};
        value_type m_bfactor{};

        value_type m_k1;
        value_type m_k2;
        value_type m_k3;
    };

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
        constexpr CTFAnisotropic() = default;

        /// Create an anisotropic CTF.
        /// \param pixel_size   HW pixel size in A/p.
        /// \param defocus      .defocus: defocus in μm, positive is underfocused.
        ///                     .astigmatism: amount of astigmatism in μm. `(defocus_u - defocus_v) / 2`
        ///                     .angle: angle of astigmatism in radians. 0 places `defocus_u` along the y-axis
        /// \param voltage      Acceleration voltage in kV.
        /// \param amplitude    Fraction of amplitude contrast (value in range [0, 1]).
        /// \param cs           Spherical aberration in mm.
        /// \param phase_shift  Angle of phase shift applied to CTF in radians.
        /// \param bfactor      B-factor in A^2, negative is decay.
        NOA_HD constexpr CTFAnisotropic(
                pixel_size_type pixel_size,
                defocus_type defocus,
                value_type voltage,
                value_type amplitude,
                value_type cs,
                value_type phase_shift,
                value_type bfactor) noexcept
                : m_pixel_size(pixel_size),
                  m_phase_shift(phase_shift),
                  m_cs_angstroms(cs * static_cast<Real>(1e7)), // mm -> angstroms
                  m_voltage_volts(voltage * static_cast<Real>(1e3)), // kV -> V
                  m_amplitude(amplitude),
                  m_bfactor(bfactor) {
            set_defocus(defocus);
            set_lambda_and_cs_();
            set_amplitude_fraction_();
        }

    public: // getters
        [[nodiscard]] NOA_HD constexpr pixel_size_type pixel_size() const noexcept { return m_pixel_size; }
        [[nodiscard]] NOA_HD constexpr value_type phase_shift() const noexcept { return m_phase_shift; }
        [[nodiscard]] NOA_HD constexpr value_type amplitude() const noexcept { return m_amplitude; }
        [[nodiscard]] NOA_HD constexpr value_type bfactor() const noexcept { return m_bfactor; }
        [[nodiscard]] NOA_HD constexpr defocus_type defocus() const noexcept {
            return {-m_defocus_angstroms.value * static_cast<Real>(1e-4), // angstroms -> micrometers
                    m_defocus_angstroms.astigmatism * static_cast<Real>(1e-4), // angstroms -> micrometers
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
        NOA_HD constexpr void set_bfactor(value_type bfactor) noexcept { m_bfactor = bfactor; }

        NOA_HD constexpr void set_defocus(defocus_type defocus) noexcept {
            m_defocus_angstroms.value = -defocus.value * static_cast<Real>(1e4); // micrometers -> angstroms
            m_defocus_angstroms.astigmatism = defocus.astigmatism * static_cast<Real>(1e4); // micrometers -> angstroms
            m_defocus_angstroms = defocus.angle;
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
        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type phase_at(Vec2<Coord> frequency) const noexcept {
            const auto scaled_frequency = frequency.template as<value_type>() / m_pixel_size;
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_frequency);
            const auto rho = noa::geometry::cartesian2rho(scaled_frequency);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;

            const auto phase = m_k1 * r2 * phi2defocus_(phi) + m_k2 * r4 - m_phase_shift - m_k3;
            return phase;
        }

        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type value_at(Vec2<Coord> frequency) const noexcept {
            const auto scaled_frequency = frequency.template as<value_type>() / m_pixel_size;
            const auto phi = noa::geometry::cartesian2phi<false>(scaled_frequency);
            const auto rho = noa::geometry::cartesian2rho(scaled_frequency);

            const auto r1 = rho;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;

            const auto phase = m_k1 * r2 * phi2defocus_(phi) + m_k2 * r4 - m_phase_shift - m_k3;
            auto ctf = -noa::math::sin(phase);
            if (m_bfactor != 0)
                ctf *= noa::math::exp((m_bfactor / 4) * r2);
            return ctf;
        }

    private:
        [[nodiscard]] NOA_HD constexpr value_type phi2defocus_(value_type phi) const noexcept {
            const auto ellipse_angle = (phi - m_defocus_angstroms.angle);
            return m_defocus_angstroms.value + m_defocus_angstroms.astigmatism * noa::math::cos(2 * ellipse_angle);
        }

        NOA_HD void set_lambda_and_cs_() noexcept {
            const auto voltage = static_cast<f64>(m_voltage_volts);
            const auto lambda = relativistic_electron_wavelength(voltage) * 1e10; // angstroms
            constexpr f64 PI = noa::math::Constant<f64>::PI;
            m_k1 = PI * lambda;
            m_k2 = PI * 0.5 * m_cs_angstroms * lambda * lambda * lambda;
        }

        NOA_HD constexpr void set_amplitude_fraction_() noexcept {
            m_k3 = noa::math::atan(m_amplitude / noa::math::sqrt(1 - m_amplitude * m_amplitude));
        }

    private:
        pixel_size_type m_pixel_size{};
        defocus_type m_defocus_angstroms{};
        value_type m_phase_shift{};
        value_type m_cs_angstroms{};
        value_type m_voltage_volts{};
        value_type m_amplitude{};
        value_type m_bfactor{};

        value_type m_k1;
        value_type m_k2;
        value_type m_k3;
    };
}
