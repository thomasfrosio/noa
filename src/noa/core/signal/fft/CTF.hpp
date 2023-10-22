#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/geometry/Polar.hpp"

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
}

namespace noa::signal::fft {
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
        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type phase_at(Coord fftfreq) const noexcept {
            const auto r1 = static_cast<value_type>(fftfreq) / m_pixel_size;
            const auto r2 = r1 * r1;
            const auto r4 = r2 * r2;
            const auto phase = (m_k1 * r2 * m_defocus_angstroms) + (m_k2 * r4) - m_phase_shift - m_k3;
            return phase;
        }

        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
        [[nodiscard]] NOA_HD constexpr value_type value_at(Coord fftfreq) const noexcept {
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
        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
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

        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
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

        template<typename Coord, typename = std::enable_if_t<std::is_floating_point_v<Coord>>>
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
    template<typename T> struct proclaim_is_ctf : std::false_type {};
    template<typename T> using is_ctf = std::bool_constant<proclaim_is_ctf<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_v = is_ctf<T>::value;
    template<typename... Ts> constexpr bool are_ctf_v = bool_and<is_ctf_v<Ts>...>::value;

    template<typename T> struct proclaim_is_ctf_isotropic : std::false_type {};
    template<typename T> using is_ctf_isotropic = std::bool_constant<proclaim_is_ctf_isotropic<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_isotropic_v = is_ctf_isotropic<T>::value;
    template<typename... Ts> constexpr bool are_ctf_isotropic_v = bool_and<is_ctf_isotropic_v<Ts>...>::value;

    template<typename T> struct proclaim_is_ctf_anisotropic : std::false_type {};
    template<typename T> using is_ctf_anisotropic = std::bool_constant<proclaim_is_ctf_anisotropic<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_ctf_anisotropic_v = is_ctf_anisotropic<T>::value;
    template<typename... Ts> constexpr bool are_ctf_anisotropic_v = bool_and<is_ctf_anisotropic_v<Ts>...>::value;

    template<typename T> constexpr bool is_ctf_f32_v = is_ctf_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_f64_v = is_ctf_v<T> && std::is_same_v<value_type_t<T>, double>;
    template<typename T> constexpr bool is_ctf_isotropic_f32_v = is_ctf_isotropic_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_isotropic_f64_v = is_ctf_isotropic_v<T> && std::is_same_v<value_type_t<T>, double>;
    template<typename T> constexpr bool is_ctf_anisotropic_f32_v = is_ctf_anisotropic_v<T> && std::is_same_v<value_type_t<T>, float>;
    template<typename T> constexpr bool is_ctf_anisotropic_f64_v = is_ctf_anisotropic_v<T> && std::is_same_v<value_type_t<T>, double>;

    template<typename T> struct proclaim_is_ctf<noa::signal::fft::CTFIsotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf<noa::signal::fft::CTFAnisotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf_isotropic<noa::signal::fft::CTFIsotropic<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ctf_anisotropic<noa::signal::fft::CTFAnisotropic<T>> : std::true_type {};
}
