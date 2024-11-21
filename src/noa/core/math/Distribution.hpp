#pragma once

#include <random>

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    /// Uniform distribution.
    template<nt::numeric T>
    class Uniform {
    public:
        using value_type = T;
        using real_type = nt::value_type_t<value_type>;
        using compute_type = std::conditional_t<nt::integer<T>, f64, value_type>;

        constexpr explicit Uniform(value_type min, value_type max) noexcept :
            m_min(static_cast<compute_type>(min)),
            m_max(static_cast<compute_type>(max)) {}

        constexpr explicit Uniform(real_type min, real_type max) noexcept requires nt::complex<value_type> :
            m_min{min, min},
            m_max{max, max} {}

        constexpr auto operator()(auto& generator) const noexcept -> value_type requires nt::integer<value_type> {
            const auto uniform_real = generator.template next<compute_type>(); // [0,1]
            const auto uniform_int = round(uniform_real * (m_max - m_min) + m_min);
            return static_cast<value_type>(uniform_int);
        }

        constexpr auto operator()(auto& generator) const noexcept -> value_type requires nt::real<value_type> {
            auto uniform_real = generator.template next<compute_type>(); // [0,1]
            return uniform_real * (m_max - m_min) + m_min;
        }

        constexpr auto operator()(auto& generator) const noexcept -> value_type requires nt::complex<value_type> {
            using real_t = nt::value_type_t<compute_type>;
            auto uniform_real = generator.template next<real_t>(); // [0,1]
            auto uniform_imag = generator.template next<real_t>(); // [0,1]
            return {.real=uniform_real * (m_max.real - m_min.real) + m_min.real,
                    .imag=uniform_imag * (m_max.imag - m_min.imag) + m_min.imag};
        }

    private:
        compute_type m_min;
        compute_type m_max;
    };

    /// Normal distribution.
    template<nt::real T>
    class Normal {
    public:
        using value_type = T;

        constexpr explicit Normal(value_type mean = 0, value_type stddev = 1) noexcept :
            m_mean(mean), m_stddev(stddev) {}

        constexpr auto operator()(auto& generator) noexcept -> value_type {
            value_type real = next_(generator, m_saved, m_saved_is_available);
            return real * m_stddev + m_mean;
        }

    private:
        static constexpr auto next_(auto& generator, value_type& saved, bool& is_available) noexcept -> value_type {
            value_type ret;
            if (is_available) {
                is_available = false;
                ret = saved;
            } else {
                value_type x, y, r2;
                do {
                    x = 2 * generator.template next<value_type>() - 1;
                    y = 2 * generator.template next<value_type>() - 1;
                    r2 = x * x + y * y;
                } while (r2 > 1 || r2 == 0);

                const value_type mult = sqrt(-2 * log(r2) / r2);
                saved = x * mult;
                is_available = true;
                ret = y * mult;
            }
            return ret;
        }

    private:
        value_type m_mean{};
        value_type m_saved{};
        value_type m_stddev{};
        bool m_saved_is_available{};

        template<nt::real U>
        friend class LogNormal;
    };

    /// Lognormal distribution.
    template<nt::real T>
    class LogNormal {
    public:
        using value_type = T;

        constexpr explicit LogNormal(value_type mean = {}, value_type stddev = 1) noexcept :
            m_mean(mean), m_stddev(stddev) {}

        constexpr auto operator()(auto& generator) noexcept -> value_type requires nt::real<value_type> {
            value_type real = Normal<value_type>::next_(generator, m_saved, m_saved_is_available);
            return exp(real * m_stddev + m_mean);
        }

    private:
        value_type m_mean{};
        value_type m_saved{};
        value_type m_stddev{};
        bool m_saved_is_available{};
    };

    /// Poisson distribution.
    template<nt::integer T>
    class Poisson {
    public:
        using value_type = T;
        using compute_type = f64;

        constexpr explicit Poisson(T mean) noexcept :
            m_mean(static_cast<compute_type>(mean)), m_threshold(std::exp(-m_mean)) {}

        constexpr auto next(auto& generator) noexcept -> compute_type {
            T x = 0;
            compute_type prod = 1.0;
            do {
                prod *= generator.template next<compute_type>();
                x += 1;
            } while (prod > m_threshold);
            return static_cast<value_type>(x - 1);
        }

    private:
        compute_type m_mean;
        compute_type m_threshold;
    };
}

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(distribution);
    template<typename T> struct proclaim_is_distribution<Uniform<T>> : std::true_type {};
    template<typename T> struct proclaim_is_distribution<Normal<T>> : std::true_type {};
    template<typename T> struct proclaim_is_distribution<LogNormal<T>> : std::true_type {};
    template<typename T> struct proclaim_is_distribution<Poisson<T>> : std::true_type {};
}

namespace noa {
    /// xoshiro256** and xoshiro256+ 64-bit pseudorandom number generators.
    class RandomBitsGenerator {
    public:
        constexpr RandomBitsGenerator() = default;

        NOA_HD constexpr explicit RandomBitsGenerator(u64 seed) {
            auto splitmix64 = [](u64& x) -> u64 {
                u64 z = (x += 0x9e3779b97f4a7c15ull);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                return z ^ (z >> 31);
            };
            s[0] = splitmix64(seed);
            s[1] = splitmix64(s[0]);
            s[2] = splitmix64(s[1]);
            s[3] = splitmix64(s[2]);
        }

        constexpr auto operator()() -> u64 {
            return next_ss_();
        }

        template<typename T>
        NOA_HD constexpr auto next() noexcept {
            if constexpr (std::is_same_v<T, u16>) {
                return static_cast<u16>(next_ss_() >> 48);
            } else if constexpr (std::is_same_v<T, u32>) {
                return static_cast<u32>(next_ss_() >> 32);
            } else if constexpr (std::is_same_v<T, u64>) {
                return next_ss_();
            } else if constexpr (std::is_same_v<T, f32>) {
                // Fill the mantissa with random bits and normalise in range [0,1)
                return static_cast<f32>(next_p_() >> 40) / static_cast<f32>(1ul << 24);
            } else if constexpr (std::is_same_v<T, f64>) {
                // Fill the mantissa with random bits and normalise in range [0,1)
                return static_cast<f64>(next_p_() >> 11) / static_cast<f64>(1ul << 53);
            } else {
                // Getting a signed integer is more complicated, I think,
                // and since I'm not familiar with this subject, don't support
                // this case and use a f64 to compute the distribution and then
                // round (which is what curand suggests doing anyway).
                static_assert(nt::always_false<T>);
            }
        }

        NOA_HD static constexpr auto min() noexcept -> u64 { return 0; }
        NOA_HD static constexpr auto max() noexcept -> u64 { return u64(-1); }

    private:
        // https://prng.di.unimi.it/xoshiro256starstar.c
        NOA_HD constexpr auto next_ss_() noexcept -> u64 {
            auto rotl = [](u64 x, int k) -> u64 {
                return (x << k) | (x >> (64 - k));
            };
            const u64 result = rotl(s[1] * 5, 7) * 9;
            const u64 t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return result;
        }

        // https://prng.di.unimi.it/xoshiro256plus.c
        NOA_HD constexpr auto next_p_() noexcept -> u64 {
            auto rotl = [](u64 x, int k) -> u64 {
                return (x << k) | (x >> (64 - k));
            };
            const u64 result = s[0] + s[3];
            const u64 t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return result;
        }

    private:
        u64 s[4]{};
    };

    /// Element-wise randomize operator.
    template<typename Distribution>
    struct Randomizer {
        using enable_vectorization = bool;

        constexpr explicit Randomizer(const Distribution& distribution, u64 seed = 0) noexcept :
            m_distribution(distribution), random_seed(seed) {}

        constexpr void init(auto uid) noexcept {
            m_generator = RandomBitsGenerator(random_seed + static_cast<u64>(uid) + 1);
        }

        template<typename T>
        constexpr auto operator()(T& output) noexcept {
            if constexpr (nt::complex<T> and not nt::complex<nt::value_type_t<Distribution>>) {
                output = static_cast<T::value_type>(m_distribution(m_generator));
            } else {
                output = static_cast<T>(m_distribution(m_generator));
            }
        }

        constexpr auto operator()() noexcept {
            return m_distribution(m_generator);
        }

    private:
        RandomBitsGenerator m_generator;
        Distribution m_distribution;
        u64 random_seed{};
    };

    /// Returns a random value generator.
    /// Use the call operator to get a value.
    template<nt::distribution T>
    [[nodiscard]] auto random_generator(const T& distribution) noexcept -> Randomizer<T> {
        Randomizer randomizer(distribution);
        randomizer.init(std::random_device{}());
        return randomizer;
    }

    /// Returns a random value.
    template<nt::distribution T>
    [[nodiscard]] auto random_value(const T& distribution) noexcept -> T::value_type {
        return random_generator(distribution)();
    }
}
