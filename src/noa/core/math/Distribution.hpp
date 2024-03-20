#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::traits {
    template<typename> struct proclaim_is_distribution : std::false_type {};
    template<typename T> using is_distribution = std::bool_constant<proclaim_is_distribution<T>::value>;
    template<typename T> constexpr bool is_distribution_v = is_distribution<std::decay_t<T>>::value;
}

namespace noa {
    struct Distribution {};

    /// Uniform distribution.
    template<typename T> requires nt::is_numeric_v<T>
    class Uniform : public Distribution {
    public:
        using result_type = T;
        using compute_type = std::conditional_t<nt::is_int_v<T>, f64, result_type>;

        constexpr explicit Uniform(result_type min, result_type max)
                : m_min(static_cast<compute_type>(min)),
                  m_max(static_cast<compute_type>(max)) {}

        constexpr result_type operator()(auto& generator) const noexcept requires nt::is_int_v<result_type> {
            const auto uniform_real = generator.template next<compute_type>(); // [0,1]
            const auto uniform_int = round(uniform_real * (m_max - m_min) + m_min);
            return static_cast<result_type>(uniform_int);
        }

        constexpr result_type operator()(auto& generator) const noexcept requires nt::is_real_v<result_type> {
            auto uniform_real = generator.template next<compute_type>(); // [0,1]
            return uniform_real * (m_max - m_min) + m_min;
        }

        constexpr result_type operator()(auto& generator) const noexcept requires nt::is_complex_v<result_type> {
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
    template<typename T> requires nt::is_real_v<T>
    class Normal : public Distribution {
    public:
        using result_type = T;

        constexpr explicit Normal(result_type mean = 0, result_type stddev = 1)
                : m_mean(mean), m_stddev(stddev) {}

        constexpr result_type operator()(auto& generator) noexcept {
            result_type real = next_(generator, m_saved, m_saved_is_available);
            return real * m_stddev + m_mean;
        }

    private:
        static constexpr result_type next_(auto& generator, result_type& saved, bool& is_available) noexcept {
            result_type ret;
            if (is_available) {
                is_available = false;
                ret = saved;
            } else {
                result_type x, y, r2;
                do {
                    x = 2 * generator.template next<result_type>() - 1;
                    y = 2 * generator.template next<result_type>() - 1;
                    r2 = x * x + y * y;
                } while (r2 > 1 || r2 == 0);

                const result_type mult = sqrt(-2 * log(r2) / r2);
                saved = x * mult;
                is_available = true;
                ret = y * mult;
            }
            return ret;
        }

    private:
        result_type m_mean{};
        result_type m_saved{};
        result_type m_stddev{};
        bool m_saved_is_available{};

        template<typename U> requires nt::is_real_v<U>
        friend class LogNormal;
    };

    /// Lognormal distribution.
    template<typename T> requires nt::is_real_v<T>
    class LogNormal : public Distribution {
    public:
        using result_type = T;

        constexpr explicit LogNormal(result_type mean = {}, result_type stddev = 1)
                : m_mean(mean), m_stddev(stddev) {}

        constexpr result_type operator()(auto& generator) noexcept requires nt::is_real_v<result_type> {
            result_type real = Normal<result_type>::next_(generator, m_saved, m_saved_is_available);
            return exp(real * m_stddev + m_mean);
        }

    private:
        result_type m_mean{};
        result_type m_saved{};
        result_type m_stddev{};
        bool m_saved_is_available{};
    };

    /// Poisson distribution.
    template<typename T> requires nt::is_int_v<T>
    class Poisson : public Distribution {
    public:
        using result_type = T;
        using compute_type = f64;

        constexpr explicit Poisson(T mean) : m_mean(static_cast<compute_type>(mean)), m_threshold(std::exp(-m_mean)) {}

        constexpr compute_type next(auto& generator) noexcept {
            T x = 0;
            compute_type prod = 1.0;
            do {
                prod *= generator.template next<compute_type>();
                x += 1;
            } while (prod > m_threshold);
            return static_cast<result_type>(x - 1);
        }

    private:
        compute_type m_mean;
        compute_type m_threshold;
    };
}

namespace noa::traits {
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

        constexpr explicit RandomBitsGenerator(u64 seed) {
            auto splitmix64 = [](u64& x) -> u64 {
                u64 z = (x += 0x9e3779b97f4a7c15uLL);
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9uLL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebuLL;
                return z ^ (z >> 31);
            };
            s[0] = splitmix64(seed);
            s[1] = splitmix64(s[0]);
            s[2] = splitmix64(s[1]);
            s[3] = splitmix64(s[2]);
        }

        constexpr u64 operator()() {
            return next_ss_();
        }

        template<typename T>
        constexpr auto next() {
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
                static_assert(nt::always_false_v<T>);
            }
        }

        static constexpr u64 min() { return 0; }
        static constexpr u64 max() { return u64(-1); }

    private:
        // https://prng.di.unimi.it/xoshiro256starstar.c
        constexpr u64 next_ss_() {
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
        constexpr u64 next_p_() {
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
        using allow_vectorization = bool;

        constexpr explicit Randomizer(const Distribution& distribution, u64 seed)
                : m_distribution(distribution), random_seed(seed) {}

        constexpr void init(auto uid) {
            m_generator = RandomBitsGenerator(random_seed + static_cast<u64>(uid) + 1);
        }

        template<typename T>
        constexpr auto operator()(T& output) noexcept {
            output = static_cast<T>(m_distribution(m_generator));
        }

    private:
        RandomBitsGenerator m_generator;
        Distribution m_distribution;
        u64 random_seed{};
    };
}
