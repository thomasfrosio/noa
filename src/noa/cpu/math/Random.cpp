#include <random>
#include "noa/cpu/math/Random.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

// TODO Add MKL support

namespace {
    // The "xoshiro256** 1.0" generator.
    // C++ port by Arthur O'Dwyer (2021).
    // Based on the C version by David Blackman and Sebastiano Vigna (2018),
    // https://prng.di.unimi.it/xoshiro256starstar.c
    struct xoshiro256ss {
        uint64_t s[4] {};

        static constexpr uint64_t splitmix64(uint64_t& x) {
            uint64_t z = (x += 0x9e3779b97f4a7c15uLL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9uLL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebuLL;
            return z ^ (z >> 31);
        }

        constexpr xoshiro256ss() = default;

        constexpr explicit xoshiro256ss(uint64_t seed) {
            s[0] = splitmix64(seed);
            s[1] = splitmix64(seed);
            s[2] = splitmix64(seed);
            s[3] = splitmix64(seed);
        }

        using result_type = uint64_t;
        static constexpr uint64_t min() { return 0; }
        static constexpr uint64_t max() { return uint64_t(-1); }

        static constexpr uint64_t rotl(uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        constexpr uint64_t operator()() {
            const uint64_t result = rotl(s[1] * 5, 7) * 9;
            const uint64_t t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return result;
        }
    };

    template<typename T>
    using supported_float = std::conditional_t<std::is_same_v<T, noa::f16>, noa::f32, T>;

    template<typename T>
    using uniform_distributor_t =
            std::conditional_t<std::is_integral_v<T>,
                               std::uniform_int_distribution<T>,
                               std::uniform_real_distribution<supported_float<T>>>;

    template<typename T>
    using normal_distributor_t =
            std::conditional_t<std::is_integral_v<T>,
                               std::normal_distribution<noa::f32>,
                               std::normal_distribution<supported_float<T>>>;

    template<typename T>
    using lognormal_distributor_t =
            std::conditional_t<std::is_integral_v<T>,
                               std::lognormal_distribution<noa::f32>,
                               std::lognormal_distribution<supported_float<T>>>;

    template<typename T>
    using poisson_distributor_t =
            std::conditional_t<std::is_integral_v<T>,
                               std::poisson_distribution<T>,
                               std::poisson_distribution<int64_t>>;


    template<typename Distributor>
    struct RandomOperator {
        explicit RandomOperator(Distributor distributor)
                : m_distributor(distributor) {}

        template<typename T>
        auto operator()(T) noexcept {
            return m_distributor(m_generator);
        }

        void initialize(noa::i64 thread_id) {
            m_generator = xoshiro256ss(std::random_device{}() + static_cast<size_t>(thread_id) + 1);
        }

    private:
        xoshiro256ss m_generator{};
        Distributor m_distributor;
    };

    template<typename Distributor>
    struct RandomOperatorComplex {
        explicit RandomOperatorComplex(
                Distributor distributor_real,
                Distributor distributor_complex)
                : m_distributor_real(distributor_real),
                  m_distributor_complex(distributor_complex) {}

        template<typename T>
        auto operator()(T) noexcept {
            using result_t = typename Distributor::result_type;
            using output_t = noa::Complex<result_t>;
            return output_t{m_distributor_real(m_generator),
                            m_distributor_complex(m_generator)};
        }

        void initialize(noa::i64 thread_id) {
            m_generator = xoshiro256ss(std::random_device{}() + static_cast<size_t>(thread_id) + 1);
        }

    private:
        xoshiro256ss m_generator{};
        Distributor m_distributor_real;
        Distributor m_distributor_complex;
    };
}

namespace noa::cpu::math {
    template<typename T, typename U, typename>
    void randomize(noa::math::uniform_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U min, U max, i64 threads) {
        NOA_ASSERT(output && all(shape > 0));

        if constexpr (nt::is_complex_v<T>) {
            if constexpr (nt::is_real_v<U>) {
                using real_t = nt::value_type_t<T>;
                using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, f32, real_t>;
                const auto reinterpreted = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                randomize(noa::math::uniform_t{}, reinterpret_cast<real_t*>(output),
                          reinterpreted.strides, reinterpreted.shape,
                          static_cast<supported_float>(min), static_cast<supported_float>(max), threads);
            } else {
                using real_t = nt::value_type_t<T>;
                using distributor_t = uniform_distributor_t<real_t>;
                using result_t = typename distributor_t::result_type;
                auto min_ = static_cast<result_t>(min.real);
                auto max_ = static_cast<result_t>(max.real);
                const distributor_t distributor_real(min_, max_);
                min_ = static_cast<result_t>(min.imag);
                max_ = static_cast<result_t>(max.imag);
                const distributor_t distributor_imag(min_, max_);

                RandomOperatorComplex randomizer(distributor_real, distributor_imag);
                static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
                noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
            }
        } else {
            using distributor_t = uniform_distributor_t<T>;
            using result_t = typename distributor_t::result_type;
            const auto min_ = static_cast<result_t>(min);
            const auto max_ = static_cast<result_t>(max);
            const distributor_t distributor(min_, max_);
            RandomOperator randomizer(distributor);
            static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
            noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
        }
    }

    template<typename T, typename U, typename>
    void randomize(noa::math::normal_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U mean, U stddev, i64 threads) {
        NOA_ASSERT(output && all(shape > 0));

        if constexpr (nt::is_complex_v<T>) {
            if constexpr (nt::is_real_v<U>) {
                using real_t = nt::value_type_t<T>;
                using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, f32, real_t>;
                const auto reinterpreted = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                randomize(noa::math::normal_t{}, reinterpret_cast<real_t*>(output),
                          reinterpreted.strides, reinterpreted.shape,
                          static_cast<supported_float>(mean), static_cast<supported_float>(stddev), threads);
            } else {
                using real_t = nt::value_type_t<T>;
                using distributor_t = normal_distributor_t<real_t>;
                using result_t = typename distributor_t::result_type;
                auto mean_ = static_cast<result_t>(mean.real);
                auto stddev_ = static_cast<result_t>(stddev.real);
                const distributor_t distributor_real(mean_, stddev_);
                mean_ = static_cast<result_t>(mean.imag);
                stddev_ = static_cast<result_t>(stddev.imag);
                const distributor_t distributor_imag(mean_, stddev_);

                RandomOperatorComplex randomizer(distributor_real, distributor_imag);
                static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
                noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
            }
        } else {
            using distributor_t = normal_distributor_t<T>;
            using result_t = typename distributor_t::result_type;
            const auto mean_ = static_cast<result_t>(mean);
            const auto stddev_ = static_cast<result_t>(stddev);
            const distributor_t distributor(mean_, stddev_);
            RandomOperator randomizer(distributor);
            static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
            noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
        }
    }

    template<typename T, typename U, typename>
    void randomize(noa::math::log_normal_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   U mean, U stddev, i64 threads) {
        NOA_ASSERT(output && all(shape > 0));

        if constexpr (nt::is_complex_v<T>) {
            if constexpr (nt::is_real_v<U>) {
                using real_t = nt::value_type_t<T>;
                using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, f32, real_t>;
                const auto reinterpreted = indexing::Reinterpret(shape, strides, output).template as<real_t>();
                randomize(noa::math::log_normal_t{}, reinterpret_cast<real_t*>(output),
                          reinterpreted.strides, reinterpreted.shape,
                          static_cast<supported_float>(mean), static_cast<supported_float>(stddev), threads);
            } else {
                using real_t = nt::value_type_t<T>;
                using distributor_t = lognormal_distributor_t<real_t>;
                using result_t = typename distributor_t::result_type;
                auto mean_ = static_cast<result_t>(mean.real);
                auto stddev_ = static_cast<result_t>(stddev.real);
                const distributor_t distributor_real(mean_, stddev_);
                mean_ = static_cast<result_t>(mean.imag);
                stddev_ = static_cast<result_t>(stddev.imag);
                const distributor_t distributor_imag(mean_, stddev_);

                RandomOperatorComplex randomizer(distributor_real, distributor_imag);
                static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
                noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
            }
        } else {
            using distributor_t = lognormal_distributor_t<T>;
            using result_t = typename distributor_t::result_type;
            const auto mean_ = static_cast<result_t>(mean);
            const auto stddev_ = static_cast<result_t>(stddev);
            const distributor_t distributor(mean_, stddev_);
            RandomOperator randomizer(distributor);
            static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
            noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
        }
    }

    template<typename T, typename>
    void randomize(noa::math::poisson_t, T* output,
                   const Strides4<i64>& strides, const Shape4<i64>& shape,
                   f32 lambda, i64 threads) {
        NOA_ASSERT(output && all(shape > 0));

        if constexpr (nt::is_complex_v<T>) {
            using real_t = nt::value_type_t<T>;
            const auto reinterpreted = indexing::Reinterpret(shape, strides, output).template as<real_t>();
            randomize(noa::math::poisson_t{}, reinterpret_cast<real_t*>(output),
                      reinterpreted.strides, reinterpreted.shape, lambda, threads);
        } else {
            using distributor_t = poisson_distributor_t<T>;
            const distributor_t distributor(static_cast<double>(lambda));
            RandomOperator randomizer(distributor);
            static_assert(nt::is_detected_v<nt::has_initialize, decltype(randomizer)>);
            noa::cpu::utils::ewise_unary(output, strides, shape, randomizer, threads);
        }
    }

    #define INSTANTIATE_RANDOM_(T, U)                                                                                       \
    template void randomize<T, U, void>(noa::math::uniform_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64);     \
    template void randomize<T, U, void>(noa::math::normal_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64);      \
    template void randomize<T, U, void>(noa::math::log_normal_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64)

    INSTANTIATE_RANDOM_(i16, i16);
    INSTANTIATE_RANDOM_(u16, u16);
    INSTANTIATE_RANDOM_(i32, i32);
    INSTANTIATE_RANDOM_(u32, u32);
    INSTANTIATE_RANDOM_(i64, i64);
    INSTANTIATE_RANDOM_(u64, u64);
    INSTANTIATE_RANDOM_(f16, f16);
    INSTANTIATE_RANDOM_(f32, f32);
    INSTANTIATE_RANDOM_(f64, f64);
    INSTANTIATE_RANDOM_(c16, f16);
    INSTANTIATE_RANDOM_(c32, f32);
    INSTANTIATE_RANDOM_(c64, f64);
    INSTANTIATE_RANDOM_(c16, c16);
    INSTANTIATE_RANDOM_(c32, c32);
    INSTANTIATE_RANDOM_(c64, c64);

    #define INSTANTIATE_RANDOM_POISSON_(T) \
    template void randomize<T, void>(noa::math::poisson_t, T*, const Strides4<i64>&, const Shape4<i64>&, f32, i64)

    INSTANTIATE_RANDOM_POISSON_(f16);
    INSTANTIATE_RANDOM_POISSON_(f32);
    INSTANTIATE_RANDOM_POISSON_(f64);

    #define INSTANTIATE_RANDOM_HALF_(T, U)                                                                                  \
    template void randomize<T, U, void>(noa::math::uniform_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64);     \
    template void randomize<T, U, void>(noa::math::normal_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64);      \
    template void randomize<T, U, void>(noa::math::log_normal_t, T*, const Strides4<i64>&, const Shape4<i64>&, U, U, i64)

    INSTANTIATE_RANDOM_HALF_(f16, f32);
    INSTANTIATE_RANDOM_HALF_(c16, f32);
    INSTANTIATE_RANDOM_HALF_(c16, c32);
}
