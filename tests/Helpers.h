#pragma once

#include <noa/Types.h>
#include <noa/Math.h>
#include <noa/util/traits/BaseTypes.h>

#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <catch2/catch.hpp>

#define REQUIRE_ERRNO_GOOD(err) REQUIRE(err == ::Noa::Errno::good)

#define REQUIRE_FOR_ALL(range, predicate) for (auto& e: range) REQUIRE((predicate(e)))

#define REQUIRE_RANGE_EQUALS_ULP(actual, expected, ulp)                                 \
REQUIRE(actual.size() == expected.size());                                              \
for (size_t idx_{0}; idx_ < actual.size(); ++idx_) {                                    \
INFO("index: " << idx_);                                                                \
REQUIRE((Test::almostEquals(static_cast<float>(actual[idx_]), expected[idx_]), ulp)); }

#define REQUIRE_RANGE_EQUALS_ULP_OR_INVALID_SIZE(actual, expected, ulp, err)            \
if (actual.size() != expected.size()) REQUIRE(err == Errno::invalid_size);              \
else { REQUIRE_ERRNO_GOOD(err); REQUIRE_RANGE_EQUALS_ULP(actual, expected, ulp) }

#define REQUIRE_RANGE_EQUALS(actual, expected)          \
REQUIRE(actual.size() == expected.size());              \
for (size_t idx_{0}; idx_ < actual.size(); ++idx_) {    \
INFO("index: " << idx_);                                \
REQUIRE(actual[idx_] == expected[idx_]); }

#define REQUIRE_RANGE_EQUALS_OR_INVALID_SIZE(actual, expected, err)         \
if (actual.size() != expected.size()) REQUIRE(err == Errno::invalid_size);  \
else { REQUIRE_ERRNO_GOOD(err); REQUIRE_RANGE_EQUALS(actual, expected) }

namespace Test {
    template<typename T, size_t N>
    inline std::vector<T> toVector(const std::array<T, N>& array) {
        return std::vector<T>(array.cbegin(), array.cend());
    }

    inline bool almostEquals(float actual, float expected, uint64_t maxUlpDiff = 2) {
        static_assert(sizeof(float) == sizeof(int32_t));

        if (std::isnan(actual) || std::isnan(expected))
            return false;

        auto convert = [](float value) {
            int32_t i;
            std::memcpy(&i, &value, sizeof(value));
            return i;
        };

        auto lc = convert(actual);
        auto rc = convert(expected);

        if ((lc < 0) != (rc < 0)) {
            // Potentially we can have +0 and -0
            return actual == expected;
        }

        auto ulpDiff = std::abs(lc - rc);
        return static_cast<uint64_t>(ulpDiff) <= maxUlpDiff;
    }

    inline bool almostEquals(double actual, double expected, uint64_t maxUlpDiff = 2) {
        static_assert(sizeof(double) == sizeof(int64_t));

        if (std::isnan(actual) || std::isnan(expected))
            return false;

        auto convert = [](double value) {
            int64_t i;
            std::memcpy(&i, &value, sizeof(value));
            return i;
        };

        auto lc = convert(actual);
        auto rc = convert(expected);

        if ((lc < 0) != (rc < 0))
            return actual == expected;

        auto ulpDiff = std::abs(lc - rc);
        return static_cast<uint64_t>(ulpDiff) <= maxUlpDiff;
    }
}

// RANDOM:
namespace Test {
    template<typename T>
    class RealRandomizer {
    private:
        std::random_device rand_dev;
        std::mt19937 generator;
        std::uniform_real_distribution<T> distribution;
    public:
        RealRandomizer(T range_from, T range_to) : generator(rand_dev()), distribution(range_from, range_to) {}
        inline T get() { return distribution(generator); }
    };

    template<>
    class RealRandomizer<Noa::cfloat_t> {
    private:
        std::random_device rand_dev;
        std::mt19937 generator;
        std::uniform_real_distribution<float> distribution;
    public:
        RealRandomizer(float from, float to) : generator(rand_dev()), distribution(from, to) {}
        RealRandomizer(double from, double to)
                : generator(rand_dev()), distribution(static_cast<float>(from), static_cast<float>(to)) {}
        inline Noa::cfloat_t get() { return Noa::cfloat_t(distribution(generator), distribution(generator)); }
    };

    template<>
    class RealRandomizer<Noa::cdouble_t> {
    private:
        std::random_device rand_dev{};
        std::mt19937 generator;
        std::uniform_real_distribution<double> distribution;
    public:
        RealRandomizer(double from, double to) : generator(rand_dev()), distribution(from, to) {}
        RealRandomizer(float from, float to)
                : generator(rand_dev()), distribution(static_cast<double>(from), static_cast<double>(to)) {}
        inline Noa::cdouble_t get() { return Noa::cdouble_t(distribution(generator), distribution(generator)); }
    };

    template<typename T>
    class IntRandomizer {
    private:
        std::random_device rand_dev;
        std::mt19937 generator;
        std::uniform_int_distribution<T> distribution;
    public:
        IntRandomizer(T range_from, T range_to) : generator(rand_dev()), distribution(range_from, range_to) {}
        inline T get() { return distribution(generator); }
    };

    // More flexible
    template<typename T>
    struct MyIntRandomizer { using type = IntRandomizer<T>; };
    template<typename T>
    struct MyRealRandomizer { using type = RealRandomizer<T>; };
    template<typename T>
    using Randomizer = typename std::conditional_t<Noa::Traits::is_int_v<T>,
                                                   MyIntRandomizer<T>, MyRealRandomizer<T>>::type;

    inline int pseudoRandom(int range_from, int range_to) {
        int out = range_from + std::rand() / (RAND_MAX / (range_to - range_from + 1) + 1);
        return out;
    }

    inline Noa::size3_t getRandomShape(uint ndim) {
        if (ndim == 2) {
            Test::IntRandomizer<size_t> randomizer(32, 128);
            return Noa::size3_t{randomizer.get(), randomizer.get(), 1};
        } else if (ndim == 3) {
            Test::IntRandomizer<size_t> randomizer(32, 64);
            return Noa::size3_t{randomizer.get(), randomizer.get(), randomizer.get()};
        } else {
            Test::IntRandomizer<size_t> randomizer(32, 1024);
            return Noa::size3_t{randomizer.get(), 1, 1};
        }
    }
}

// INITIALIZE DATA:
namespace Test {
    template<typename T, typename U>
    inline void initDataRandom(T* data, size_t elements, Test::IntRandomizer<U>& randomizer) {
        if constexpr (std::is_same_v<T, Noa::cfloat_t>) {
            initDataRandom(reinterpret_cast<float*>(data), elements * 2, randomizer);
            return;
        } else if constexpr (std::is_same_v<T, Noa::cdouble_t>) {
            initDataRandom(reinterpret_cast<double*>(data), elements * 2, randomizer);
            return;
        } else {
            for (size_t idx{0}; idx < elements; ++idx)
                data[idx] = static_cast<T>(randomizer.get());
        }
    }

    template<typename T>
    inline void initDataRandom(T* data, size_t elements, Test::RealRandomizer<T>& randomizer) {
        for (size_t idx{0}; idx < elements; ++idx)
            data[idx] = randomizer.get();
    }

    template<typename T>
    inline void initDataZero(T* data, size_t elements) {
        for (size_t idx{0}; idx < elements; ++idx)
            data[idx] = 0;
    }
}

// COMPUTE DIFFERENCES:
namespace Test {
    template<typename T>
    inline T getDifference(const T* in, const T* out, size_t elements) {
        if constexpr (std::is_same_v<T, Noa::cfloat_t> || std::is_same_v<T, Noa::cdouble_t>) {
            T diff{0}, tmp;
            for (size_t idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += T{tmp.real() > 0 ? tmp.real() : -tmp.real(),
                          tmp.imag() > 0 ? tmp.imag() : -tmp.imag()};
            }
            return diff;
        } else {
            T diff{0}, tmp;
            for (size_t idx{0}; idx < elements; ++idx) {
                tmp = out[idx] - in[idx];
                diff += tmp > 0 ? tmp : -tmp;
            }
            return diff;
        }
    }

    template<typename T>
    inline T getAverageDifference(const T* in, const T* out, size_t elements) {
        T diff = getDifference(in, out, elements);
        if constexpr (std::is_same_v<T, Noa::cfloat_t>) {
            return diff / static_cast<float>(elements);
        } else if constexpr (std::is_same_v<T, Noa::cdouble_t>) {
            return diff / static_cast<double>(elements);
        } else {
            return diff / static_cast<T>(elements);;
        }
    }

    template<typename T, typename U>
    inline void normalize(T* array, size_t size, U scale) {
        for (size_t idx{0}; idx < size; ++idx) {
            array[idx] *= scale;
        }
    }
}

// MATCHERS:
namespace Test {
    template<typename T, typename U>
    class WithinAbs : public Catch::MatcherBase<T> {
        T m_expected;
        U m_epsilon;
    public:
        WithinAbs(T expected, U epsilon) : m_expected(expected), m_epsilon(epsilon) {}

        bool match(const T& value) const override {
            if constexpr (std::is_same_v<T, Noa::cfloat_t>) {
                return Noa::Math::abs(value.real() - m_expected.real()) <= static_cast<float>(m_epsilon) &&
                       Noa::Math::abs(value.imag() - m_expected.imag()) <= static_cast<float>(m_epsilon);
            } else if constexpr (std::is_same_v<T, Noa::cdouble_t>) {
                return Noa::Math::abs(value.real() - m_expected.real()) <= static_cast<double>(m_epsilon) &&
                       Noa::Math::abs(value.imag() - m_expected.imag()) <= static_cast<double>(m_epsilon);
            } else {
                return Noa::Math::abs(value - m_expected) <= static_cast<T>(m_epsilon);
            }
        }

        std::string describe() const override {
            std::ostringstream ss;
            ss << "is equal to " << m_expected << " +/- abs epsilon of " << m_epsilon;
            return ss.str();
        }
    };

    /// Whether or not the tested value is equal to @a expected_value +/- @a epsilon.
    /// @note For complex types, the same epsilon is applied to the real and imaginary part.
    template<typename T, typename U,
             typename = std::enable_if_t<std::is_floating_point_v<U> &&
                                         (Noa::Traits::is_float_v<T> || Noa::Traits::is_complex_v<T>)>>
    inline WithinAbs<T, U> isWithinAbs(T expected_value, U epsilon) {
        return WithinAbs(expected_value, epsilon);
    }
}
