#pragma once

#include <noa/Types.h>
#include <noa/util/Math.h>

#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define REQUIRE_ERRNO_GOOD(err) REQUIRE(err == ::Noa::Errno::good)

#define REQUIRE_FOR_N(ptr, size, predicate) for (size_t i_{0}; i < size; ++i) REQUIRE((predicate(i)))

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

    inline int pseudoRandom(int range_from, int range_to) {
        int out = range_from + std::rand() / (RAND_MAX / (range_to - range_from + 1) + 1);
        return out;
    }

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

    template<typename T, typename U>
    inline void normalize(T* array, size_t size, U scale) {
        for (size_t idx{0}; idx < size; ++idx) {
            array[idx] *= scale;
        }
    }

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

    template<typename T, typename U>
    inline void initDataRandom(T* data, size_t elements, Test::RealRandomizer<U>& randomizer) {
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
    inline void initDataZero(T* data, size_t elements) {
        for (size_t idx{0}; idx < elements; ++idx)
            data[idx] = 0;
    }

    inline Noa::size3_t getShapeReal(uint ndim) {
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
