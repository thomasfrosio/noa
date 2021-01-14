#pragma once
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


//
// Some random functions.
//
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
    inline T random(T range_from, T range_to) {
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        std::uniform_int_distribution<T> distribution(range_from, range_to);
        return distribution(generator);
    }

    inline int pseudoRandom(int range_from, int range_to) {
        int out = range_from + std::rand() / (RAND_MAX / (range_to - range_from + 1) + 1);
        return out;
    }
}
