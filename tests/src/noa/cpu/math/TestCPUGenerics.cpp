#include <noa/cpu/math/Generics.h>

#include <noa/cpu/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU::Math: Generics: generics with no parameters", "[noa][cpu][math]", float, double) {
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();
    PtrHost<TestType> data(elements);
    PtrHost<TestType> expected(elements);
    PtrHost<TestType> results(elements);

    WHEN("data can be negative") {
        Test::Randomizer<TestType> randomizer(-10., 10.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);
        WHEN("oneMinus") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) - data[idx];

            // Out of place.
            Math::oneMinus(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::oneMinus(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("inverse") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) / data[idx];

            // Out of place.
            Math::inverse(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::inverse(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("square") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * data[idx];

            // Out of place.
            Math::square(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::square(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("exp") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::exp(data[idx]);

            // Out of place.
            Math::exp(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::exp(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("abs") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::abs(data[idx]);

            // Out of place.
            Math::abs(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::abs(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("cos") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::cos(data[idx]);

            // Out of place.
            Math::cos(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::cos(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("sin") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::sin(data[idx]);

            // Out of place.
            Math::sin(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::sin(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("data should be positive") {
        Test::Randomizer<TestType> randomizer(1., 100.);
        Test::initDataRandom(data.get(), data.elements(), randomizer);

        WHEN("sqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::sqrt(data[idx]);

            // Out of place.
            Math::sqrt(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::sqrt(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("rsqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::rsqrt(data[idx]);

            // Out of place.
            Math::rsqrt(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::rsqrt(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("log") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::log(data[idx]);

            // Out of place.
            Math::log(data.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::log(data.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU::Math: Generics: complex types", "[noa][cpu][math]", cfloat_t, cdouble_t) {
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();
    PtrHost<TestType> data(elements);
    PtrHost<TestType> expected(elements);
    PtrHost<TestType> results(elements);

    Test::Randomizer<TestType> randomizer(-10., 10.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    WHEN("oneMinus") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) - data[idx];

        // Out of place.
        Math::oneMinus(data.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::oneMinus(data.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("inverse") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) / data[idx];

        // Out of place.
        Math::inverse(data.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::inverse(data.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("squared") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * data[idx];

        // Out of place.
        Math::square(data.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::square(data.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("abs") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = Math::abs(data[idx]);

        // Out of place.
        Math::abs(data.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::abs(data.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("normalize") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = Math::normalize(data[idx]);

        // Out of place.
        Math::normalize(data.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::normalize(data.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}

TEMPLATE_TEST_CASE("CPU::Math: Generics: generics with arguments", "[noa][cpu][math]", int, uint, float, double) {
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();
    PtrHost<TestType> data(elements);
    PtrHost<TestType> expected(elements);
    PtrHost<TestType> results(elements);

    Test::Randomizer<TestType> randomizer(0., 10.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    WHEN("pow") {
        if constexpr (std::is_floating_point_v<TestType>) {
            TestType exponent = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::pow(data[idx], exponent);

            // Out of place.
            Math::pow(data.get(), exponent, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::pow(data.get(), exponent, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("pow") {
        TestType low = randomizer.get();
        TestType high = low + 4;
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = Math::clamp(data[idx], low, high);

        // Out of place.
        Math::clamp(data.get(), low, high, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::clamp(data.get(), low, high, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

    }

    WHEN("min") {
        WHEN("single threshold") {
            TestType threshold = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::min(data[idx], threshold);

            // Out of place.
            Math::min(data.get(), threshold, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::min(data.get(), threshold, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            PtrHost<TestType> array(elements);
            Test::initDataRandom(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::min(data[idx], array[idx]);

            // Out of place.
            Math::min(data.get(), array.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::min(data.get(), array.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("max") {
        WHEN("single threshold") {
            TestType threshold = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::max(data[idx], threshold);

            // Out of place.
            Math::max(data.get(), threshold, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::max(data.get(), threshold, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            PtrHost<TestType> array(elements);
            Test::initDataRandom(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = Math::max(data[idx], array[idx]);

            // Out of place.
            Math::max(data.get(), array.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::max(data.get(), array.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}
