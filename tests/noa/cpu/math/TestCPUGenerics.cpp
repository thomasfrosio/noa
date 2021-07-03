#include <noa/cpu/math/Generics.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("math:: generics with no parameters", "[noa][cpu][math]", float, double) {
    size_t elements = test::IntRandomizer<size_t>(0, 100).get();
    memory::PtrHost<TestType> data(elements);
    memory::PtrHost<TestType> expected(elements);
    memory::PtrHost<TestType> results(elements);

    WHEN("data can be negative") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::initDataRandom(data.get(), data.elements(), randomizer);
        WHEN("oneMinus") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) - data[idx];

            // Out of place.
            math::oneMinus(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::oneMinus(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("inverse") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) / data[idx];

            // Out of place.
            math::inverse(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::inverse(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("square") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * data[idx];

            // Out of place.
            math::square(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::square(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("exp") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::exp(data[idx]);

            // Out of place.
            math::exp(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::exp(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("abs") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::abs(data[idx]);

            // Out of place.
            math::abs(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::abs(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("cos") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::cos(data[idx]);

            // Out of place.
            math::cos(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::cos(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("sin") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::sin(data[idx]);

            // Out of place.
            math::sin(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::sin(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("data should be positive") {
        test::Randomizer<TestType> randomizer(1., 100.);
        test::initDataRandom(data.get(), data.elements(), randomizer);

        WHEN("sqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::sqrt(data[idx]);

            // Out of place.
            math::sqrt(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::sqrt(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("rsqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::rsqrt(data[idx]);

            // Out of place.
            math::rsqrt(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::rsqrt(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("log") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::log(data[idx]);

            // Out of place.
            math::log(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::log(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("math:: generics with complex types", "[noa][cpu][math]", cfloat_t, cdouble_t) {
    size_t elements = test::IntRandomizer<size_t>(0, 100).get();
    memory::PtrHost<TestType> data(elements);
    memory::PtrHost<TestType> expected(elements);
    memory::PtrHost<TestType> results(elements);

    test::Randomizer<TestType> randomizer(-10., 10.);
    test::initDataRandom(data.get(), data.elements(), randomizer);

    WHEN("oneMinus") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) - data[idx];

        // Out of place.
        math::oneMinus(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::oneMinus(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("inverse") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) / data[idx];

        // Out of place.
        math::inverse(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::inverse(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("squared") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * data[idx];

        // Out of place.
        math::square(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::square(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("abs") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = math::abs(data[idx]);

        // Out of place.
        math::abs(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::abs(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    WHEN("normalize") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = math::normalize(data[idx]);

        // Out of place.
        math::normalize(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::normalize(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}

TEMPLATE_TEST_CASE("math:: generics with arguments", "[noa][cpu][math]", int, uint, float, double) {
    size_t elements = test::IntRandomizer<size_t>(0, 100).get();
    memory::PtrHost<TestType> data(elements);
    memory::PtrHost<TestType> expected(elements);
    memory::PtrHost<TestType> results(elements);

    test::Randomizer<TestType> randomizer(0., 10.);
    test::initDataRandom(data.get(), data.elements(), randomizer);

    WHEN("pow") {
        if constexpr (std::is_floating_point_v<TestType>) {
            TestType exponent = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::pow(data[idx], exponent);

            // Out of place.
            math::pow(data.get(), exponent, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::pow(data.get(), exponent, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("pow") {
        TestType low = randomizer.get();
        TestType high = low + 4;
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = math::clamp(data[idx], low, high);

        // Out of place.
        math::clamp(data.get(), low, high, results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::clamp(data.get(), low, high, data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

    }

    WHEN("min") {
        WHEN("single threshold") {
            TestType threshold = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::min(data[idx], threshold);

            // Out of place.
            math::min(data.get(), threshold, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::min(data.get(), threshold, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            memory::PtrHost<TestType> array(elements);
            test::initDataRandom(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::min(data[idx], array[idx]);

            // Out of place.
            math::min(data.get(), array.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::min(data.get(), array.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("max") {
        WHEN("single threshold") {
            TestType threshold = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::max(data[idx], threshold);

            // Out of place.
            math::max(data.get(), threshold, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::max(data.get(), threshold, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            memory::PtrHost<TestType> array(elements);
            test::initDataRandom(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::max(data[idx], array[idx]);

            // Out of place.
            math::max(data.get(), array.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            math::max(data.get(), array.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}
