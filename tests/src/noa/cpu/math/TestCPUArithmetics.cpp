#include <noa/cpu/math/Arithmetics.h>

#include <noa/cpu/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU: Arithmetics: multiply", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    using real_t = std::conditional_t<std::is_same_v<TestType, cfloat_t>, float, double>;
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("single value") {
        TestType value = randomizer.get();

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * value;

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::multiply(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::multiply(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("element-wise") {
        PtrHost<TestType> values(elements);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * values[idx];

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::multiply(data.get(), values.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::multiply(data.get(), values.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("single value: complex & real") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            real_t value = Test::RealRandomizer<real_t>(0, 100).get();

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * value;

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::multiply(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiply(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("element-wise: complex* & real*") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::RealRandomizer<real_t> real_randomizer(0, 100);
            PtrHost<real_t> values(elements);
            Test::initDataRandom(values.get(), values.elements(), real_randomizer);

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * values[idx];

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::multiply(data.get(), values.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiply(data.get(), values.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: divide", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    using real_t = std::conditional_t<std::is_same_v<TestType, cfloat_t>, float, double>;

    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("single value") {
        TestType value = randomizer.get();

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] / value;

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::divide(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::divide(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("element-wise") {
        PtrHost<TestType> values(elements);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] / values[idx];

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::divide(data.get(), values.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::divide(data.get(), values.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("single value: complex* & real") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            real_t value = Test::RealRandomizer<real_t>(0, 100).get();

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / value;

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::divide(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divide(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("element-wise: complex* & real*") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::RealRandomizer<real_t> real_randomizer(0, 100);
            PtrHost<real_t> values(elements);
            Test::initDataRandom(values.get(), values.elements(), real_randomizer);

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / values[idx];

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::divide(data.get(), values.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divide(data.get(), values.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: add", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    using real_t = std::conditional_t<std::is_same_v<TestType, cfloat_t>, float, double>;

    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("single value") {
        TestType value = randomizer.get();

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] + value;

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::add(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::add(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("element-wise") {
        PtrHost<TestType> values(elements);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] + values[idx];

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::add(data.get(), values.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::add(data.get(), values.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("single value: complex* & real") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            real_t value = Test::RealRandomizer<real_t>(0, 100).get();

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + value;

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::add(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::add(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("element-wise: complex* & real*") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::RealRandomizer<real_t> real_randomizer(0, 100);
            PtrHost<real_t> values(elements);
            Test::initDataRandom(values.get(), values.elements(), real_randomizer);

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + values[idx];

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::add(data.get(), values.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::add(data.get(), values.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: subtract", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    using real_t = std::conditional_t<std::is_same_v<TestType, cfloat_t>, float, double>;

    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("single value") {
        TestType value = randomizer.get();

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] - value;

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::subtract(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::subtract(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("element-wise") {
        PtrHost<TestType> values(elements);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] - values[idx];

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::subtract(data.get(), values.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::subtract(data.get(), values.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("single value: complex* & real") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            real_t value = Test::RealRandomizer<real_t>(0, 100).get();

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - value;

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::subtract(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtract(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("element-wise: complex* & real*") {
        if constexpr (Noa::Traits::is_complex_v<TestType>) {
            Test::RealRandomizer<real_t> real_randomizer(0, 100);
            PtrHost<real_t> values(elements);
            Test::initDataRandom(values.get(), values.elements(), real_randomizer);

            PtrHost<TestType> expected(elements);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - values[idx];

            // Out of place.
            PtrHost<TestType> results(elements);
            Math::subtract(data.get(), values.get(), results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtract(data.get(), values.get(), data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: one minus", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = TestType(1) - data[idx];

    // Out of place.
    PtrHost<TestType> results(elements);
    Math::oneMinus(data.get(), results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::oneMinus(data.get(), data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: inverse", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = TestType(1) / data[idx];

    // Out of place.
    PtrHost<TestType> results(elements);
    Math::inverse(data.get(), results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::inverse(data.get(), data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: fma", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    PtrHost<TestType> multiplicands(elements);
    PtrHost<TestType> addends(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataRandom(multiplicands.get(), multiplicands.elements(), randomizer);
    Test::initDataRandom(addends.get(), addends.elements(), randomizer);

    PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] * multiplicands[idx] + addends[idx];

    // Out of place.
    PtrHost<TestType> results(elements);
    Math::fma(data.get(), multiplicands.get(), addends.get(), results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::fma(data.get(), multiplicands.get(), addends.get(), data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: squaredDifference", "[noa][cpu][math]",
                   int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("single value") {
        TestType value = randomizer.get();

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = (data[idx] - value) * (data[idx] - value);

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::squaredDistance(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::squaredDistance(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("element-wise") {
        PtrHost<TestType> values(elements);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = (data[idx] - values[idx]) * (data[idx] - values[idx]);

        // Out of place.
        PtrHost<TestType> results(elements);
        Math::squaredDistance(data.get(), values.get(), results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::squaredDistance(data.get(), values.get(), data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}
