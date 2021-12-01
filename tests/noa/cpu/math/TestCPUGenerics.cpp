#include <noa/cpu/math/Generics.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math:: generics with no parameters", "[noa][cpu][math]", float, double) {
    size_t elements = test::Randomizer<size_t>(0, 100).get();
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> expected(elements);
    cpu::memory::PtrHost<TestType> results(elements);

    WHEN("data can be negative") {
        test::Randomizer<TestType> randomizer(-10., 10.);
        test::randomize(data.get(), data.elements(), randomizer);
        WHEN("oneMinus") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) - data[idx];

            // Out of place.
            cpu::math::oneMinus(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::oneMinus(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("inverse") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = TestType(1) / data[idx];

            // Out of place.
            cpu::math::inverse(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::inverse(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("square") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * data[idx];

            // Out of place.
            cpu::math::square(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::square(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("exp") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::exp(data[idx]);

            // Out of place.
            cpu::math::exp(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::exp(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("abs") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::abs(data[idx]);

            // Out of place.
            cpu::math::abs(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::abs(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("cos") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::cos(data[idx]);

            // Out of place.
            cpu::math::cos(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::cos(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("sin") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::sin(data[idx]);

            // Out of place.
            cpu::math::sin(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::sin(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    WHEN("data should be positive") {
        test::Randomizer<TestType> randomizer(1., 100.);
        test::randomize(data.get(), data.elements(), randomizer);

        WHEN("sqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::sqrt(data[idx]);

            // Out of place.
            cpu::math::sqrt(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::sqrt(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("rsqrt") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::rsqrt(data[idx]);

            // Out of place.
            cpu::math::rsqrt(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::rsqrt(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("log") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::log(data[idx]);

            // Out of place.
            cpu::math::log(data.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::log(data.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("cpu::math:: generics with complex types", "[noa][cpu][math]", cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;
    size_t elements = test::Randomizer<size_t>(0, 100).get();
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> expected(elements);
    cpu::memory::PtrHost<TestType> results(elements);

    test::Randomizer<TestType> randomizer(-10., 10.);
    test::randomize(data.get(), data.elements(), randomizer);

    AND_THEN("oneMinus") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) - data[idx];

        // Out of place.
        cpu::math::oneMinus(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::oneMinus(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("inverse") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = TestType(1) / data[idx];

        // Out of place.
        cpu::math::inverse(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::inverse(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("squared") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = data[idx] * data[idx];

        // Out of place.
        cpu::math::square(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::square(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("abs") {
        cpu::memory::PtrHost<real_t> real_expected(elements);
        cpu::memory::PtrHost<real_t> real_result(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            real_expected[idx] = math::abs(data[idx]);

        cpu::math::abs(data.get(), real_result.get(), elements);
        real_t diff = test::getDifference(real_expected.get(), real_result.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("normalize") {
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = math::normalize(data[idx]);

        // Out of place.
        cpu::math::normalize(data.get(), results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::normalize(data.get(), data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("real, imag") {
        cpu::memory::PtrHost<real_t> real_expected(elements);
        cpu::memory::PtrHost<real_t> real_result(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            real_expected[idx] = math::real(data[idx]);

        cpu::math::real(data.get(), real_result.get(), elements);
        real_t diff = test::getDifference(real_expected.get(), real_result.get(), elements);
        REQUIRE(diff == real_t(0));

        for (size_t idx{0}; idx < elements; ++idx)
            real_expected[idx] = math::imag(data[idx]);

        cpu::math::imag(data.get(), real_result.get(), elements);
        diff = test::getDifference(real_expected.get(), real_result.get(), elements);
        REQUIRE(diff == real_t(0));
    }

    AND_THEN("realAndImag, complex") {
        cpu::memory::PtrHost<real_t> real(elements);
        cpu::memory::PtrHost<real_t> imag(elements);

        cpu::math::realAndImag(data.get(), real.get(), imag.get(), elements);
        cpu::math::complex(real.get(), imag.get(), results.get(), elements);
        TestType diff = test::getDifference(data.get(), results.get(), elements);
        REQUIRE(diff == TestType(0));
    }
}

TEMPLATE_TEST_CASE("cpu::math:: generics with arguments", "[noa][cpu][math]", int, uint, float, double) {
    size_t elements = test::Randomizer<size_t>(0, 100).get();
    cpu::memory::PtrHost<TestType> data(elements);
    cpu::memory::PtrHost<TestType> expected(elements);
    cpu::memory::PtrHost<TestType> results(elements);

    test::Randomizer<TestType> randomizer(0., 10.);
    test::randomize(data.get(), data.elements(), randomizer);

    WHEN("pow") {
        if constexpr (std::is_floating_point_v<TestType>) {
            TestType exponent = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::pow(data[idx], exponent);

            // Out of place.
            cpu::math::pow(data.get(), exponent, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::pow(data.get(), exponent, data.get(), elements);
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
        cpu::math::clamp(data.get(), low, high, results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::clamp(data.get(), low, high, data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

    }

    WHEN("min") {
        WHEN("single threshold") {
            TestType threshold = randomizer.get();
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::min(data[idx], threshold);

            // Out of place.
            cpu::math::min(data.get(), threshold, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::min(data.get(), threshold, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            cpu::memory::PtrHost<TestType> array(elements);
            test::randomize(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::min(data[idx], array[idx]);

            // Out of place.
            cpu::math::min(data.get(), array.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::min(data.get(), array.get(), data.get(), elements);
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
            cpu::math::max(data.get(), threshold, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::max(data.get(), threshold, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        WHEN("element-wise") {
            cpu::memory::PtrHost<TestType> array(elements);
            test::randomize(array.get(), elements, randomizer);
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = math::max(data[idx], array[idx]);

            // Out of place.
            cpu::math::max(data.get(), array.get(), results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::max(data.get(), array.get(), data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}
