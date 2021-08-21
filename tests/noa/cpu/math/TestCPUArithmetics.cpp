#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math:: Arithmetics", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 100.);

    size_t elements = test::IntRandomizer<size_t>(1, 100).get();
    uint batches = test::IntRandomizer<uint>(1, 4).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);
    TestType value = randomizer.get();
    cpu::memory::PtrHost<TestType> values(batches);
    cpu::memory::PtrHost<TestType> array(elements);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataRandom(values.get(), values.elements(), randomizer);
    test::initDataRandom(array.get(), array.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());

    AND_THEN("multiply") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::multiplyByValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::multiplyByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::multiplyByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("divide") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::divideByValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::divideByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::divideByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("add") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::addValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::addValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::addArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("subtract") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::subtractValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::subtractValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::subtractArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("cpu::math:: Arithmetics: complex & real", "[noa][cpu][math]", cfloat_t, cdouble_t) {
    using real_t = noa::traits::value_type_t<TestType>;
    test::Randomizer<TestType> randomizer(1., 100.);
    test::Randomizer<real_t> randomizer_real(1., 100.);

    size_t elements = test::IntRandomizer<size_t>(1, 100).get();
    uint batches = test::IntRandomizer<uint>(1, 4).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);
    real_t value = randomizer_real.get();
    cpu::memory::PtrHost<real_t> values(batches);
    cpu::memory::PtrHost<real_t> array(elements);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataRandom(values.get(), values.elements(), randomizer_real);
    test::initDataRandom(array.get(), array.elements(), randomizer_real);
    test::initDataZero(expected.get(), expected.elements());

    AND_THEN("multiply") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::multiplyByValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::multiplyByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::multiplyByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::multiplyByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("divide") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::divideByValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::divideByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::divideByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::divideByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("add") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::addValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::addValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::addArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::addArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("subtract") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - value;

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements);
            cpu::math::subtractValue(data.get(), value, results.get(), elements);
            TestType diff = test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractValue(data.get(), value, data.get(), elements);
            diff = test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - values[batch];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::subtractValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractValue(data.get(), values.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - array[idx];

            // Out of place.
            cpu::memory::PtrHost<TestType> results(elements * batches);
            cpu::math::subtractArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            cpu::math::subtractArray(data.get(), array.get(), data.get(), elements, batches);
            diff = test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("cpu::math:: Arithmetics: divide safe (divide by 0 returns 0)", "[noa][cpu][math]", float, double) {
    test::RealRandomizer<TestType> randomizer(-1, 1);

    size_t elements = test::IntRandomizer<size_t>(1, 100).get();
    uint batches = test::IntRandomizer<uint>(1, 4).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> expected(elements * batches);
    cpu::memory::PtrHost<TestType> array(elements);

    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataZero(expected.get(), expected.elements());
    test::initDataZero(array.get(), array.elements());

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements * batches);
    test::initDataRandom(results.get(), results.elements(), randomizer);
    cpu::math::divideSafeByArray(data.get(), array.get(), results.get(), elements, batches);
    TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::divideSafeByArray(data.get(), array.get(), data.get(), elements, batches);
    diff = test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}
