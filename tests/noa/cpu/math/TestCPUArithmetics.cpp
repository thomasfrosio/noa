#include <noa/cpu/math/Arithmetics.h>

#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU: Arithmetics", "[noa][cpu][math]", int, uint, float, double, cfloat_t, cdouble_t) {
    Test::Randomizer<TestType> randomizer(1., 100.);

    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();
    uint batches = Test::IntRandomizer<uint>(1, 4).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> expected(elements * batches);
    TestType value = randomizer.get();
    Memory::PtrHost<TestType> values(batches);
    Memory::PtrHost<TestType> array(elements);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataRandom(values.get(), values.elements(), randomizer);
    Test::initDataRandom(array.get(), array.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());

    AND_THEN("multiply") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::multiplyByValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::multiplyByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::multiplyByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("divide") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::divideByValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::divideByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::divideByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("add") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::addValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::addValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::addArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("subtract") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::subtractValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::subtractValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::subtractArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: complex & real", "[noa][cpu][math]", cfloat_t, cdouble_t) {
    using real_t = Noa::Traits::value_type_t<TestType>;
    Test::Randomizer<TestType> randomizer(1., 100.);
    Test::Randomizer<real_t> randomizer_real(1., 100.);

    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();
    uint batches = Test::IntRandomizer<uint>(1, 4).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> expected(elements * batches);
    real_t value = randomizer_real.get();
    Memory::PtrHost<real_t> values(batches);
    Memory::PtrHost<real_t> array(elements);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataRandom(values.get(), values.elements(), randomizer_real);
    Test::initDataRandom(array.get(), array.elements(), randomizer_real);
    Test::initDataZero(expected.get(), expected.elements());

    AND_THEN("multiply") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] * value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::multiplyByValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::multiplyByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] * array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::multiplyByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::multiplyByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("divide") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] / value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::divideByValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::divideByValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] / array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::divideByArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::divideByArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("add") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] + value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::addValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::addValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] + array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::addArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::addArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }

    AND_THEN("subtract") {
        AND_THEN("value") {
            for (size_t idx{0}; idx < elements; ++idx)
                expected[idx] = data[idx] - value;

            // Out of place.
            Memory::PtrHost<TestType> results(elements);
            Math::subtractValue(data.get(), value, results.get(), elements);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractValue(data.get(), value, data.get(), elements);
            diff = Test::getDifference(expected.get(), data.get(), elements);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("values") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - values[batch];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::subtractValue(data.get(), values.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractValue(data.get(), values.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }

        AND_THEN("array") {
            for (uint batch{0}; batch < batches; ++batch)
                for (size_t idx{0}; idx < elements; ++idx)
                    expected[batch * elements + idx] = data[batch * elements + idx] - array[idx];

            // Out of place.
            Memory::PtrHost<TestType> results(elements * batches);
            Math::subtractArray(data.get(), array.get(), results.get(), elements, batches);
            TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic

            // In place.
            Math::subtractArray(data.get(), array.get(), data.get(), elements, batches);
            diff = Test::getDifference(expected.get(), data.get(), elements * batches);
            REQUIRE(diff == TestType(0)); // this should be deterministic
        }
    }
}

TEMPLATE_TEST_CASE("CPU: Arithmetics: divide safe (divide by 0 returns 0)", "[noa][cpu][math]", float, double) {
    Test::RealRandomizer<TestType> randomizer(-1, 1);

    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();
    uint batches = Test::IntRandomizer<uint>(1, 4).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> expected(elements * batches);
    Memory::PtrHost<TestType> array(elements);

    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataZero(expected.get(), expected.elements());
    Test::initDataZero(array.get(), array.elements());

    // Out of place.
    Memory::PtrHost<TestType> results(elements * batches);
    Test::initDataRandom(results.get(), results.elements(), randomizer);
    Math::divideSafeByArray(data.get(), array.get(), results.get(), elements, batches);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::divideSafeByArray(data.get(), array.get(), data.get(), elements, batches);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}
