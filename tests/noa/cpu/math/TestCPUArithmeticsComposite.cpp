#include <noa/cpu/math/ArithmeticsComposite.h>

#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU: ArithmeticsComposite: multiplyAdd", "[noa][cpu][math]",
                   int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    uint batches = Test::IntRandomizer<uint>(1, 4).get();
    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Memory::PtrHost<TestType> multiplicands(elements);
    Memory::PtrHost<TestType> addends(elements);
    Test::initDataRandom(data.get(), data.elements(), randomizer);
    Test::initDataRandom(multiplicands.get(), multiplicands.elements(), randomizer);
    Test::initDataRandom(addends.get(), addends.elements(), randomizer);

    Memory::PtrHost<TestType> expected(elements * batches);
    for (uint batch{0}; batch < batches; ++batch)
        for (size_t idx{0}; idx < elements; ++idx)
            expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[idx] + addends[idx];

    // Out of place.
    Memory::PtrHost<TestType> results(elements * batches);
    Math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), results.get(), elements, batches);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), data.get(), elements, batches);
    diff = Test::getDifference(expected.get(), data.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: ArithmeticsComposite: squaredDifference", "[noa][cpu][math]",
                   int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    uint batches = Test::IntRandomizer<uint>(1, 5).get();
    size_t elements = Test::IntRandomizer<size_t>(0, 100).get();

    Memory::PtrHost<TestType> data(elements * batches);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("value") {
        TestType value = randomizer.get();

        Memory::PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = (data[idx] - value) * (data[idx] - value);

        // Out of place.
        Memory::PtrHost<TestType> results(elements);
        Math::squaredDistanceFromValue(data.get(), value, results.get(), elements);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::squaredDistanceFromValue(data.get(), value, data.get(), elements);
        diff = Test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("values") {
        Memory::PtrHost<TestType> values(batches);
        Test::initDataRandom(values.get(), values.elements(), randomizer);

        Memory::PtrHost<TestType> expected(elements * batches);
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - values[batch]) *
                                                   (data[batch * elements + idx] - values[batch]);

        // Out of place.
        Memory::PtrHost<TestType> results(elements * batches);
        Math::squaredDistanceFromValue(data.get(), values.get(), results.get(), elements, batches);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::squaredDistanceFromValue(data.get(), values.get(), data.get(), elements, batches);
        diff = Test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("array") {
        Memory::PtrHost<TestType> array(elements);
        Test::initDataRandom(array.get(), array.elements(), randomizer);

        Memory::PtrHost<TestType> expected(elements * batches);
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - array[idx]) *
                                                   (data[batch * elements + idx] - array[idx]);

        // Out of place.
        Memory::PtrHost<TestType> results(elements * batches);
        Math::squaredDistanceFromArray(data.get(), array.get(), results.get(), elements, batches);
        TestType diff = Test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        Math::squaredDistanceFromArray(data.get(), array.get(), data.get(), elements, batches);
        diff = Test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}
