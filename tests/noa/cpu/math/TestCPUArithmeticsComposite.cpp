#include <noa/cpu/math/ArithmeticsComposite.h>

#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("CPU: ArithmeticsComposite: multiplyAdd", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    uint batches = test::IntRandomizer<uint>(1, 4).get();
    size_t elements = test::IntRandomizer<size_t>(1, 100).get();

    memory::PtrHost<TestType> data(elements * batches);
    memory::PtrHost<TestType> multiplicands(elements);
    memory::PtrHost<TestType> addends(elements);
    test::initDataRandom(data.get(), data.elements(), randomizer);
    test::initDataRandom(multiplicands.get(), multiplicands.elements(), randomizer);
    test::initDataRandom(addends.get(), addends.elements(), randomizer);

    memory::PtrHost<TestType> expected(elements * batches);
    for (uint batch{0}; batch < batches; ++batch)
        for (size_t idx{0}; idx < elements; ++idx)
            expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[idx] + addends[idx];

    // Out of place.
    memory::PtrHost<TestType> results(elements * batches);
    math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), results.get(), elements, batches);
    TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), data.get(), elements, batches);
    diff = test::getDifference(expected.get(), data.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: ArithmeticsComposite: squaredDifference", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    uint batches = test::IntRandomizer<uint>(1, 5).get();
    size_t elements = test::IntRandomizer<size_t>(0, 100).get();

    memory::PtrHost<TestType> data(elements * batches);
    test::initDataRandom(data.get(), data.elements(), randomizer);

    AND_THEN("value") {
        TestType value = randomizer.get();

        memory::PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = (data[idx] - value) * (data[idx] - value);

        // Out of place.
        memory::PtrHost<TestType> results(elements);
        math::squaredDistanceFromValue(data.get(), value, results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::squaredDistanceFromValue(data.get(), value, data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("values") {
        memory::PtrHost<TestType> values(batches);
        test::initDataRandom(values.get(), values.elements(), randomizer);

        memory::PtrHost<TestType> expected(elements * batches);
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - values[batch]) *
                                                   (data[batch * elements + idx] - values[batch]);

        // Out of place.
        memory::PtrHost<TestType> results(elements * batches);
        math::squaredDistanceFromValue(data.get(), values.get(), results.get(), elements, batches);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::squaredDistanceFromValue(data.get(), values.get(), data.get(), elements, batches);
        diff = test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("array") {
        memory::PtrHost<TestType> array(elements);
        test::initDataRandom(array.get(), array.elements(), randomizer);

        memory::PtrHost<TestType> expected(elements * batches);
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - array[idx]) *
                                                   (data[batch * elements + idx] - array[idx]);

        // Out of place.
        memory::PtrHost<TestType> results(elements * batches);
        math::squaredDistanceFromArray(data.get(), array.get(), results.get(), elements, batches);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        math::squaredDistanceFromArray(data.get(), array.get(), data.get(), elements, batches);
        diff = test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}
