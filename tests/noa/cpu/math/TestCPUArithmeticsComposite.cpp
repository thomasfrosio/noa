#include <noa/cpu/math/ArithmeticsComposite.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math::multiplyAdd()", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t batches = test::Randomizer<size_t>(1, 4).get();
    size_t elements = test::Randomizer<size_t>(1, 100).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    cpu::memory::PtrHost<TestType> multiplicands(elements);
    cpu::memory::PtrHost<TestType> addends(elements);
    test::randomize(data.get(), data.elements(), randomizer);
    test::randomize(multiplicands.get(), multiplicands.elements(), randomizer);
    test::randomize(addends.get(), addends.elements(), randomizer);

    cpu::memory::PtrHost<TestType> expected(elements * batches);
    for (size_t batch{0}; batch < batches; ++batch)
        for (size_t idx{0}; idx < elements; ++idx)
            expected[batch * elements + idx] = data[batch * elements + idx] * multiplicands[idx] + addends[idx];

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements * batches);
    cpu::math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), results.get(), elements, batches);
    TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::multiplyAddArray(data.get(), multiplicands.get(), addends.get(), data.get(), elements, batches);
    diff = test::getDifference(expected.get(), data.get(), elements * batches);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("cpu::math::squaredDifference*()", "[noa][cpu][math]",
                   int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t batches = test::Randomizer<size_t>(1, 5).get();
    size_t elements = test::Randomizer<size_t>(0, 100).get();

    cpu::memory::PtrHost<TestType> data(elements * batches);
    test::randomize(data.get(), data.elements(), randomizer);

    AND_THEN("value") {
        TestType value = randomizer.get();

        cpu::memory::PtrHost<TestType> expected(elements);
        for (size_t idx{0}; idx < elements; ++idx)
            expected[idx] = (data[idx] - value) * (data[idx] - value);

        // Out of place.
        cpu::memory::PtrHost<TestType> results(elements);
        cpu::math::squaredDistanceFromValue(data.get(), value, results.get(), elements);
        TestType diff = test::getDifference(expected.get(), results.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::squaredDistanceFromValue(data.get(), value, data.get(), elements);
        diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("values") {
        cpu::memory::PtrHost<TestType> values(batches);
        test::randomize(values.get(), values.elements(), randomizer);

        cpu::memory::PtrHost<TestType> expected(elements * batches);
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - values[batch]) *
                                                   (data[batch * elements + idx] - values[batch]);

        // Out of place.
        cpu::memory::PtrHost<TestType> results(elements * batches);
        cpu::math::squaredDistanceFromValue(data.get(), values.get(), results.get(), elements, batches);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::squaredDistanceFromValue(data.get(), values.get(), data.get(), elements, batches);
        diff = test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }

    AND_THEN("array") {
        cpu::memory::PtrHost<TestType> array(elements);
        test::randomize(array.get(), array.elements(), randomizer);

        cpu::memory::PtrHost<TestType> expected(elements * batches);
        for (size_t batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                expected[batch * elements + idx] = (data[batch * elements + idx] - array[idx]) *
                                                   (data[batch * elements + idx] - array[idx]);

        // Out of place.
        cpu::memory::PtrHost<TestType> results(elements * batches);
        cpu::math::squaredDistanceFromArray(data.get(), array.get(), results.get(), elements, batches);
        TestType diff = test::getDifference(expected.get(), results.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic

        // In place.
        cpu::math::squaredDistanceFromArray(data.get(), array.get(), data.get(), elements, batches);
        diff = test::getDifference(expected.get(), data.get(), elements * batches);
        REQUIRE(diff == TestType(0)); // this should be deterministic
    }
}
