#include <noa/cpu/math/Booleans.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math::isLess()", "[noa][cpu][math]", int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = test::Randomizer<size_t>(1, 100).get();

    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), elements, randomizer);
    TestType value = randomizer.get();

    cpu::memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] < value;

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::math::isLess(data.get(), value, results.get(), elements);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::isLess(data.get(), value, data.get(), elements);
    diff = test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("cpu::math::isGreater()", "[noa][cpu][math]", int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = test::Randomizer<size_t>(1, 100).get();

    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), elements, randomizer);
    TestType value = randomizer.get();

    cpu::memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] > value;

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::math::isGreater(data.get(), value, results.get(), elements);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::isGreater(data.get(), value, data.get(), elements);
    diff = test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("cpu::math::isWithin()", "[noa][cpu][math]", int, uint, float, double) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = test::Randomizer<size_t>(1, 100).get();

    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), elements, randomizer);
    TestType low = test::Randomizer<TestType>(0, 20).get();
    TestType high = test::Randomizer<TestType>(80, 100).get();

    cpu::memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] > low && high > data[idx];

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::math::isWithin(data.get(), low, high, results.get(), elements);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::isWithin(data.get(), low, high, data.get(), elements);
    diff = test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("cpu::math::logicNOT()", "[noa][cpu][math]", int, uint) {
    test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = test::Randomizer<size_t>(1, 100).get();

    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), elements, randomizer);

    cpu::memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = !data[idx];

    // Out of place.
    cpu::memory::PtrHost<TestType> results(elements);
    cpu::math::logicNOT(data.get(), results.get(), elements);
    TestType diff = test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    cpu::math::logicNOT(data.get(), data.get(), elements);
    diff = test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}
