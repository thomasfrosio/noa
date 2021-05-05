#include <noa/cpu/math/Booleans.h>

#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU: Booleans: isLess", "[noa][cpu][math]", int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();

    Memory::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);
    TestType value = randomizer.get();

    Memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] < value;

    // Out of place.
    Memory::PtrHost<TestType> results(elements);
    Math::isLess(data.get(), value, results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::isLess(data.get(), value, data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Booleans: isGreater", "[noa][cpu][math]", int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();

    Memory::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);
    TestType value = randomizer.get();

    Memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] > value;

    // Out of place.
    Memory::PtrHost<TestType> results(elements);
    Math::isGreater(data.get(), value, results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::isGreater(data.get(), value, data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Booleans: isWithin", "[noa][cpu][math]", int, uint, float, double) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();

    Memory::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);
    TestType low = Test::Randomizer<TestType>(0, 20).get();
    TestType high = Test::Randomizer<TestType>(80, 100).get();

    Memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = data[idx] > low && high > data[idx];

    // Out of place.
    Memory::PtrHost<TestType> results(elements);
    Math::isWithin(data.get(), low, high, results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::isWithin(data.get(), low, high, data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}

TEMPLATE_TEST_CASE("CPU: Booleans: logicNOT", "[noa][cpu][math]", int, uint) {
    Test::Randomizer<TestType> randomizer(1., 100.);
    size_t elements = Test::IntRandomizer<size_t>(1, 100).get();

    Memory::PtrHost<TestType> data(elements);
    Test::initDataRandom(data.get(), elements, randomizer);

    Memory::PtrHost<TestType> expected(elements);
    for (size_t idx{0}; idx < elements; ++idx)
        expected[idx] = !data[idx];

    // Out of place.
    Memory::PtrHost<TestType> results(elements);
    Math::logicNOT(data.get(), results.get(), elements);
    TestType diff = Test::getDifference(expected.get(), results.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic

    // In place.
    Math::logicNOT(data.get(), data.get(), elements);
    diff = Test::getDifference(expected.get(), data.get(), elements);
    REQUIRE(diff == TestType(0)); // this should be deterministic
}
