#include <noa/cpu/math/Indexes.h>

#include <noa/cpu/PtrHost.h>
#include <noa/Math.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CPU::Math: Indexes", "[noa][cpu][math]", int, float, double) {
    Test::IntRandomizer<size_t> size_randomizer(1, 1000);
    size_t elements = size_randomizer.get();
    PtrHost<TestType> data(elements);

    Test::Randomizer<TestType> randomizer(-100., 100.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    size_t expected_idx = elements / 2;

    TestType expected_value = -101;
    data[expected_idx] = expected_value;
    auto[result_idx_min, result_min] = Math::firstMin(data.get(), elements);
    REQUIRE(expected_idx == result_idx_min);
    REQUIRE(expected_value == result_min);

    expected_value = 101;
    data[expected_idx] = expected_value;
    auto[result_idx_max, result_max] = Math::firstMax(data.get(), elements);
    REQUIRE(expected_idx == result_idx_max);
    REQUIRE(expected_value == result_max);
}
