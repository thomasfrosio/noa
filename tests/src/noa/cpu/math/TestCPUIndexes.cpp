#include <noa/cpu/math/Indexes.h>
#include <noa/cpu/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("CPU::Math: Indexes - first", "[noa][cpu][math]") {
    uint batches = 64;
    size_t elements = 4096;
    PtrHost<int> data(elements * batches);
    PtrHost<size_t> idx_min_expected(batches);
    PtrHost<size_t> idx_max_expected(batches);
    PtrHost<size_t> idx_results(batches);

    Test::Randomizer<int> randomizer(-100., 100.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    Test::IntRandomizer<size_t> randomizer_indexes(0, 3000);
    Test::initDataRandom(idx_min_expected.get(), batches, randomizer_indexes);
    Test::initDataRandom(idx_max_expected.get(), batches, randomizer_indexes);

    for (uint batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data[batch * elements + idx_min] = -101;
        data[batch * elements + idx_min + 500] = -101; // just to make sure it picks the first occurrence.

        size_t idx_max = idx_max_expected[batch];
        data[batch * elements + idx_max] = 101;
        data[batch * elements + idx_max + 500] = 101;
    }

    Math::firstMin(data.get(), idx_results.get(), elements, batches);
    size_t diff = Test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    Math::firstMax(data.get(), idx_results.get(), elements, batches);
    diff = Test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}

TEST_CASE("CPU::Math: Indexes - last", "[noa][cpu][math]") {
    uint batches = 64;
    size_t elements = 4096;
    PtrHost<int> data(elements * batches);
    PtrHost<size_t> idx_min_expected(batches);
    PtrHost<size_t> idx_max_expected(batches);
    PtrHost<size_t> idx_results(batches);

    Test::Randomizer<int> randomizer(-100., 100.);
    Test::initDataRandom(data.get(), data.elements(), randomizer);

    Test::IntRandomizer<size_t> randomizer_indexes(1000, 4095);
    Test::initDataRandom(idx_min_expected.get(), batches, randomizer_indexes);
    Test::initDataRandom(idx_max_expected.get(), batches, randomizer_indexes);

    for (uint batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data[batch * elements + idx_min] = -101;
        data[batch * elements + idx_min - 500] = -101; // just to make sure it picks the last occurrence.

        size_t idx_max = idx_max_expected[batch];
        data[batch * elements + idx_max] = 101;
        data[batch * elements + idx_max - 500] = 101;
    }

    Math::lastMin(data.get(), idx_results.get(), elements, batches);
    size_t diff = Test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    Math::lastMax(data.get(), idx_results.get(), elements, batches);
    diff = Test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}
