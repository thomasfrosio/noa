#include <noa/cpu/math/Indexes.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("math::firstMin(), firstMax()", "[noa][cpu][math]") {
    uint batches = 64;
    size_t elements = 4096;
    memory::PtrHost<int> data_min(elements * batches);
    memory::PtrHost<int> data_max(elements * batches);
    memory::PtrHost<size_t> idx_min_expected(batches);
    memory::PtrHost<size_t> idx_max_expected(batches);
    memory::PtrHost<size_t> idx_results(batches);

    test::Randomizer<int> randomizer(-100., 100.);
    test::initDataRandom(data_min.get(), data_min.elements(), randomizer);
    test::initDataRandom(data_max.get(), data_max.elements(), randomizer);

    test::IntRandomizer<size_t> randomizer_indexes(0, 3000);
    test::initDataRandom(idx_min_expected.get(), batches, randomizer_indexes);
    test::initDataRandom(idx_max_expected.get(), batches, randomizer_indexes);

    for (uint batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data_min[batch * elements + idx_min] = -101;
        data_min[batch * elements + idx_min + 500] = -101; // just to make sure it picks the first occurrence.

        size_t idx_max = idx_max_expected[batch];
        data_max[batch * elements + idx_max] = 101;
        data_max[batch * elements + idx_max + 500] = 101;
    }

    math::firstMin(data_min.get(), idx_results.get(), elements, batches);
    size_t diff = test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    math::firstMax(data_max.get(), idx_results.get(), elements, batches);
    diff = test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}

TEST_CASE("math::lastMin(), lastMax()", "[noa][cpu][math]") {
    uint batches = 64;
    size_t elements = 4096;
    memory::PtrHost<int> data_min(elements * batches);
    memory::PtrHost<int> data_max(elements * batches);
    memory::PtrHost<size_t> idx_min_expected(batches);
    memory::PtrHost<size_t> idx_max_expected(batches);
    memory::PtrHost<size_t> idx_results(batches);

    test::Randomizer<int> randomizer(-100., 100.);
    test::initDataRandom(data_min.get(), data_min.elements(), randomizer);
    test::initDataRandom(data_max.get(), data_max.elements(), randomizer);

    test::IntRandomizer<size_t> randomizer_indexes(1000, 4095);
    test::initDataRandom(idx_min_expected.get(), batches, randomizer_indexes);
    test::initDataRandom(idx_max_expected.get(), batches, randomizer_indexes);

    for (uint batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data_min[batch * elements + idx_min] = -101;
        data_min[batch * elements + idx_min - 500] = -101; // just to make sure it picks the last occurrence.

        size_t idx_max = idx_max_expected[batch];
        data_max[batch * elements + idx_max] = 101;
        data_max[batch * elements + idx_max - 500] = 101;
    }

    math::lastMin(data_min.get(), idx_results.get(), elements, batches);
    size_t diff = test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    math::lastMax(data_max.get(), idx_results.get(), elements, batches);
    diff = test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}