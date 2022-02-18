#include <noa/cpu/math/Find.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::math::firstMin(), firstMax()", "[noa][cpu][math]") {
    size_t batches = 64;
    size_t elements = 4096;
    cpu::Stream stream;
    cpu::memory::PtrHost<int> data_min(elements * batches);
    cpu::memory::PtrHost<int> data_max(elements * batches);
    cpu::memory::PtrHost<size_t> idx_min_expected(batches);
    cpu::memory::PtrHost<size_t> idx_max_expected(batches);
    cpu::memory::PtrHost<size_t> idx_results(batches);

    test::Randomizer<int> randomizer(-100., 100.);
    test::randomize(data_min.get(), data_min.elements(), randomizer);
    test::randomize(data_max.get(), data_max.elements(), randomizer);

    test::Randomizer<size_t> randomizer_indexes(0, 3000);
    test::randomize(idx_min_expected.get(), batches, randomizer_indexes);
    test::randomize(idx_max_expected.get(), batches, randomizer_indexes);

    for (size_t batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data_min[batch * elements + idx_min] = -101;
        data_min[batch * elements + idx_min + 500] = -101; // just to make sure it picks the first occurrence.

        size_t idx_max = idx_max_expected[batch];
        data_max[batch * elements + idx_max] = 101;
        data_max[batch * elements + idx_max + 500] = 101;
    }

    cpu::math::firstMin(data_min.get(), elements, idx_results.get(), elements, batches, stream);
    size_t diff = test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    cpu::math::firstMax(data_max.get(), elements, idx_results.get(), elements, batches, stream);
    diff = test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}

TEST_CASE("cpu::math::lastMin(), lastMax()", "[noa][cpu][math]") {
    size_t batches = 64;
    size_t elements = 4096;
    cpu::Stream stream;
    cpu::memory::PtrHost<int> data_min(elements * batches);
    cpu::memory::PtrHost<int> data_max(elements * batches);
    cpu::memory::PtrHost<size_t> idx_min_expected(batches);
    cpu::memory::PtrHost<size_t> idx_max_expected(batches);
    cpu::memory::PtrHost<size_t> idx_results(batches);

    test::Randomizer<int> randomizer(-100., 100.);
    test::randomize(data_min.get(), data_min.elements(), randomizer);
    test::randomize(data_max.get(), data_max.elements(), randomizer);

    test::Randomizer<size_t> randomizer_indexes(1000, 4095);
    test::randomize(idx_min_expected.get(), batches, randomizer_indexes);
    test::randomize(idx_max_expected.get(), batches, randomizer_indexes);

    for (size_t batch = 0; batch < batches; ++batch) {
        size_t idx_min = idx_min_expected[batch];
        data_min[batch * elements + idx_min] = -101;
        data_min[batch * elements + idx_min - 500] = -101; // just to make sure it picks the last occurrence.

        size_t idx_max = idx_max_expected[batch];
        data_max[batch * elements + idx_max] = 101;
        data_max[batch * elements + idx_max - 500] = 101;
    }

    cpu::math::lastMin(data_min.get(), elements, idx_results.get(), elements, batches, stream);
    size_t diff = test::getDifference(idx_min_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);

    cpu::math::lastMax(data_max.get(), elements, idx_results.get(), elements, batches, stream);
    diff = test::getDifference(idx_max_expected.get(), idx_results.get(), batches);
    REQUIRE(diff == 0);
}
