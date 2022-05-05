#include <noa/cpu/math/Find.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math::find(), batched", "[noa][cpu][math]", int32_t, uint64_t, float, double) {
    const size4_t shape{3, 64, 128, 128};
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size_t elements_unbatched = shape[1] * shape[2] * shape[3];
    cpu::Stream stream;

    cpu::memory::PtrHost<int> data_min(elements);
    cpu::memory::PtrHost<int> data_max(elements);
    cpu::memory::PtrHost<size_t> offset_min_expected(shape[0]);
    cpu::memory::PtrHost<size_t> offset_max_expected(shape[0]);
    cpu::memory::PtrHost<size_t> offset_results(shape[0]);

    test::Randomizer<int> randomizer(-100., 100.);
    test::randomize(data_min.get(), data_min.elements(), randomizer);
    test::randomize(data_max.get(), data_max.elements(), randomizer);

    test::Randomizer<size_t> randomizer_indexes(size_t{0}, elements_unbatched - 100);
    test::randomize(offset_min_expected.get(), shape[0], randomizer_indexes);
    test::randomize(offset_max_expected.get(), shape[0], randomizer_indexes);

    for (size_t batch = 0; batch < shape[0]; ++batch) {
        size_t offset_min = offset_min_expected[batch];
        data_min[batch * elements_unbatched + offset_min] = -101;
        data_min[batch * elements_unbatched + offset_min + 50] = -101; // just to make sure it picks the first occurrence.

        size_t offset_max = offset_max_expected[batch];
        data_max[batch * elements_unbatched + offset_max] = 101;
        data_max[batch * elements_unbatched + offset_max + 50] = 101;
    }

    cpu::math::find(math::min_t{}, data_min.share(), stride, shape, offset_results.share(), true, stream);
    stream.synchronize();
    size_t diff = test::getDifference(offset_min_expected.get(), offset_results.get(), shape[0]);
    REQUIRE(diff == 0);

    size_t offset = cpu::math::find(math::min_t{}, data_min.share(), elements, stream);
    REQUIRE(offset == offset_min_expected[0]);

    cpu::math::find(math::max_t{}, data_max.share(), stride, shape, offset_results.share(), true, stream);
    stream.synchronize();
    diff = test::getDifference(offset_max_expected.get(), offset_results.get(), shape[0]);
    REQUIRE(diff == 0);

    offset = cpu::math::find(math::max_t{}, data_max.share(), elements, stream);
    REQUIRE(offset == offset_max_expected[0]);
}
