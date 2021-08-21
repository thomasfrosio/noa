#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Transpose.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::memory::transpose()", "[noa][cpu][memory]") {
    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    path_t filename_data;
    path_t filename_expected;
    size3_t shape;
    uint3_t permutation;
    bool in_place;
    test::assets::memory::getTransposeParams(test_number, &filename_data, &filename_expected,
                                             &shape, &permutation, &in_place);

    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> data(elements);
    cpu::memory::PtrHost<float> expected(elements);
    cpu::memory::PtrHost<float> result(elements);

    MRCFile file;
    file.open(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    if (in_place) {
        cpu::memory::transpose(data.get(), shape, data.get(), permutation, 1);
        float diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == 0);
    } else {
        cpu::memory::transpose(data.get(), shape, result.get(), permutation, 1);
        float diff = test::getDifference(expected.get(), result.get(), elements);
        REQUIRE(diff == 0);
    }
}
