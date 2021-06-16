#include <noa/cpu/memory/PtrHost.h>
#include <noa/io/files/MRCFile.h>
#include <noa/cpu/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("Filter::median", "[noa][cpu][filter]") {
    using namespace Noa;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_expected;
    uint window;
    BorderMode mode;
    size3_t shape;

    Test::Assets::Filter::getMedianData(test_number, &filename_data);
    Test::Assets::Filter::getMedianParams(test_number, &filename_expected, &shape, &mode, &window);

    size_t elements = getElements(shape);
    Memory::PtrHost<float> data(elements);
    Memory::PtrHost<float> expected(elements);
    Memory::PtrHost<float> result(elements);

    MRCFile file(filename_data, IO::READ);
    file.readAll(data.get());
    file.open(filename_expected, IO::READ);
    file.readAll(expected.get());

    if (test_number < 5)
        Filter::median1(data.get(), result.get(), shape, 1, mode, window);
    else if (test_number < 9)
        Filter::median2(data.get(), result.get(), shape, 1, mode, window);
    else
        Filter::median3(data.get(), result.get(), shape, 1, mode, window);

    float diff = Test::getAverageDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, Test::isWithinRel(0.f, 1e-6));
}
