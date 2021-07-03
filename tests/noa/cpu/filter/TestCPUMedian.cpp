#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Median.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

TEST_CASE("filter::median()", "[noa][cpu][filter]") {
    using namespace noa;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    INFO("test_number = " << test_number);
    path_t filename_data;
    path_t filename_expected;
    uint window;
    BorderMode mode;
    size3_t shape;

    test::assets::filter::getMedianData(test_number, &filename_data);
    test::assets::filter::getMedianParams(test_number, &filename_expected, &shape, &mode, &window);

    size_t elements = getElements(shape);
    memory::PtrHost<float> data(elements);
    memory::PtrHost<float> expected(elements);
    memory::PtrHost<float> result(elements);

    MRCFile file(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    if (test_number < 5)
        filter::median1(data.get(), result.get(), shape, 1, mode, window);
    else if (test_number < 9)
        filter::median2(data.get(), result.get(), shape, 1, mode, window);
    else
        filter::median3(data.get(), result.get(), shape, 1, mode, window);

    float diff = test::getAverageDifference(expected.get(), result.get(), elements);
    REQUIRE_THAT(diff, test::isWithinRel(0.f, 1e-6));
}
