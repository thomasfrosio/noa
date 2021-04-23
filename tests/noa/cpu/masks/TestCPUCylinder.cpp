#include <noa/cpu/masks/Cylinder.h>

#include <noa/cpu/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// Just compare against manually checked data.
TEST_CASE("CPU::Mask - cylinder", "[noa][cpu][masks]") {
    Test::Randomizer<float> randomizer(-5, 5);
    path_t path_data = Test::PATH_TEST_DATA / "masks";
    MRCFile file;

    size3_t shape;
    float3_t shifts;
    float radius_xy{}, radius_z{};
    float taper{};

    int test_number = GENERATE(1, 2, 3);
    if (test_number == 1) {
        shape = {256, 256, 64};
        shifts = {0, 0, 0};
        radius_xy = 60;
        radius_z = 20;
        taper = 0;
        path_data /= "cylinder_01.mrc";
    } else if (test_number == 2) {
        shape = {128, 128, 128};
        shifts = {-11, 11, 0};
        radius_xy = 31;
        radius_z = 45;
        taper = 11;
        path_data /= "cylinder_02.mrc";
    } else if (test_number == 3) {
        shape = {80, 91, 180};
        shifts = {-6, 0, 10};
        radius_xy = 10;
        radius_z = 50;
        taper = 6;
        path_data /= "cylinder_03.mrc";
    }
    INFO("test number: " << test_number);

    size_t elements = getElements(shape);
    PtrHost<float> mask_expected(elements);
    file.open(path_data, IO::READ);
    file.readAll(mask_expected.get());

    PtrHost<float> input_expected(elements);
    PtrHost<float> input_result(elements);
    PtrHost<float> mask_result(elements);

    AND_THEN("invert = false") {
        Test::initDataRandom(input_expected.get(), elements, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

        // Test saving the mask.
        Mask::cylinder(mask_result.get(), shape, shifts, radius_xy, radius_z, taper);
        float diff = Test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-7));

        // Test on-the-fly, in-place.
        Mask::cylinder(input_result.get(), input_result.get(), shape, shifts, radius_xy, radius_z, taper, 1);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        diff = Test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-7));
    }

    AND_THEN("invert = true") {
        for (size_t idx = 0; idx < elements; ++idx)
            mask_expected[idx] = 1 - mask_expected[idx]; // test data is invert=true
        Test::initDataRandom(input_expected.get(), elements, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

        // Test saving the mask. Default should be invert=false
        Mask::cylinder<true>(mask_result.get(), shape, shifts, radius_xy, radius_z, taper);
        float diff = Test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-7));

        // Test on-the-fly, in-place.
        Mask::cylinder<true>(input_result.get(), input_result.get(), shape, shifts, radius_xy, radius_z, taper, 1);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        diff = Test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(float(0.), 1e-7));
    }
}
