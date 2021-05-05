#include <noa/cpu/masks/Cylinder.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// Just compare against manually checked data.
TEST_CASE("CPU::Mask - cylinder", "[noa][cpu][masks]") {
    Test::Randomizer<float> randomizer(-5, 5);
    path_t filename;
    MRCFile file;

    size3_t shape;
    float3_t shifts;
    float radius_xy{}, radius_z{};
    float taper{};

    int test_number = GENERATE(1, 2, 3);

    INFO("test number: " << test_number);
    Test::Assets::Mask::getCylinderParams(test_number, &filename, &shape, &shifts, &radius_xy, &radius_z, &taper);

    size_t elements = getElements(shape);
    Memory::PtrHost<float> mask_expected(elements);
    file.open(filename, IO::READ);
    file.readAll(mask_expected.get());

    Memory::PtrHost<float> input_expected(elements);
    Memory::PtrHost<float> input_result(elements);
    Memory::PtrHost<float> mask_result(elements);

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
