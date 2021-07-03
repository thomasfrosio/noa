#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/mask/Sphere.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just compare against manually checked data.
TEST_CASE("mask::sphere()", "[noa][cpu][masks]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t filename = test::PATH_TEST_DATA / "masks";
    MRCFile file;

    size3_t shape;
    float3_t shifts;
    float radius{};
    float taper{};

    int test_number = GENERATE(1, 2, 3, 4, 5);
    INFO("test number: " << test_number);
    test::assets::mask::getSphereParams(test_number, &filename, &shape, &shifts, &radius, &taper);

    size_t elements = getElements(shape);
    memory::PtrHost<float> mask_expected(elements);
    file.open(filename, io::READ);
    file.readAll(mask_expected.get());

    memory::PtrHost<float> input_expected(elements);
    memory::PtrHost<float> input_result(elements);
    memory::PtrHost<float> mask_result(elements);

    AND_THEN("invert = true") {
        test::initDataRandom(input_expected.get(), elements, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

        // Test saving the mask.
        mask::sphere<true>(mask_result.get(), shape, shifts, radius, taper);
        float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

        // Test on-the-fly, in-place.
        mask::sphere<true>(input_result.get(), input_result.get(), shape, shifts, radius, taper, 1);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
    }

    AND_THEN("invert = false") {
        for (size_t idx = 0; idx < elements; ++idx)
            mask_expected[idx] = 1 - mask_expected[idx]; // test data is invert=true
        test::initDataRandom(input_expected.get(), elements, randomizer);
        std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

        // Test saving the mask. Default should be invert=false
        mask::sphere(mask_result.get(), shape, shifts, radius, taper);
        float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

        // Test on-the-fly, in-place.
        mask::sphere(input_result.get(), input_result.get(), shape, shifts, radius, taper, 1);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
    }
}
