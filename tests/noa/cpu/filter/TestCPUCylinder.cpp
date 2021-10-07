#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Cylinder.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just compare against manually checked data.
TEST_CASE("cpu::filter::cylinder()", "[assets][noa][cpu][filter]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t path_base = test::PATH_TEST_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["cylinder"]["tests"];
    MRCFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        auto shape = test["shape"].as<size3_t>();
        auto shift = test["shift"].as<float3_t>();
        auto radius_xy = test["radius_xy"].as<float>();
        auto radius_z = test["radius_z"].as<float>();
        auto taper = test["taper"].as<float>();
        auto filename_expected = path_base / test["expected"].as<path_t>();

        file.open(filename_expected, io::READ);
        if (all(file.getShape() != shape))
            FAIL("asset shape is not correct");
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> mask_expected(elements);
        file.readAll(mask_expected.get());

        cpu::memory::PtrHost<float> input_expected(elements);
        cpu::memory::PtrHost<float> input_result(elements);
        cpu::memory::PtrHost<float> mask_result(elements);

        AND_THEN("invert = false") {
            if (invert) {
                invert = false;
                for (size_t idx = 0; idx < elements; ++idx)
                    mask_expected[idx] = 1 - mask_expected[idx];
            }

            test::initDataRandom(input_expected.get(), elements, randomizer);
            std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

            // Test saving the mask.
            cpu::filter::cylinder(mask_result.get(), shape, shift, radius_xy, radius_z, taper);
            float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

            // Test on-the-fly, in-place.
            cpu::filter::cylinder(input_result.get(), input_result.get(), shape, shift, radius_xy, radius_z, taper, 1);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= mask_expected[idx];
            diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
        }

        AND_THEN("invert = true") {
            if (!invert)
                for (size_t idx = 0; idx < elements; ++idx)
                    mask_expected[idx] = 1 - mask_expected[idx];

            test::initDataRandom(input_expected.get(), elements, randomizer);
            std::memcpy(input_result.get(), input_expected.get(), elements * sizeof(float));

            // Test saving the mask. Default should be invert=false
            cpu::filter::cylinder<true>(mask_result.get(), shape, shift, radius_xy, radius_z, taper);
            float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

            // Test on-the-fly, in-place.
            cpu::filter::cylinder<true>(input_result.get(), input_result.get(), shape, shift,
                                        radius_xy, radius_z, taper, 1);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= mask_expected[idx];
            diff = test::getAverageDifference(input_result.get(), input_expected.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
        }
    }
}
