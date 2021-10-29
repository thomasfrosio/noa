#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Rectangle.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just compare against manually checked data.
TEST_CASE("cpu::filter::rectangle()", "[assets][noa][cpu][filter]") {
    test::Randomizer<float> randomizer(-5, 5);
    path_t path_base = test::PATH_TEST_DATA / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "param.yaml")["rectangle"]["tests"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        auto shape = test["shape"].as<size3_t>();
        auto shift = test["shift"].as<float3_t>();
        auto radius = test["radius"].as<float3_t>();
        auto taper = test["taper"].as<float>();
        auto filename_expected = path_base / test["expected"].as<path_t>();

        file.open(filename_expected, io::READ);
        if (all(file.shape() != shape))
            FAIL("asset shape is not correct");
        size_t size = elements(shape);
        cpu::memory::PtrHost<float> mask_expected(size);
        file.readAll(mask_expected.get());

        cpu::memory::PtrHost<float> input_expected(size);
        cpu::memory::PtrHost<float> input_result(size);
        cpu::memory::PtrHost<float> mask_result(size);

        AND_THEN("invert = false") {
            if (invert) {
                invert = false;
                for (size_t idx = 0; idx < size; ++idx)
                    mask_expected[idx] = 1 - mask_expected[idx];
            }
            test::initDataRandom(input_expected.get(), size, randomizer);
            std::memcpy(input_result.get(), input_expected.get(), size * sizeof(float));

            // Test saving the mask.
            cpu::filter::rectangle(mask_result.get(), shape, shift, radius, taper);
            float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), size);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

            // Test on-the-fly, in-place.
            cpu::filter::rectangle(input_result.get(), input_result.get(), shape, shift, radius, taper, 1);
            for (size_t idx = 0; idx < size; ++idx)
                input_expected[idx] *= mask_expected[idx];
            diff = test::getAverageDifference(input_result.get(), input_expected.get(), size);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
        }

        AND_THEN("invert = true") {
            if (!invert)
                for (size_t idx = 0; idx < size; ++idx)
                    mask_expected[idx] = 1 - mask_expected[idx];

            test::initDataRandom(input_expected.get(), size, randomizer);
            std::memcpy(input_result.get(), input_expected.get(), size * sizeof(float));

            // Test saving the mask. Default should be invert=false
            cpu::filter::rectangle<true>(mask_result.get(), shape, shift, radius, taper);
            float diff = test::getAverageDifference(mask_expected.get(), mask_result.get(), size);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));

            // Test on-the-fly, in-place.
            cpu::filter::rectangle<true>(input_result.get(), input_result.get(), shape, shift, radius, taper, 1);
            for (size_t idx = 0; idx < size; ++idx)
                input_expected[idx] *= mask_expected[idx];
            diff = test::getAverageDifference(input_result.get(), input_expected.get(), size);
            REQUIRE_THAT(diff, test::isWithinAbs(float(0.), 1e-7));
        }
    }
}
