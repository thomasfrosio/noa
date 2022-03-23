#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Shape.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Just compare against manually checked data.
TEST_CASE("cpu::filter::cylinder()", "[assets][noa][cpu][filter]") {
    test::Randomizer<float> randomizer(-5, 5);
    const path_t path_base = test::NOA_DATA_PATH / "filter";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["tests"];
    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<size4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float>();
        const auto length = test["length"].as<float>();
        const auto taper = test["taper"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();

        file.open(filename_expected, io::READ);
        if (all(file.shape() != shape))
            FAIL("asset shape is not correct");
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

            test::randomize(input_expected.get(), elements, randomizer);
            test::copy(input_expected.get(), input_result.get(), elements);

            // Test saving the mask.
            cpu::filter::cylinder<false, float>(nullptr, {}, mask_result.get(), stride, shape,
                                                center, radius, length, taper, stream);
            REQUIRE(test::Matcher(test::MATCH_ABS, mask_expected.get(), mask_result.get(), elements, 1e-4));

            // Test on-the-fly, in-place.
            cpu::filter::cylinder<false>(input_result.get(), stride, input_result.get(), stride, shape,
                                         center, radius, length, taper, stream);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= mask_expected[idx];
            REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-4));
        }

        AND_THEN("invert = true") {
            if (!invert)
                for (size_t idx = 0; idx < elements; ++idx)
                    mask_expected[idx] = 1 - mask_expected[idx];

            test::randomize(input_expected.get(), elements, randomizer);
            test::copy(input_expected.get(), input_result.get(), elements);

            // Test saving the mask. Default should be invert=false
            cpu::filter::cylinder<true, float>(
                    nullptr, {}, mask_result.get(), stride, shape, center, radius, length, taper, stream);
            REQUIRE(test::Matcher(test::MATCH_ABS, mask_expected.get(), mask_result.get(), elements, 1e-4));

            // Test on-the-fly, in-place.
            cpu::filter::cylinder<true>(input_result.get(), stride, input_result.get(), stride, shape,
                                        center, radius, length, taper, stream);
            for (size_t idx = 0; idx < elements; ++idx)
                input_expected[idx] *= mask_expected[idx];
            REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 1e-4));
        }
    }
}
