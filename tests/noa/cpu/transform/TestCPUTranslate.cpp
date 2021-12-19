#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Translate.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::translate2D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["translate2D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto shift = param["shift"].as<float2_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::translate2D(input.get(), size2_t(shape.x, shape.y), output.get(), size2_t(shape.x, shape.y),
                                    shift, interp, border, border_value);

        if (interp == INTERP_LINEAR) {
            // it seems that 1e-5f is fine as well
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cpu::transform::translate3D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["translate3D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto shift = param["shift"].as<float3_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::translate3D(input.get(), shape, output.get(), shape,
                                    shift, interp, border, border_value);

        if (interp == INTERP_LINEAR) {
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
