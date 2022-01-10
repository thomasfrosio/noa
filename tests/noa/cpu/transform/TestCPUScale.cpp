#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/transform/Scale.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::scale2D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["scale2D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto scale = param["scale"].as<float2_t>();
    auto center = param["center"].as<float2_t>();

    io::ImageFile file;
    cpu::Stream stream;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size2_t shape_2d = {shape.x, shape.y};
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::scale2D(input.get(), shape.x, output.get(), shape.x, shape_2d,
                                scale, center, interp, border, border_value, stream);

        if (interp == INTERP_LINEAR) {
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cpu::transform::scale3D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["scale3D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto scale = param["scale"].as<float3_t>();
    auto center = param["center"].as<float3_t>();

    io::ImageFile file;
    cpu::Stream stream;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size2_t shape_2d = {shape.x, shape.y};
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::scale3D(input.get(), shape_2d, output.get(), shape_2d, shape,
                                scale, center, interp, border, border_value, stream);

        if (interp == INTERP_LINEAR) {
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
