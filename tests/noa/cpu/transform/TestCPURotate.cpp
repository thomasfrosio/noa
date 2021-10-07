#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Rotate.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::rotate2D() -- vs scipy", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["rotate2D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto rotate = math::toRad(param["rotate"].as<float>());
    auto center = param["center"].as<float2_t>();

    MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.getShape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::rotate2D(input.get(), output.get(), size2_t(shape.x, shape.y),
                                 rotate, center, interp, border, border_value);

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-4f); // it is usually around 2e-5
            REQUIRE(math::abs(max) < 1e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEST_CASE("cpu::transform::rotate3D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["rotate3D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();
    auto euler = math::toRad(param["euler"].as<float3_t>());
    auto center = param["center"].as<float3_t>();

    float33_t matrix(transform::toMatrix<true>(euler));
    MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        auto expected_filename = path_base / test["expected"].as<path_t>();
        auto interp = test["interp"].as<InterpMode>();
        auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.getShape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::transform::rotate3D(input.get(), output.get(), shape,
                                 matrix, center, interp, border, border_value);

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-4f);
            REQUIRE(math::abs(max) < 1e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}
