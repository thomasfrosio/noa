#include <noa/common/io/ImageFile.h>
#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::apply2D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["apply2D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();

    auto center = param["center"].as<float2_t>();
    auto scale = param["scale"].as<float2_t>();
    auto rotate = math::toRad(param["rotate"].as<float>());
    auto shift = param["shift"].as<float2_t>();
    float33_t matrix(transform::translate(center) *
                     transform::translate(shift) *
                     float33_t(transform::rotate(rotate)) *
                     float33_t(transform::scale(scale)) *
                     transform::translate(-center));
    matrix = math::inverse(matrix);

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
        { // 3x3 matrix
            cpu::transform::apply2D(input.get(), {shape.x, shape.y}, output.get(), {shape.x, shape.y},
                                    matrix, interp, border, border_value);
            if (interp == INTERP_LINEAR) {
                // sometimes it is slightly higher than 1e-4
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }

        { // 2x3 matrix
            cpu::transform::apply2D(input.get(), {shape.x, shape.y}, output.get(), {shape.x, shape.y},
                                    float23_t(matrix), interp, border, border_value);

            if (interp == INTERP_LINEAR) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::transform::apply2D(), cubic", "[assets][noa][cpu][transform]") {
    constexpr bool GENERATE_TEST_DATA = false;
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["apply2D_cubic"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();

    auto center = param["center"].as<float2_t>();
    auto scale = param["scale"].as<float2_t>();
    auto rotate = math::toRad(param["rotate"].as<float>());
    float33_t matrix(transform::translate(center) *
                     float33_t(transform::rotate(rotate)) *
                     float33_t(transform::scale(scale)) *
                     transform::translate(-center));
    matrix = math::inverse(matrix);

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

        cpu::memory::PtrHost<float> expected(elements);
        if constexpr (GENERATE_TEST_DATA) {
            cpu::transform::apply2D(input.get(), {shape.x, shape.y}, expected.get(), {shape.x, shape.y},
                                    matrix, interp, border, border_value);
            file.open(expected_filename, io::READ);
            file.shape(shape);
            file.writeAll(expected.get());
        } else {
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());

            cpu::memory::PtrHost<float> output(elements);
            cpu::transform::apply2D(input.get(), {shape.x, shape.y}, output.get(), {shape.x, shape.y}, matrix,
                                    interp, border, border_value);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
        }
    }
}

TEST_CASE("cpu::transform::apply3D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["apply3D"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();

    auto center = param["center"].as<float3_t>();
    auto scale = param["scale"].as<float3_t>();
    auto euler = math::toRad(param["euler"].as<float3_t>());
    auto shift = param["shift"].as<float3_t>();
    float44_t matrix(transform::translate(center) *
                     transform::translate(shift) *
                     float44_t(transform::toMatrix(euler)) *
                     float44_t(transform::scale(scale)) *
                     transform::translate(-center));
    matrix = math::inverse(matrix);

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
        {
            cpu::transform::apply3D(input.get(), shape, output.get(), shape, matrix, interp, border, border_value);
            if (interp == INTERP_LINEAR) {
                // it's mostly around 5e-5
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }

        {
            cpu::transform::apply3D(input.get(), shape, output.get(), shape, float34_t(matrix),
                                    interp, border, border_value);
            if (interp == INTERP_LINEAR) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::transform::apply3D(), cubic", "[assets][noa][cpu][transform]") {
    constexpr bool GENERATE_TEST_DATA = false;
    path_t path_base = test::PATH_NOA_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["apply3D_cubic"];
    auto input_filename = path_base / param["input"].as<path_t>();
    auto border_value = param["border_value"].as<float>();

    auto center = param["center"].as<float3_t>();
    auto scale = param["scale"].as<float3_t>();
    auto euler = math::toRad(param["euler"].as<float3_t>());
    float44_t matrix(transform::translate(center) *
                     float44_t(transform::toMatrix(euler)) *
                     float44_t(transform::scale(scale)) *
                     transform::translate(-center));
    matrix = math::inverse(matrix);

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

        cpu::memory::PtrHost<float> expected(elements);
        if constexpr (GENERATE_TEST_DATA) {
            cpu::transform::apply3D(input.get(), shape, expected.get(), shape,
                                    matrix, interp, border, border_value);
            file.open(expected_filename, io::READ);
            file.shape(shape);
            file.writeAll(expected.get());
        } else {
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());

            cpu::memory::PtrHost<float> output(elements);
            cpu::transform::apply3D(input.get(), shape, output.get(), shape, matrix,
                                    interp, border, border_value);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
        }
    }
}
