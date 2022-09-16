#include <noa/common/io/MRCFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/Transform.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::transform2D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();

    const auto center = param["center"].as<float2_t>();
    const auto scale = param["scale"].as<float2_t>();
    const auto rotate = math::deg2rad(param["rotate"].as<float>());
    const auto shift = param["shift"].as<float2_t>();
    float33_t matrix(geometry::translate(center) *
                     geometry::translate(shift) *
                     float33_t(geometry::rotate(rotate)) *
                     float33_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::Stream stream;
        { // 3x3 matrix
            cpu::geometry::transform2D(input.share(), stride, shape, output.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            if (interp == INTERP_LINEAR) {
                // sometimes it is slightly higher than 1e-4
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }

        { // 2x3 matrix
            cpu::geometry::transform2D(input.share(), stride, shape, output.share(), stride, shape,
                                       float23_t(matrix), interp, border, border_value, true, stream);
            stream.synchronize();

            if (interp == INTERP_LINEAR) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::geometry::transform2D(), cubic", "[assets][noa][cpu][transform]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D_cubic"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();

    const auto center = param["center"].as<float2_t>();
    const auto scale = param["scale"].as<float2_t>();
    const auto rotate = math::deg2rad(param["rotate"].as<float>());
    float33_t matrix(geometry::translate(center) *
                     float33_t(geometry::rotate(rotate)) *
                     float33_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        cpu::Stream stream;
        if constexpr (GENERATE_TEST_DATA) {
            cpu::geometry::transform2D(input.share(), stride, shape, expected.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            file.open(expected_filename, io::READ);
            file.shape(shape);
            file.writeAll(expected.get());
        } else {
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());

            cpu::memory::PtrHost<float> output(elements);
            cpu::geometry::transform2D(input.share(), stride, shape, output.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
        }
    }
}

TEST_CASE("cpu::geometry::transform3D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();

    const auto center = param["center"].as<float3_t>();
    const auto scale = param["scale"].as<float3_t>();
    const auto euler = math::deg2rad(param["euler"].as<float3_t>());
    const auto shift = param["shift"].as<float3_t>();
    float44_t matrix(geometry::translate(center) *
                     geometry::translate(shift) *
                     float44_t(geometry::euler2matrix(euler)) * // ZYZ intrinsic right-handed
                     float44_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cpu::memory::PtrHost<float> output(elements);
        cpu::Stream stream;
        {
            cpu::geometry::transform3D(input.share(), stride, shape, output.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            if (interp == INTERP_LINEAR) {
                // it's mostly around 5e-5
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }

        {
            cpu::geometry::transform3D(input.share(), stride, shape, output.share(), stride, shape,
                                       float34_t(matrix), interp, border, border_value, true, stream);
            stream.synchronize();
            if (interp == INTERP_LINEAR) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::geometry::transform3D(), cubic", "[assets][noa][cpu][geometry]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D_cubic"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();

    const auto center = param["center"].as<float3_t>();
    const auto scale = param["scale"].as<float3_t>();
    const auto euler = math::deg2rad(param["euler"].as<float3_t>());
    float44_t matrix(geometry::translate(center) *
                     float44_t(geometry::euler2matrix(euler)) * // ZYZ intrinsic right-handed
                     float44_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        cpu::memory::PtrHost<float> expected(elements);
        cpu::Stream stream;
        if constexpr (GENERATE_TEST_DATA) {
            cpu::geometry::transform3D(input.share(), stride, shape, expected.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            file.open(expected_filename, io::READ);
            file.shape(shape);
            file.writeAll(expected.get());
        } else {
            file.open(expected_filename, io::READ);
            file.readAll(expected.get());

            cpu::memory::PtrHost<float> output(elements);
            cpu::geometry::transform3D(input.share(), stride, shape, output.share(), stride, shape,
                                       matrix, interp, border, border_value, true, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
        }
    }
}
