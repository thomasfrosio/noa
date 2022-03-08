#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::shift2D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["shift2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();
    const auto shift = param["shift"].as<float2_t>();

    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::SERIAL);
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
        cpu::geometry::shift2D(input.get(), stride, shape, output.get(), stride, shape,
                               shift, interp, border, border_value, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            // it seems that 1e-5f is fine as well
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cpu::geometry::shift3D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["shift3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto border_value = param["border_value"].as<float>();
    const auto shift = param["shift"].as<float3_t>();

    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::SERIAL);
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
        cpu::geometry::shift3D(input.get(), stride, shape, output.get(), stride, shape,
                               shift, interp, border, border_value, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
