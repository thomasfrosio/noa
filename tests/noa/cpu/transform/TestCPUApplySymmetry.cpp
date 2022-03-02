#include <noa/common/io/ImageFile.h>

#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>

#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Shape.h>
#include <noa/cpu/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::transform2D() - symmetry", "[assets][noa][cpu][geometry]") {
    const path_t base_path = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(base_path / "tests.yaml")["transform2D_symmetry"];
    const path_t input_path = base_path / param["input"].as<path_t>();
    io::ImageFile file;

    cpu::Stream stream(cpu::Stream::SERIAL);

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const size4_t shape{1, 1, 512, 512};
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);

        const float3_t center{size3_t{shape.get() + 1} / 2};
        cpu::filter::rectangle<false, float>(
                nullptr, {}, input.get(), stride, shape, center, {1, 64, 128}, 5, stream);
        cpu::memory::PtrHost<float> tmp(elements);
        cpu::filter::rectangle<false, float>(
                nullptr, {}, tmp.get(), stride, shape, center + float3_t{0, 64, 128}, {1, 32, 32}, 3, stream);
        cpu::math::ewise(input.get(), stride, tmp.get(), stride, input.get(), stride, shape, math::plus_t{}, stream);
        stream.synchronize();

        file.open(input_path, io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);
    }

    for (size_t i = 0; i < param["tests"].size(); ++i) {
        INFO("test number = " << i);

        // Parameters:
        const auto current = param["tests"][i];
        const auto filename_expected = base_path / current["expected"].as<path_t>();
        const auto shift = current["shift"].as<float2_t>();
        const float22_t matrix = geometry::rotate(-math::toRad(current["angle"].as<float>())); // inverse
        const geometry::Symmetry symmetry(current["symmetry"].as<std::string>());
        const auto center = current["center"].as<float2_t>();
        const auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(input_path, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        cpu::memory::PtrHost<float> output(elements);

        file.readAll(input.get());
        cpu::geometry::transform2D(input.get(), stride, shape, output.get(), stride, shape,
                                   shift, matrix, symmetry, center, interp, true, stream);
        stream.synchronize();

        if constexpr (COMPUTE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(output.get(), false);
        } else {
            cpu::memory::PtrHost<float> expected(elements);
            file.open(filename_expected, io::READ);
            file.readAll(expected.get());
            if (interp != INTERP_NEAREST) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::geometry::transform3D() - symmetry", "[assets][noa][cpu][geometry]") {
    const path_t base_path = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(base_path / "tests.yaml")["transform3D_symmetry"];
    io::ImageFile file;

    cpu::Stream stream(cpu::Stream::SERIAL);

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const size4_t shape{1, 150, 150, 150};
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);

        const float3_t rectangle_center{size3_t{shape.get() + 1} / 2};
        cpu::filter::rectangle<false, float>(
                nullptr, {}, input.get(), stride, shape, rectangle_center, {34, 24, 24}, 3, stream);
        stream.synchronize();
        file.open(base_path / param["input"][0].as<path_t>(), io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);

        cpu::memory::PtrHost<float> tmp(elements);
        cpu::filter::rectangle<false, float>(nullptr, {}, tmp.get(), stride, shape,
                                             rectangle_center + float3_t{50, 34, 34}, {15, 15, 15}, 3, stream);
        cpu::math::ewise(input.get(), stride, tmp.get(), stride, input.get(), stride,  shape, math::plus_t{}, stream);
        stream.synchronize();
        file.open(base_path / param["input"][1].as<path_t>(), io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);
    }

    for (size_t i = 0; i < param["tests"].size(); ++i) { //
        INFO("test number = " << i);

        // Parameters:
        const auto current = param["tests"][i];
        const auto filename_expected = base_path / current["expected"].as<path_t>();
        const auto filename_input = base_path / current["input"].as<path_t>();
        const auto shift = current["shift"].as<float3_t>();
        const auto euler = math::toRad(current["angle"].as<float3_t>());
        float33_t matrix = geometry::euler2matrix(euler, "ZYZ").transpose();
        const geometry::Symmetry symmetry(current["symmetry"].as<std::string>());
        const auto center = current["center"].as<float3_t>();
        const auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        cpu::memory::PtrHost<float> output(elements);

        file.readAll(input.get());
        cpu::geometry::transform3D(input.get(), stride, shape, output.get(), stride, shape,
                                   shift, matrix, symmetry, center, interp, true, stream);
        stream.synchronize();

        if constexpr (COMPUTE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(output.get(), false);
        } else {
            cpu::memory::PtrHost<float> expected(elements);
            file.open(filename_expected, io::READ);
            file.readAll(expected.get());
            if (interp != INTERP_NEAREST) {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected.get(), output.get(), elements, 5e-4));
            } else {
                const float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            }
        }
    }
}
