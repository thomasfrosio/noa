#include <noa/common/io/ImageFile.h>

#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Shape.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::apply2D() - symmetry", "[assets][noa][cpu][transform]") {
    path_t base_path = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(base_path / "param.yaml")["apply2D_symmetry"];
    path_t input_path = base_path / param["input"].as<path_t>();
    io::ImageFile file;

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        size3_t shape{512, 512, 1};
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);

        cpu::filter::rectangle2D<false, float>(
                nullptr, input.get(), {shape.x, shape.y}, 1, {shape.x / 2, shape.y / 2}, {128, 64}, 5);
        cpu::memory::PtrHost<float> tmp(elements);
        cpu::filter::rectangle2D<false, float>(
                nullptr, tmp.get(), {shape.x, shape.y}, 1, {shape.x / 2 + 128, shape.y / 2 + 64}, {32, 32}, 3);
        cpu::math::addArray(input.get(), tmp.get(), input.get(), elements, 1);

        file.open(input_path, io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);
    }

    for (size_t i = 0; i < param["tests"].size(); ++i) {
        INFO("test number = " << i);

        // Parameters:
        auto current = param["tests"][i];
        auto filename_expected = base_path / current["expected"].as<path_t>();
        auto shift = current["shift"].as<float2_t>();
        float22_t matrix = transform::rotate(-math::toRad(current["angle"].as<float>())); // inverse
        transform::Symmetry symmetry(current["symmetry"].as<std::string>());
        auto center = current["center"].as<float2_t>();
        auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(input_path, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        cpu::memory::PtrHost<float> output(elements);

        file.readAll(input.get());
        cpu::transform::apply2D(input.get(), output.get(), {shape.x, shape.y}, shift, matrix, symmetry, center, interp);

        if constexpr (COMPUTE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(output.get(), false);
        } else {
            cpu::memory::PtrHost<float> expected(elements);
            file.open(filename_expected, io::READ);
            file.readAll(expected.get());
            if (interp != INTERP_NEAREST) {
                cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
                float min, max, mean;
                cpu::math::minMaxMean(output.get(), &min, &max, &mean, elements, 1);
                REQUIRE(math::abs(min) < 5e-4f);
                REQUIRE(math::abs(max) < 5e-4f);
                REQUIRE(math::abs(mean) < 1e-6f);
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
            }
        }
    }
}

TEST_CASE("cpu::transform::apply3D() - symmetry", "[assets][noa][cpu][transform]") {
    path_t base_path = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(base_path / "param.yaml")["apply3D_symmetry"];
    io::ImageFile file;

    constexpr bool COMPUTE_ASSETS = true;
    if constexpr (COMPUTE_ASSETS) {
        size3_t shape{150, 150, 150};
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);

        float3_t rectangle_center(shape / size_t{2});
        cpu::filter::rectangle3D<false, float>(nullptr, input.get(), shape, 1, rectangle_center, {24, 24, 34}, 3);
        file.open(base_path / param["input"][0].as<path_t>(), io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);

        cpu::memory::PtrHost<float> tmp(elements);
        cpu::filter::rectangle3D<false, float>(
                nullptr, tmp.get(), shape, 1, rectangle_center + float3_t{34, 34, 50}, {15, 15, 15}, 3);
        cpu::math::addArray(input.get(), tmp.get(), input.get(), elements, 1);
        file.open(base_path / param["input"][1].as<path_t>(), io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);
    }

    for (size_t i = 0; i < param["tests"].size(); ++i) {
        INFO("test number = " << i);

        // Parameters:
        auto current = param["tests"][i];
        auto filename_expected = base_path / current["expected"].as<path_t>();
        auto filename_input = base_path / current["input"].as<path_t>();
        auto shift = current["shift"].as<float3_t>();
        float33_t matrix = transform::toMatrix<true>(math::toRad(current["angle"].as<float3_t>()));
        transform::Symmetry symmetry(current["symmetry"].as<std::string>());
        auto center = current["center"].as<float3_t>();
        auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = noa::elements(shape);
        cpu::memory::PtrHost<float> input(elements);
        cpu::memory::PtrHost<float> output(elements);

        file.readAll(input.get());
        cpu::transform::apply3D(input.get(), output.get(), shape, shift, matrix, symmetry, center, interp);

        if constexpr (COMPUTE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(output.get(), false);
        } else {
            cpu::memory::PtrHost<float> expected(elements);
            file.open(filename_expected, io::READ);
            file.readAll(expected.get());
            if (interp != INTERP_NEAREST) {
                cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
                float min, max, mean;
                cpu::math::minMaxMean(output.get(), &min, &max, &mean, elements, 1);
                REQUIRE(math::abs(min) < 5e-4f);
                REQUIRE(math::abs(max) < 5e-4f);
                REQUIRE(math::abs(mean) < 1e-6f);
            } else {
                float diff = test::getDifference(expected.get(), output.get(), elements);
                REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
            }
        }
    }
}
