#include <noa/Geometry.h>
#include <noa/Math.h>
#include <noa/IO.h>
#include <noa/Memory.h>
#include <noa/Signal.h>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::geometry::transform2D, symmetry", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D_symmetry"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::memory::empty<float>({1, 1, 512, 512});
        const auto center = float2_t{256, 256};
        noa::signal::rectangle({}, asset, center, {64, 128}, 5);
        noa::signal::rectangle(asset, asset, center + float2_t{64, 128}, {32, 32}, 3, {}, noa::math::plus_t{}, 1.f);
        noa::io::save(asset, input_filename);
    }

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto shift = test["shift"].as<float2_t>();
        const auto symmetry = noa::geometry::Symmetry(test["symmetry"].as<std::string>());
        const auto angle = math::deg2rad(test["angle"].as<float>());
        const auto center = test["center"].as<float2_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto inv_matrix = noa::geometry::rotate(-angle);

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(input);
            noa::geometry::transform2D(input, output, shift, inv_matrix, symmetry, center, interp);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::load<float>(expected_filename, true, options);
            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, BorderMode::BORDER_ZERO, 0.f, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform2D(input_texture, output, shift, inv_matrix, symmetry, center);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform3D, symmetry", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D_symmetry"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::memory::empty<float>({1, 150, 150, 150});
        const auto rectangle_center = float3_t(150 / 2);
        noa::signal::rectangle({}, asset, rectangle_center, {34, 24, 24}, 3);
        noa::io::save(asset, path_base / param["input"][0].as<path_t>());

        noa::signal::rectangle(asset, asset, rectangle_center + float3_t{50, 34, 34},
                               {15, 15, 15}, 3, {}, noa::math::plus_t{}, 1.f);
        noa::io::save(asset, path_base / param["input"][1].as<path_t>());
    }

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<path_t>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto shift = test["shift"].as<float3_t>();
        const auto symmetry = noa::geometry::Symmetry(test["symmetry"].as<std::string>());
        const auto angles = math::deg2rad(test["angles"].as<double3_t>());
        const auto center = test["center"].as<float3_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto inv_matrix = static_cast<float33_t>(noa::geometry::euler2matrix(angles).transpose());

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(input);
            noa::geometry::transform3D(input, output, shift, inv_matrix, symmetry, center, interp);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::load<float>(expected_filename, false, options);
            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, BorderMode::BORDER_ZERO, 0.f, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform3D(input_texture, output, shift, inv_matrix, symmetry, center);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEMPLATE_TEST_CASE("unified::geometry::symmetry2D", "[noa][unified]", float, double) {
    const char* symbol = GENERATE("  c1", "C2", " C7", "d1", "D3 ");
    const auto symmetry = geometry::Symmetry(symbol);
    const auto shape = test::getRandomShapeBatched(2);
    const auto center = float2_t(shape.get(2)) / 2;

    const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -10, 10);
    const auto expected = noa::memory::like(input);
    noa::geometry::transform2D(input, expected, {}, {}, symmetry, center);
    expected.eval();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        const auto input_copy = input.to(options);
        const auto output = noa::memory::like(input_copy);

        // With arrays:
        noa::geometry::symmetrize2D(input_copy, output, symmetry, center);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));

        // With textures:
        if constexpr (std::is_same_v<TestType, float>) {
            const auto input_texture = noa::Texture<TestType>(
                    input_copy, device, INTERP_LINEAR, BORDER_ZERO, 0, /*layered=*/ true);
            noa::memory::fill(output, TestType{0}); // erase
            noa::geometry::symmetrize2D(input_texture, output, symmetry, center);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::symmetry3D", "[noa][unified]", float, double) {
    const char* symbol = GENERATE("c1", "  C2", "C7 ", " D1", "D3", " o", "i1", "I2  ");
    const auto symmetry = geometry::Symmetry(symbol);
    const auto shape = test::getRandomShape(3);
    const auto center = float3_t(shape.get(1)) / 2;

    const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -10, 10);
    const auto expected = noa::memory::like(input);
    noa::geometry::transform3D(input, expected, {}, {}, symmetry, center);
    expected.eval();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        const auto input_copy = input.to(options);
        const auto output = noa::memory::like(input_copy);

        // With arrays:
        noa::geometry::symmetrize3D(input_copy, output, symmetry, center);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));

        // With textures:
        if constexpr (std::is_same_v<TestType, float>) {
            const auto input_texture = noa::Texture<TestType>(input_copy, device, INTERP_LINEAR, BORDER_ZERO);
            noa::memory::fill(output, TestType{0}); // erase
            noa::geometry::symmetrize3D(input_texture, output, symmetry, center);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
        }
    }
}
