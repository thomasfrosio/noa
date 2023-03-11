#include <noa/core/Math.hpp>
#include <noa/unified/geometry/Prefilter.hpp>
#include <noa/unified/geometry/Transform.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::geometry::transform_2d, symmetry", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d_symmetry"];
    const auto input_filename = path_base / param["input"].as<Path>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::memory::empty<float>({1, 1, 512, 512});
        const auto center = Vec2<f32>{256, 256};
        noa::geometry::rectangle({}, asset, center, {64, 128}, 5);
        noa::geometry::rectangle(asset, asset, center + Vec2<f32>{64, 128}, {32, 32}, 3, {}, noa::plus_t{}, 1.f);
        noa::io::save(asset, input_filename);
    }

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto shift = test["shift"].as<Vec2<f32>>();
        const auto symmetry = noa::geometry::Symmetry(test["symmetry"].as<std::string>());
        const auto angle = math::deg2rad(test["angle"].as<float>());
        const auto center = test["center"].as<Vec2<f32>>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto inv_matrix = noa::geometry::rotate(-angle);

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(input);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_and_symmetrize_2d(input, output, shift, inv_matrix, symmetry, center, interp);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::load_data<f32>(expected_filename, true, options);
            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::get_difference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, BorderMode::ZERO, 0.f, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_and_symmetrize_2d(input_texture, output, shift, inv_matrix, symmetry, center);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const float diff = test::get_difference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_3d, symmetry", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_3d_symmetry"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::memory::empty<float>({1, 150, 150, 150});
        const auto rectangle_center = Vec3<f32>(150 / 2);
        noa::geometry::rectangle({}, asset, rectangle_center, {34, 24, 24}, 3);
        noa::io::save(asset, path_base / param["input"][0].as<Path>());

        noa::geometry::rectangle(asset, asset, rectangle_center + Vec3 < f32 > {50, 34, 34},
                                 {15, 15, 15}, 3, {}, noa::plus_t{}, 1.f);
        noa::io::save(asset, path_base / param["input"][1].as<Path>());
    }

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto shift = test["shift"].as<Vec3<f32>>();
        const auto symmetry = noa::geometry::Symmetry(test["symmetry"].as<std::string>());
        const auto angles = math::deg2rad(test["angles"].as<Vec3<f64>>());
        const auto center = test["center"].as<Vec3<f32>>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto inv_matrix = static_cast<Float33>(noa::geometry::euler2matrix(angles).transpose());

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(input);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_and_symmetrize_3d(input, output, shift, inv_matrix, symmetry, center, interp);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::load_data<f32>(expected_filename, false, options);
            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::get_difference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, BorderMode::ZERO, 0.f, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_and_symmetrize_3d(input_texture, output, shift, inv_matrix, symmetry, center);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const float diff = test::get_difference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEMPLATE_TEST_CASE("unified::geometry::symmetry_2d", "[noa][unified]", f32, f64) {
    const char* symbol = GENERATE("  c1", "C2", " C7", "d1", "D3 ");
    const auto symmetry = geometry::Symmetry(symbol);
    const auto shape = test::get_random_shape4_batched(2);
    const auto center = shape.filter(2, 3).vec().as<f32>() / 2;

    const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -10, 10);
    const auto expected = noa::memory::like(input);
    noa::geometry::transform_and_symmetrize_2d(input, expected, {}, {}, symmetry, center);
    expected.eval();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        const auto input_copy = input.to(options);
        const auto output = noa::memory::like(input_copy);

        // With arrays:
        noa::geometry::symmetrize_2d(input_copy, output, symmetry, center);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));

        // With textures:
        if constexpr (std::is_same_v<TestType, float>) {
            const auto input_texture = noa::Texture<TestType>(
                    input_copy, device, InterpMode::LINEAR, BorderMode::ZERO, 0, /*layered=*/ true);
            noa::memory::fill(output, TestType{0}); // erase
            noa::geometry::symmetrize_2d(input_texture, output, symmetry, center);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::symmetry3D", "[noa][unified]", float, double) {
    const char* symbol = GENERATE("c1", "  C2", "C7 ", " D1", "D3", " o", "i1", "I2  ");
    const auto symmetry = geometry::Symmetry(symbol);
    const auto shape = test::get_random_shape4(3);
    const auto center = shape.pop_front().vec().as<f32>() / 2;

    const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -10, 10);
    const auto expected = noa::memory::like(input);
    noa::geometry::transform_and_symmetrize_3d(input, expected, {}, {}, symmetry, center);
    expected.eval();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        const auto input_copy = input.to(options);
        const auto output = noa::memory::like(input_copy);

        // With arrays:
        noa::geometry::symmetrize_3d(input_copy, output, symmetry, center);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));

        // With textures:
        if constexpr (std::is_same_v<TestType, float>) {
            const auto input_texture = noa::Texture<TestType>(input_copy, device, InterpMode::LINEAR, BorderMode::ZERO);
            noa::memory::fill(output, TestType{0}); // erase
            noa::geometry::symmetrize_3d(input_texture, output, symmetry, center);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-4));
        }
    }
}
