#include <noa/Geometry.h>
#include <noa/Math.h>
#include <noa/IO.h>
#include <noa/Memory.h>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::geometry::transform2D, vs scipy", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();

        const auto center = test["center"].as<double2_t>();
        const auto scale = test["scale"].as<double2_t>();
        const auto rotate = math::deg2rad(test["rotate"].as<double>());
        const auto shift = test["shift"].as<double2_t>();
        const auto inv_matrix = static_cast<float33_t>(
                noa::math::inverse(
                        noa::geometry::translate(center) *
                        noa::geometry::translate(shift) *
                        double33_t(noa::geometry::rotate(rotate)) *
                        double33_t(noa::geometry::scale(scale)) *
                        noa::geometry::translate(-center)));

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, true, options);
            const auto expected = noa::io::load<float>(expected_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            noa::geometry::transform2D(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            if (device.gpu() &&
                (border == BORDER_VALUE || border == BORDER_REFLECT) ||
                ((border == BORDER_MIRROR || border == BORDER_PERIODIC) &&
                 interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform2D(input_texture, output, inv_matrix);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform2D(), cubic", "[noa][unified][assets]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D_cubic"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    const auto cvalue = param["cvalue"].as<float>();
    const auto center = param["center"].as<double2_t>();
    const auto scale = param["scale"].as<double2_t>();
    const auto rotate = math::deg2rad(param["rotate"].as<double>());
    const auto inv_matrix = static_cast<float23_t>(
            noa::math::inverse(
                    noa::geometry::translate(center) *
                    double33_t(noa::geometry::rotate(rotate)) *
                    double33_t(noa::geometry::scale(scale)) *
                    noa::geometry::translate(-center)));

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::load<float>(input_filename, true);
            const auto output = noa::memory::like(input);
            noa::geometry::transform2D(input, output, inv_matrix, interp, border, cvalue);
            noa::io::save(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, true, options);
            const auto expected = noa::io::load<float>(expected_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            noa::geometry::transform2D(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }

            // With textures:
            if (device.gpu() &&
                (border == BORDER_VALUE || border == BORDER_REFLECT) ||
                ((border == BORDER_MIRROR || border == BORDER_PERIODIC) &&
                 interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform2D(input_texture, output, inv_matrix);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform2D, cpu vs gpu", "[noa][geometry]",
                   float, double, cfloat_t, cdouble_t) {
    if (!noa::Device::any(noa::Device::GPU))
        return;

    const InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_VALUE, BORDER_PERIODIC);
    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::math::deg2rad(test::Randomizer<float>(-360., 360.).get());
    const auto shape = test::getRandomShapeBatched(2u);
    const auto center = float2_t(shape.get(2)) / test::Randomizer<float>(1, 4).get();
    const auto rotation_matrix =
            noa::geometry::translate(center) *
            float33_t(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(noa::Device("cpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    noa::geometry::transform2D(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform2D(input_gpu, output_gpu, rotation_matrix, interp, border, value);

    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform2D(), cpu vs gpu, texture interpolation", "[noa][geometry]",
                   float, cfloat_t) {
    if (!noa::Device::any(noa::Device::GPU))
        return;

    const InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if ((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::math::deg2rad(test::Randomizer<float>(-360., 360.).get());
    const auto shape = test::getRandomShapeBatched(2u);
    const auto center = float2_t(shape.get(2)) / test::Randomizer<float>(1, 4).get();
    const auto rotation_matrix =
            geometry::translate(center) *
            float33_t(geometry::rotate(-rotation)) *
            geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("cpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = noa::Texture(input_cpu, gpu_options.device(), interp, border, value);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::empty<TestType>(shape, gpu_options);

    noa::geometry::transform2D(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform2D(input_gpu, output_gpu, rotation_matrix);

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.08f; // usually around 0.03-0.06
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, min_max_error));
}

TEST_CASE("unified::geometry::transform3D, rotate vs scipy", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();

        const auto center = test["center"].as<double3_t>();
        const auto scale = test["scale"].as<double3_t>();
        const auto euler = math::deg2rad(test["euler"].as<double3_t>());
        const auto shift = test["shift"].as<double3_t>();
        const auto inv_matrix = static_cast<float44_t>(
                noa::math::inverse(
                        noa::geometry::translate(center) *
                        noa::geometry::translate(shift) *
                        double44_t(noa::geometry::euler2matrix(euler)) *
                        double44_t(noa::geometry::scale(scale)) *
                        noa::geometry::translate(-center)));

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, false, options);
            const auto expected = noa::io::load<float>(expected_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            noa::geometry::transform3D(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            if (device.gpu() &&
                (border == BORDER_VALUE || border == BORDER_REFLECT) ||
                ((border == BORDER_MIRROR || border == BORDER_PERIODIC) &&
                 interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform3D(input_texture, output, inv_matrix);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform3D(), cubic", "[noa][unified][assets]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D_cubic"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    const auto cvalue = param["cvalue"].as<float>();
    const auto center = param["center"].as<double3_t>();
    const auto scale = param["scale"].as<double3_t>();
    const auto euler = math::deg2rad(param["euler"].as<double3_t>());
    const auto inv_matrix = static_cast<float44_t>(
            noa::math::inverse(
                    noa::geometry::translate(center) *
                    double44_t(noa::geometry::euler2matrix(euler)) *
                    double44_t(noa::geometry::scale(scale)) *
                    noa::geometry::translate(-center)));

    std::vector<Device> devices{Device("gpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::load<float>(input_filename);
            const auto output = noa::memory::like(input);
            noa::geometry::transform3D(input, output, inv_matrix, interp, border, cvalue);
            noa::io::save(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, false, options);
            const auto expected = noa::io::load<float>(expected_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            noa::geometry::transform3D(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::INTERP_NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }

            // With textures:
            if (device.gpu() &&
                (border == BORDER_VALUE || border == BORDER_REFLECT) ||
                ((border == BORDER_MIRROR || border == BORDER_PERIODIC) &&
                 interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<float>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform3D(input_texture, output, inv_matrix);

            if (interp == noa::INTERP_NEAREST) {
                output.eval();
                const float diff = test::getDifference(expected.get(), output.get(), expected.size());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform3D, cpu vs gpu", "[noa][geometry]",
                   float, double, cfloat_t, cdouble_t) {
    if (!noa::Device::any(noa::Device::GPU))
        return;

    const InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_VALUE, BORDER_PERIODIC);
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = float3_t{test::Randomizer<float>(-360., 360.).get(),
                                 test::Randomizer<float>(-360., 360.).get(),
                                 test::Randomizer<float>(-360., 360.).get()};
    const float33_t matrix = geometry::euler2matrix(noa::math::deg2rad(eulers));

    const auto shape = test::getRandomShapeBatched(3u);
    const auto center = float3_t(shape.get(1)) / test::Randomizer<float>(1, 4).get();
    const float44_t rotation_matrix =
            geometry::translate(center) *
            float44_t(matrix) *
            geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("cpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    noa::geometry::transform3D(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform3D(input_gpu, output_gpu, rotation_matrix, interp, border, value);

    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform3D, cpu vs gpu, texture interpolation", "[noa][geometry]",
                   float, cfloat_t) {
    if (!noa::Device::any(noa::Device::GPU))
        return;

    const InterpMode interp = GENERATE(INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    const BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC);
    if ((border == BORDER_MIRROR || border == BORDER_PERIODIC) && interp != INTERP_LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = float3_t{test::Randomizer<float>(-360., 360.).get(),
                                 test::Randomizer<float>(-360., 360.).get(),
                                 test::Randomizer<float>(-360., 360.).get()};
    const float33_t matrix = geometry::euler2matrix(noa::math::deg2rad(eulers));

    const auto shape = test::getRandomShapeBatched(3u);
    const auto center = float3_t(shape.get(1)) / test::Randomizer<float>(1, 4).get();
    const float44_t rotation_matrix =
            geometry::translate(center) *
            float44_t(matrix) *
            geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("cpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = noa::Texture(input_cpu, gpu_options.device(), interp, border, value);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::empty<TestType>(shape, gpu_options);

    noa::geometry::transform3D(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform3D(input_gpu, output_gpu, rotation_matrix);

    float min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == INTERP_CUBIC_BSPLINE_FAST)
        min_max_error = 0.2f; // usually around 0.09
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, min_max_error));
}
