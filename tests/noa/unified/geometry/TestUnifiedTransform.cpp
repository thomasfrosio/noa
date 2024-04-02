#include <noa/core/Math.hpp>
#include <noa/unified/geometry/CubicBSplinePrefilter.hpp>
#include <noa/unified/geometry/Transform.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::geometry::transform_2d, vs scipy", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto expected_count = static_cast<i64>(param["tests"].size() * devices.size());
    REQUIRE(expected_count > 1);
    i64 count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<f32>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        const auto center = test["center"].as<Vec2<f64>>();
        const auto scale = test["scale"].as<Vec2<f64>>();
        const auto rotate = math::deg2rad(test["rotate"].as<f64>());
        const auto shift = test["shift"].as<Vec2<f64>>();
        const auto inv_matrix = noa::math::inverse(
                noa::geometry::translate(center) *
                noa::geometry::translate(shift) *
                noa::geometry::linear2affine(noa::geometry::rotate(rotate)) *
                noa::geometry::linear2affine(noa::geometry::scale(scale)) *
                noa::geometry::translate(-center)).as<f32>();

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, true, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_2d(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            if (device.is_gpu() &&
                ((border == BorderMode::VALUE || border == BorderMode::REFLECT) ||
                 ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) &&
                  interp != InterpMode::LINEAR_FAST && interp != InterpMode::NEAREST)))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_2d(input_texture, output, inv_matrix);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_2d(), cubic", "[noa][unified][assets]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d_cubic"];
    const auto input_filename = path_base / param["input"].as<Path>();

    const auto cvalue = param["cvalue"].as<f32>();
    const auto center = param["center"].as<Vec2<f64>>();
    const auto scale = param["scale"].as<Vec2<f64>>();
    const auto rotate = math::deg2rad(param["rotate"].as<f64>());
    const auto inv_matrix = noa::geometry::affine2truncated(noa::math::inverse(
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(rotate)) *
            noa::geometry::linear2affine(noa::geometry::scale(scale)) *
            noa::geometry::translate(-center)
    ).as<f32>());

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::load_data<f32>(input_filename, true);
            const auto output = noa::memory::like(input);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_2d(input, output, inv_matrix, interp, border, cvalue);
            noa::io::save(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, true, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, true, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_2d(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }

            // With textures:
            if (device.is_gpu() &&
                ((border == BorderMode::VALUE || border == BorderMode::REFLECT) ||
                 ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) &&
                  interp != InterpMode::LINEAR_FAST && interp != InterpMode::NEAREST)))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_2d(input_texture, output, inv_matrix);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform_2d, cpu vs gpu", "[noa][geometry]", f32, f64, c32, c64) {
    if (!noa::Device::is_any(noa::DeviceType::GPU))
        return;

    const InterpMode interp = GENERATE(InterpMode::LINEAR, InterpMode::COSINE, InterpMode::CUBIC, InterpMode::CUBIC_BSPLINE);
    const BorderMode border = GENERATE(BorderMode::ZERO, BorderMode::CLAMP, BorderMode::VALUE, BorderMode::PERIODIC);
    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::math::deg2rad(test::Randomizer<f32>(-360., 360.).get());
    const auto shape = test::get_random_shape4_batched(2);
    const auto center = shape.filter(2, 3).vec().as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(noa::Device("gpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST) {
        noa::geometry::cubic_bspline_prefilter(input_cpu, input_cpu);
        noa::geometry::cubic_bspline_prefilter(input_gpu, input_gpu);
    }
    noa::geometry::transform_2d(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform_2d(input_gpu, output_gpu, rotation_matrix, interp, border, value);

    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform_2d(), cpu vs gpu, texture interpolation", "[noa][geometry]",
                   f32, c32) {
    if (!noa::Device::is_any(noa::DeviceType::GPU))
        return;

    const InterpMode interp = GENERATE(InterpMode::LINEAR_FAST, InterpMode::COSINE_FAST, InterpMode::CUBIC_BSPLINE_FAST);
    const BorderMode border = GENERATE(BorderMode::ZERO, BorderMode::CLAMP, BorderMode::MIRROR, BorderMode::PERIODIC);
    if ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) && interp != InterpMode::LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::math::deg2rad(test::Randomizer<f32>(-360., 360.).get());
    const auto shape = test::get_random_shape4_batched(2);
    const auto center = shape.filter(2, 3).vec().as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
            noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("gpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = noa::Texture(input_cpu, gpu_options.device(), interp, border, value, /*layered=*/ true);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::empty<TestType>(shape, gpu_options);

    noa::geometry::transform_2d(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform_2d(input_gpu, output_gpu, rotation_matrix);

    f32 min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == InterpMode::CUBIC_BSPLINE_FAST)
        min_max_error = 0.08f; // usually around 0.03-0.06
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, min_max_error));
}

TEST_CASE("unified::geometry::transform_3d, rotate vs scipy", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_3d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto expected_count = static_cast<i64>(param["tests"].size() * devices.size());
    REQUIRE(expected_count > 1);
    i64 count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<f32>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        const auto center = test["center"].as<Vec3<f64>>();
        const auto scale = test["scale"].as<Vec3<f64>>();
        const auto euler = math::deg2rad(test["euler"].as<Vec3<f64>>());
        const auto shift = test["shift"].as<Vec3<f64>>();
        const auto inv_matrix = noa::math::inverse(
                noa::geometry::translate(center) *
                noa::geometry::translate(shift) *
                noa::geometry::linear2affine(noa::geometry::euler2matrix(euler)) *
                noa::geometry::linear2affine(noa::geometry::scale(scale)) *
                noa::geometry::translate(-center)
        ).as<f32>();

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, false, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            if (device.is_gpu() &&
                ((border == BorderMode::VALUE || border == BorderMode::REFLECT) ||
                 ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) &&
                  interp != InterpMode::LINEAR_FAST && interp != InterpMode::NEAREST)))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_3d(input_texture, output, inv_matrix);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_3d(), cubic", "[noa][unified][assets]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_3d_cubic"];
    const auto input_filename = path_base / param["input"].as<Path>();

    const auto cvalue = param["cvalue"].as<f32>();
    const auto center = param["center"].as<Vec3<f64>>();
    const auto scale = param["scale"].as<Vec3<f64>>();
    const auto euler = noa::math::deg2rad(param["euler"].as<Vec3<f64>>());
    const auto inv_matrix = noa::geometry::affine2truncated(noa::math::inverse(
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::euler2matrix(euler)) *
            noa::geometry::linear2affine(noa::geometry::scale(scale)) *
            noa::geometry::translate(-center)
    )).as<f32>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::load_data<f32>(input_filename);
            const auto output = noa::memory::like(input);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, interp, border, cvalue);
            noa::io::save(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, false, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, false, options);

            // With arrays:
            const auto output = noa::memory::like(expected);
            if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST)
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, interp, border, cvalue);

            if (interp == noa::InterpMode::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }

            // With textures:
            if (device.is_gpu() &&
                ((border == BorderMode::VALUE || border == BorderMode::REFLECT) ||
                 ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) &&
                  interp != InterpMode::LINEAR_FAST && interp != InterpMode::NEAREST)))
                continue; // gpu textures have limitations on the addressing mode

            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, border, cvalue, /*layered=*/ false, /*prefilter=*/ false);
            noa::memory::fill(output, 0.f); // erase
            noa::geometry::transform_3d(input_texture, output, inv_matrix);

            if (interp == noa::InterpMode::NEAREST) {
                output.eval();
                const f32 diff = test::get_difference(expected.get(), output.get(), expected.ssize());
                REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform_3d, cpu vs gpu", "[noa][geometry]", f32, f64, c32, c64) {
    if (!noa::Device::is_any(noa::DeviceType::GPU))
        return;

    const InterpMode interp = GENERATE(InterpMode::LINEAR, InterpMode::COSINE, InterpMode::CUBIC, InterpMode::CUBIC_BSPLINE);
    const BorderMode border = GENERATE(BorderMode::ZERO, BorderMode::CLAMP, BorderMode::VALUE, BorderMode::PERIODIC);
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = Vec3<f32>{test::Randomizer<f32>(-360., 360.).get(),
                                  test::Randomizer<f32>(-360., 360.).get(),
                                  test::Randomizer<f32>(-360., 360.).get()};
    const Float33 matrix = geometry::euler2matrix(noa::math::deg2rad(eulers));

    const auto shape = test::get_random_shape4_batched(3);
    const auto center = shape.pop_front().vec().as<f32>() / test::Randomizer<f32>(1, 4).get();
    const Float44 rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(matrix) *
            noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("gpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    if (interp == InterpMode::CUBIC_BSPLINE || interp == InterpMode::CUBIC_BSPLINE_FAST) {
        noa::geometry::cubic_bspline_prefilter(input_cpu, input_cpu);
        noa::geometry::cubic_bspline_prefilter(input_gpu, input_gpu);
    }
    noa::geometry::transform_3d(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform_3d(input_gpu, output_gpu, rotation_matrix, interp, border, value);

    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform_3d, cpu vs gpu, texture interpolation", "[noa][geometry]",
                   f32, c32) {
    if (!noa::Device::is_any(noa::DeviceType::GPU))
        return;

    const InterpMode interp = GENERATE(InterpMode::LINEAR_FAST, InterpMode::COSINE_FAST, InterpMode::CUBIC_BSPLINE_FAST);
    const BorderMode border = GENERATE(BorderMode::ZERO, BorderMode::CLAMP, BorderMode::MIRROR, BorderMode::PERIODIC);
    if ((border == BorderMode::MIRROR || border == BorderMode::PERIODIC) && interp != InterpMode::LINEAR_FAST)
        return;

    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = Vec3<f32>{test::Randomizer<f32>(-360., 360.).get(),
                                  test::Randomizer<f32>(-360., 360.).get(),
                                  test::Randomizer<f32>(-360., 360.).get()};
    const Float33 matrix = geometry::euler2matrix(noa::math::deg2rad(eulers));

    const auto shape = test::get_random_shape4(3);
    const auto center = shape.pop_front().vec().as<f32>() / test::Randomizer<f32>(1, 4).get();
    const Float44 rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(matrix) *
            noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption(Device("gpu"), noa::Allocator::MANAGED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -2, 2);
    const auto input_gpu = noa::Texture(input_cpu, gpu_options.device(), interp, border, value);
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::empty<TestType>(shape, gpu_options);

    noa::geometry::transform_3d(input_cpu, output_cpu, rotation_matrix, interp, border, value);
    noa::geometry::transform_3d(input_gpu, output_gpu, rotation_matrix);

    f32 min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == InterpMode::CUBIC_BSPLINE_FAST)
        min_max_error = 0.2f; // usually around 0.09
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu, min_max_error));
}
