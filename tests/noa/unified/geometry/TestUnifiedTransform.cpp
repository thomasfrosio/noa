#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/CubicBSplinePrefilter.hpp>
#include <noa/unified/geometry/Transform.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Texture.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Utils.hpp"

using namespace ::noa::types;
using Interp = noa::Interp;
using Border = noa::Border;

TEST_CASE("unified::geometry::transform_2d, vs scipy", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto expected_count = static_cast<i64>(param["tests"].size() * devices.size());
    REQUIRE(expected_count > 1);
    i64 count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<f32>();
        const auto interp = test["interp"].as<Interp>();
        const auto border = test["border"].as<Border>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        const auto center = test["center"].as<Vec2<f64>>();
        const auto scale = test["scale"].as<Vec2<f64>>();
        const auto rotate = noa::deg2rad(test["rotate"].as<f64>());
        const auto shift = test["shift"].as<Vec2<f64>>();
        const auto inv_matrix = noa::inverse(
            noa::geometry::translate(center) *
            noa::geometry::translate(shift) *
            noa::geometry::linear2affine(noa::geometry::rotate(rotate)) *
            noa::geometry::linear2affine(noa::geometry::scale(scale)) *
            noa::geometry::translate(-center)
        ).as<f32>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = true}, options);
            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = true}, options);

            // With arrays:
            const auto output = noa::like(expected);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);

            noa::geometry::transform_2d(input, output, inv_matrix, {
                .interp = interp,
                .border = border,
                .cvalue = cvalue,
            });

            if (interp.is_almost_any(Interp::NEAREST)) {
                // For nearest-neighbor, the border can be off by one pixel,
                // so here just check the total difference.
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(input, device, interp, {
                .border = border,
                .cvalue = cvalue,
                .prefilter = false, // already prefiltered
            });

            noa::fill(output, 0); // erase
            noa::geometry::transform_2d(input_texture, output, inv_matrix);

            if (interp == noa::Interp::NEAREST) {
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_2d(), others", "[noa][unified][assets]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d_cubic"];
    const auto input_filename = path_base / param["input"].as<Path>();

    const auto cvalue = param["cvalue"].as<f32>();
    const auto center = param["center"].as<Vec2<f64>>();
    const auto scale = param["scale"].as<Vec2<f64>>();
    const auto rotate = noa::deg2rad(param["rotate"].as<f64>());
    const auto inv_matrix = noa::geometry::affine2truncated(noa::inverse(
        noa::geometry::translate(center) *
        noa::geometry::linear2affine(noa::geometry::rotate(rotate)) *
        noa::geometry::linear2affine(noa::geometry::scale(scale)) *
        noa::geometry::translate(-center)
    ).as<f32>());

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<noa::Interp>();
        const auto border = test["border"].as<noa::Border>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = true});
            const auto output = noa::like(input);
            if (interp.is_almost_any(noa::Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_2d(input, output, inv_matrix, {interp, border, cvalue});
            noa::io::write(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = true}, options);
            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = true}, options);
            if (interp.is_almost_any(noa::Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(expected);
            noa::geometry::transform_2d(input, output, inv_matrix, {interp, border, cvalue});

            if (interp == noa::Interp::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }

            // With textures:
            const auto input_texture = noa::Texture<f32>(input, device, interp, {
                .border = border,
                .cvalue = cvalue,
                .prefilter= false, // already prefiltered
            });
            noa::fill(output, 0); // erase
            noa::geometry::transform_2d(input_texture, output, inv_matrix);

            if (interp == noa::Interp::NEAREST) {
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform_2d, cpu vs gpu", "[noa][geometry]", f32, f64, c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const Interp interp = GENERATE(
        Interp::LINEAR,
        Interp::CUBIC,
        Interp::CUBIC_BSPLINE,
        Interp::LANCZOS4,
        Interp::LANCZOS6,
        Interp::LANCZOS8
    );
    const Border border = GENERATE(
        Border::ZERO,
        Border::CLAMP,
        Border::VALUE,
        Border::PERIODIC
    );
    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::deg2rad(test::Randomizer<f32>(-360., 360.).get());
    const auto shape = test::random_shape_batched(2);
    const auto center = shape.filter(2, 3).vec.as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto inverse_rotation_matrix =
        noa::geometry::translate(center) *
        noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
        noa::geometry::translate(-center);

    const auto gpu_options = noa::ArrayOption{.device="gpu", .allocator="unified"};
    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::like(input_gpu);

    if (interp.is_almost_any(Interp::CUBIC_BSPLINE)) {
        noa::geometry::cubic_bspline_prefilter(input_cpu, input_cpu);
        noa::geometry::cubic_bspline_prefilter(input_gpu, input_gpu);
    }
    noa::geometry::transform_2d(input_cpu, output_cpu, inverse_rotation_matrix, {interp, border, value});
    noa::geometry::transform_2d(input_gpu, output_gpu, inverse_rotation_matrix, {interp, border, value});

    REQUIRE(test::allclose_abs(output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform_2d(), cpu vs gpu, texture interpolation", "[noa][geometry]", f32, c32) {
    if (not Device::is_any_gpu())
        return;

    const Interp interp = GENERATE(
        Interp::NEAREST_FAST,
        Interp::LINEAR_FAST,
        Interp::CUBIC_FAST,
        Interp::CUBIC_BSPLINE_FAST,
        Interp::LANCZOS4_FAST,
        Interp::LANCZOS6_FAST,
        Interp::LANCZOS8_FAST
    );
    const Border border = GENERATE(
        Border::ZERO,
        Border::CLAMP,
        Border::MIRROR,
        Border::PERIODIC,
        Border::REFLECT
    );
    INFO(interp);
    INFO(border);

    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::deg2rad(test::Randomizer<f32>(-360., 360.).get());
    const auto shape = test::random_shape_batched(2);
    const auto center = shape.filter(2, 3).vec.as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto inverse_rotation_matrix =
        noa::geometry::translate(center) *
        noa::geometry::linear2affine(noa::geometry::rotate(-rotation)) *
        noa::geometry::translate(-center);

    const auto gpu_options = ArrayOption{.device="gpu", .allocator="unified"};
    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = noa::Texture<TestType>(input_cpu, gpu_options.device, interp, {.border=border, .cvalue=value});
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::empty<TestType>(shape, gpu_options);

    noa::geometry::transform_2d(input_cpu, output_cpu, inverse_rotation_matrix, {interp, border, value});
    noa::geometry::transform_2d(input_gpu, output_gpu, inverse_rotation_matrix);

    f32 min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == noa::Interp::CUBIC_BSPLINE_FAST)
        min_max_error = 0.08f; // usually around 0.03-0.06
    REQUIRE(test::allclose_abs(output_cpu, output_gpu, min_max_error));
}

TEST_CASE("unified::geometry::transform_3d, rotate vs scipy", "[noa][unified][assets]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_3d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto expected_count = static_cast<i64>(param["tests"].size() * devices.size());
    REQUIRE(expected_count > 1);
    i64 count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto cvalue = test["cvalue"].as<f32>();
        const auto interp = test["interp"].as<noa::Interp>();
        const auto border = test["border"].as<noa::Border>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        const auto center = test["center"].as<Vec3<f64>>();
        const auto scale = test["scale"].as<Vec3<f64>>();
        const auto euler = noa::deg2rad(test["euler"].as<Vec3<f64>>());
        const auto shift = test["shift"].as<Vec3<f64>>();
        const auto inv_matrix = noa::inverse(
            noa::geometry::translate(center) *
            noa::geometry::translate(shift) *
            noa::geometry::linear2affine(noa::geometry::euler2matrix(euler, {.axes = "zyz"})) *
            noa::geometry::linear2affine(noa::geometry::scale(scale)) *
            noa::geometry::translate(-center)
        ).as<f32>();

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = false}, options);
            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = false}, options);

            // With arrays:
            const auto output = noa::like(expected);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, {interp, border, cvalue});

            if (interp == noa::Interp::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, {.border = border, .cvalue = cvalue, .prefilter = false});
            noa::fill(output, 0); // erase
            noa::geometry::transform_3d(input_texture, output, inv_matrix);

            if (interp == noa::Interp::NEAREST) {
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
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
    const auto euler = noa::deg2rad(param["euler"].as<Vec3<f64>>());
    const auto inv_matrix = noa::geometry::affine2truncated(noa::inverse(
        noa::geometry::translate(center) *
        noa::geometry::linear2affine(noa::geometry::euler2matrix(euler, {.axes = "zyz"})) *
        noa::geometry::linear2affine(noa::geometry::scale(scale)) *
        noa::geometry::translate(-center)
    )).as<f32>();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<noa::Interp>();
        const auto border = test["border"].as<noa::Border>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::io::read_data<f32>(input_filename);
            const auto output = noa::like(input);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, {interp, border, cvalue});
            noa::io::write(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = false}, options);
            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = false}, options);

            // With arrays:
            const auto output = noa::like(expected);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::geometry::cubic_bspline_prefilter(input, input);
            noa::geometry::transform_3d(input, output, inv_matrix, {interp, border, cvalue});

            if (interp == Interp::NEAREST) {
                // For nearest neighbour, the border can be off by one pixel,
                // so here just check the total difference.
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::Texture<f32>(
                    input, device, interp, {.border = border, .cvalue = cvalue, .prefilter = false});
            noa::fill(output, 0); // erase
            noa::geometry::transform_3d(input_texture, output, inv_matrix);

            if (interp == Interp::NEAREST) {
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform_3d, cpu vs gpu", "[noa][geometry]", f32, f64, c32, c64) {
    if (not noa::Device::is_any_gpu())
        return;

    const Interp interp = GENERATE(
        Interp::NEAREST,
        Interp::LINEAR,
        Interp::CUBIC,
        Interp::CUBIC_BSPLINE,
        Interp::LANCZOS4,
        Interp::LANCZOS6,
        Interp::LANCZOS8
    );
    const Border border = GENERATE(
        Border::ZERO,
        Border::CLAMP,
        Border::VALUE,
        Border::PERIODIC,
        Border::REFLECT
    );
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = Vec{
        test::Randomizer<f32>(-360., 360.).get(),
        test::Randomizer<f32>(-360., 360.).get(),
        test::Randomizer<f32>(-360., 360.).get(),
    };
    const auto matrix = noa::geometry::euler2matrix(noa::deg2rad(eulers));

    const auto shape = test::random_shape_batched(3);
    const auto center = shape.pop_front().vec.as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto rotation_matrix =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(matrix) *
            noa::geometry::translate(-center);

    const auto gpu_options = ArrayOption{.device = "gpu", .allocator = "unified"};
    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = input_cpu.to(gpu_options);
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::like(input_gpu);

    if (interp.is_almost_any(Interp::CUBIC_BSPLINE)) {
        noa::geometry::cubic_bspline_prefilter(input_cpu, input_cpu);
        noa::geometry::cubic_bspline_prefilter(input_gpu, input_gpu);
    }
    noa::geometry::transform_3d(input_cpu, output_cpu, rotation_matrix, {interp, border, value});
    noa::geometry::transform_3d(input_gpu, output_gpu, rotation_matrix, {interp, border, value});

    REQUIRE(test::allclose_abs(output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("unified::geometry::transform_3d, cpu vs gpu, texture interpolation", "[noa][geometry]", f32, c32) {
    if (not Device::is_any_gpu())
        return;

    const Interp interp = GENERATE(
        Interp::NEAREST_FAST,
        Interp::LINEAR_FAST,
        Interp::CUBIC_FAST,
        Interp::CUBIC_BSPLINE_FAST,
        Interp::LANCZOS4_FAST,
        Interp::LANCZOS6_FAST,
        Interp::LANCZOS8_FAST
    );
    const Border border = GENERATE(
        Border::ZERO,
        Border::CLAMP,
        Border::MIRROR,
        Border::PERIODIC,
        Border::REFLECT
    );
    INFO(interp);
    INFO(border);

    const TestType value = test::Randomizer<TestType>(-3., 3.).get();
    const auto eulers = Vec{
        test::Randomizer<f32>(-360., 360.).get(),
        test::Randomizer<f32>(-360., 360.).get(),
        test::Randomizer<f32>(-360., 360.).get(),
    };
    const auto matrix = noa::geometry::euler2matrix(noa::deg2rad(eulers));

    const auto shape = test::random_shape_batched(3);
    const auto center = shape.pop_front().vec.as<f32>() / test::Randomizer<f32>(1, 4).get();
    const auto rotation_matrix =
        noa::geometry::translate(center) *
        noa::geometry::linear2affine(matrix) *
        noa::geometry::translate(-center);

    const auto gpu_options = ArrayOption{.device = "gpu", .allocator = "unified"};
    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = noa::Texture<TestType>(input_cpu, gpu_options.device, interp, {.border=border, .cvalue=value});
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::empty<TestType>(shape, gpu_options);

    noa::geometry::transform_3d(input_cpu, output_cpu, rotation_matrix, {interp, border, value});
    noa::geometry::transform_3d(input_gpu, output_gpu, rotation_matrix);

    f32 min_max_error = 0.05f; // for linear and cosine, it is usually around 0.01-0.03
    if (interp == noa::Interp::CUBIC_BSPLINE_FAST)
        min_max_error = 0.2f; // usually around 0.09
    REQUIRE(test::allclose_abs(output_cpu, output_gpu, min_max_error));
}
