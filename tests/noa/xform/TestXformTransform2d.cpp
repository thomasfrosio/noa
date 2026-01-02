#include <noa/xform/CubicBSplinePrefilter.hpp>
#include <noa/xform/Transform.hpp>
#include <noa/xform/Texture.hpp>
#include <noa/IO.hpp>
#include <noa/Runtime.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
using Interp = noa::xform::Interp;
using Border = noa::Border;

TEST_CASE("xform::transform_2d, vs scipy", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "xform";
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

        const auto center = test["center"].as<Vec<f64, 2>>();
        const auto scale = test["scale"].as<Vec<f64, 2>>();
        const auto rotate = noa::deg2rad(test["rotate"].as<f64>());
        const auto shift = test["shift"].as<Vec<f64, 2>>();
        const auto inv_matrix = noa::inverse(
            noa::xform::translate(center) *
            noa::xform::translate(shift) *
            noa::xform::affine(noa::xform::rotate(rotate)) *
            noa::xform::affine(noa::xform::scale(scale)) *
            noa::xform::translate(-center)
        ).as<f32>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = true}, options).data;
            const auto expected = noa::read_image<f32>(expected_filename, {.enforce_2d_stack = true}, options).data;

            // With arrays:
            const auto output = noa::like(expected);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::xform::cubic_bspline_prefilter(input, input);

            noa::xform::transform_2d(input, output, inv_matrix, {
                .interp = interp,
                .border = border,
                .cvalue = cvalue,
            });

            if (interp.is_almost_any(Interp::NEAREST)) {
                // For nearest-neighbor, the border can be off by one pixel,
                // so here just check the total difference.
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::Matchers::WithinAbs(0, 1e-6));
            } else {
                // Otherwise it is usually around 2e-5, but there are some outliers...
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
            ++count;

            // With textures:
            // The input is prefiltered at this point, so no need for prefiltering here.
            const auto input_texture = noa::xform::Texture<f32>(input, device, interp, {
                .border = border,
                .cvalue = cvalue,
                .prefilter = false, // already prefiltered
            });

            noa::fill(output, 0); // erase
            noa::xform::transform_2d(input_texture, output, inv_matrix);

            if (interp == noa::xform::Interp::NEAREST) {
                const test::MatchResult results = test::allclose_abs(expected, output, 1e-4f);
                REQUIRE_THAT(results.total_abs_diff, Catch::Matchers::WithinAbs(0, 1e-6));
            } else {
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
            }
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("xform::transform_2d(), others", "[asset]") {
    constexpr bool GENERATE_TEST_DATA = false;
    const Path path_base = test::NOA_DATA_PATH / "xform";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d_more"];
    const auto input_filename = path_base / param["input"].as<Path>();

    const auto cvalue = param["cvalue"].as<f32>();
    const auto center = param["center"].as<Vec<f64, 2>>();
    const auto scale = param["scale"].as<Vec<f64, 2>>();
    const auto rotate = noa::deg2rad(param["rotate"].as<f64>());
    const auto inv_matrix = (
        noa::xform::translate(center) *
        noa::xform::affine(noa::xform::rotate(rotate)) *
        noa::xform::affine(noa::xform::scale(scale)) *
        noa::xform::translate(-center)
    ).inverse().pop_back().as<f32>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto interp = test["interp"].as<noa::xform::Interp>();
        const auto border = test["border"].as<noa::Border>();
        const auto expected_filename = path_base / test["expected"].as<Path>();

        if constexpr (GENERATE_TEST_DATA) {
            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = true}).data;
            const auto output = noa::like(input);
            if (interp.is_almost_any(noa::xform::Interp::CUBIC_BSPLINE))
                noa::xform::cubic_bspline_prefilter(input, input);
            noa::xform::transform_2d(input, output, inv_matrix, {interp, border, cvalue});
            noa::write_image(output, expected_filename);
            continue;
        }

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = true}, options).data;
            const auto expected = noa::read_image<f32>(expected_filename, {.enforce_2d_stack = true}, options).data;
            if (interp.is_almost_any(noa::xform::Interp::CUBIC_BSPLINE))
                noa::xform::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(expected);
            noa::xform::transform_2d(input, output, inv_matrix, {interp, border, cvalue});

            // It is usually around 2e-5, but there are some outliers...
            REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));


            // With textures:
            const auto input_texture = noa::xform::Texture<f32>(input, device, interp, {
                .border = border,
                .cvalue = cvalue,
                .prefilter= false, // already prefiltered
            });
            noa::fill(output, 0); // erase
            noa::xform::transform_2d(input_texture, output, inv_matrix);

            REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("xform::transform_2d, cpu vs gpu", "", f32, f64, c32, c64) {
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
    const auto rotation = noa::deg2rad(test::Randomizer<f64>(-360., 360.).get());
    const auto shape = test::random_shape_batched(2);
    const auto center = shape.filter(2, 3).vec.as<f64>() / test::Randomizer<f64>(1, 4).get();
    const auto inverse_rotation_matrix =
        noa::xform::translate(center) *
        noa::xform::affine(noa::xform::rotate(-rotation)) *
        noa::xform::translate(-center);

    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = input_cpu.to({.device="gpu", .allocator="unified"});
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::like(input_gpu);

    if (interp.is_almost_any(Interp::CUBIC_BSPLINE)) {
        noa::xform::cubic_bspline_prefilter(input_cpu, input_cpu);
        noa::xform::cubic_bspline_prefilter(input_gpu, input_gpu);
    }
    noa::xform::transform_2d(input_cpu, output_cpu, inverse_rotation_matrix, {interp, border, value});
    noa::xform::transform_2d(input_gpu, output_gpu, inverse_rotation_matrix, {interp, border, value});

    REQUIRE(test::allclose_abs(output_cpu, output_gpu, 5e-4f));
}

TEMPLATE_TEST_CASE("xform::transform_2d(), texture interpolation", "", f32, c32) {
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
        Border::VALUE,
        Border::CLAMP,
        Border::MIRROR,
        Border::PERIODIC,
        Border::REFLECT
    );
    INFO(interp);
    INFO(border);

    // Here we want to compare CPU and GPU textures, so use double precision for the transformation,
    // otherwise most of the error would come from the transformation, especially for large sizes where
    // the fraction is close or past the single floating point precision.
    const auto value = test::Randomizer<TestType>(-3., 3.).get();
    const auto rotation = noa::deg2rad(test::Randomizer<f64>(-360., 360.).get());
    INFO(rotation);
    auto shape = test::random_shape_batched(2);
    const auto center = shape.filter(2, 3).vec.as<f64>() / test::Randomizer<f64>(1, 4).get();
    const auto inverse_rotation_matrix =
        noa::xform::translate(center) *
        noa::xform::affine(noa::xform::rotate(-rotation)) *
        noa::xform::translate(-center);

    const auto gpu_options = ArrayOption{.device="gpu", .allocator="unified"};
    const auto input_cpu = noa::random(noa::Uniform<TestType>{-2, 2}, shape);
    const auto input_gpu = noa::xform::Texture<TestType>(input_cpu, gpu_options.device, interp, {
        .border = border, .cvalue = value, .prefilter  = false
    });
    const auto output_cpu = noa::like(input_cpu);
    const auto output_gpu = noa::empty<TestType>(shape, gpu_options);

    noa::xform::transform_2d(input_cpu, output_cpu, inverse_rotation_matrix, {interp, border, value});
    noa::xform::transform_2d(input_gpu, output_gpu, inverse_rotation_matrix);

    const bool is_textureable = border.is_any(Border::ZERO, Border::CLAMP, Border::MIRROR, Border::PERIODIC);
    f32 epsilon = 1e-5f; // usually around 1e-6 and 5e-6
    if (is_textureable and (interp == Interp::LINEAR_FAST or interp == Interp::CUBIC_BSPLINE_FAST))
        epsilon = 0.035f; // usually around 0.01-0.03

    const test::MatchResult results = test::allclose_abs(output_cpu, output_gpu, epsilon);
    if (interp == Interp::NEAREST_FAST) {
        // For nearest-neighbor, the border can be off by one pixel due to floating-point imprecision,
        // so here check the average difference...
        REQUIRE_THAT(noa::abs(results.total_abs_diff) / static_cast<f32>(shape.n_elements()), Catch::Matchers::WithinAbs(0, 5e-4));
    } else {
        // fmt::println("interp={}, border={}, error={}", interp, border, results.max_abs_diff);
        REQUIRE(results);
    }
}
