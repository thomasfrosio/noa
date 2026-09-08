#include <noa/xform/core/Euler.hpp>
#include <noa/xform/CubicBSplinePrefilter.hpp>
#include <noa/xform/Draw.hpp>
#include <noa/xform/Symmetry.hpp>
#include <noa/xform/Transform.hpp>
#include <noa/IO.hpp>
#include <noa/Runtime.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

namespace nx = ::noa::xform;
using namespace ::noa::types;
using Interp = nx::Interp;

TEST_CASE("xform::symmetrize_2d", "[asset]") {
    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_2d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32, 2>({512, 512});
        const auto center = Vec{256., 256.};
        nx::draw_2d({}, asset, nx::Rectangle{
            .center = center,
            .radius = Vec{64., 128.},
            .smoothness = 5.,
        }.get());
        nx::draw_2d(asset, asset, nx::Rectangle{
            .center = center + Vec{64., 128.},
            .radius = Vec{32., 32.},
            .smoothness = 3.,
        }.get(), {}, noa::Plus{});
        nx::draw_2d(asset, asset, nx::Rectangle{
            .center = center,
            .radius = Vec{2. ,2.},
            .smoothness = 0.,
        }.get(), {}, noa::Plus{});
        noa::write_image(asset, input_filename);
    }

    auto devices = std::vector<Device>{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const usize expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    usize count{};
    for (usize nb{}; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto angle = noa::deg2rad(test["angle"].as<f64>());
        const auto center = test["center"].as<Vec<f64, 2>>();
        const auto interp = test["interp"].as<Interp>();
        const auto pre_shift = test["pre_shift"].as<Vec<f64, 2>>();
        const auto post_shift = test["post_shift"].as<Vec<f64, 2>>();

        const auto inverse_pre_matrix = nx::translate(-pre_shift).as<f32>();
        const auto inverse_post_matrix = (
            nx::translate(center + post_shift) *
            nx::affine(nx::rotate(angle)) *
            nx::translate(-center)
        ).inverse().as<f32>();

        auto symmetry = nx::Symmetry<f32, 2>(test["symmetry"].as<std::string>());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (symmetry.device() != device)
                symmetry = symmetry.to(options);

            const auto input = noa::read_image<f32, 2>(input_filename, {.rank = 2}, options).data;
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter_2d(input, input);

            // With arrays:
            const auto output = noa::empty_like(input);
            nx::symmetrize_2d(
                input, output, symmetry,
                {.symmetry_center = center, .interp = interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(output, expected_filename);
                continue;
            }

            const auto expected = noa::read_image<f32, 2>(expected_filename, {.rank = 2}, options).data;
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = nx::Texture2D<f32, 2>(input, device, interp, {
                .border = noa::Border::ZERO,
                .cvalue = 0.f,
                .prefilter = false,
            });

            noa::fill(output, 0); // erase
            nx::symmetrize_2d(
                input_texture, output, symmetry,
                {.symmetry_center=center},
                inverse_pre_matrix, inverse_post_matrix);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("xform::symmetrize_3d", "[asset]") {
    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_3d"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32, 3>({150, 150, 150});
        constexpr auto center = Vec<f64, 3>::from_value(150 / 2);
        nx::draw_3d({}, asset, nx::Rectangle{.center = center, .radius = Vec{34., 24., 24.}, .smoothness = 3.}.get());
        noa::write_image(asset, path_base / param["input"][0].as<Path>());

        nx::draw_3d(asset, asset, nx::Rectangle{
            .center = center + Vec{15., 15., 15.},
            .radius = Vec{15., 15., 15.},
            .smoothness = 3.
        }.get(), {}, noa::Plus{});
        noa::write_image(asset, path_base / param["input"][1].as<Path>());
    }

    auto devices = std::vector<Device>{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const usize expected_count = param["tests"].size() * devices.size();
    usize count{};
    for (usize nb{}; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto shift = test["shift"].as<Vec<f64, 3>>();
        const auto angles = noa::deg2rad(test["angles"].as<Vec<f64, 3>>());
        const auto center = test["center"].as<Vec<f64, 3>>();
        const auto interp = test["interp"].as<Interp>();

        const auto inverse_pre_matrix = nx::translate(-shift).as<f32>();
        const auto inverse_post_matrix = (
            nx::translate(center) *
            nx::affine(nx::euler2matrix(angles, {.axes="zyz"})) *
            nx::translate(-center)
        ).inverse().as<f32>();

        auto symmetry = nx::Symmetry<f32, 3>(test["symmetry"].as<std::string>());

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            if (symmetry.device() != device)
                symmetry = symmetry.to(options);

            const auto input = noa::read_image<f32, 3>(input_filename, {.rank = 3}, options).data;
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter_2d(input, input);

            // With arrays:
            const auto output = noa::empty_like(input);

            nx::symmetrize_3d(
                input, output, symmetry,
                {.symmetry_center=center, .interp=interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(output, expected_filename);
                continue;
            }

            const auto expected = noa::read_image<f32, 3>(expected_filename, {.rank = 3}, options).data;
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = nx::Texture3D<f32, 3>(input, device, interp, {
                .border=noa::Border::ZERO,
                .cvalue=0.f,
                .prefilter=false,
            });
            noa::fill(output, 0.); // erase
            nx::symmetrize_3d(
                input_texture, output, symmetry,
                {.symmetry_center=center},
                inverse_pre_matrix, inverse_post_matrix);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("xform::symmetrize batched - cpu vs gpu") {
    if (not Device::is_any_gpu())
        return;

    constexpr auto SHAPE_OPTIONS = test::RandomShapeOptions{.size_range = {32, 64}, .batch_range = {1, 4}};
    const auto shapes = noa::make_tuple(
        Pair{test::random_shape_batched<isize, 3>(2, SHAPE_OPTIONS), Tag<2>{}},
        Pair{test::random_shape_batched<isize, 6>(2, SHAPE_OPTIONS), Tag<2>{}},
        Pair{test::random_shape_batched<isize, 4>(3, SHAPE_OPTIONS), Tag<3>{}},
        Pair{test::random_shape_batched<isize, 6>(3, SHAPE_OPTIONS), Tag<3>{}}
    );
    shapes.for_each([&]<usize N, usize R>(const Pair<Shape<isize, N>, Tag<R>>& pair) {
        const auto& shape = pair.first;
        const auto cpu_symmetry = nx::Symmetry<f32, R>("C3");
        const auto cpu_input = noa::random<f32, N>(noa::Uniform{-10.f, 10.f}, shape);
        const auto cpu_output = noa::empty_like(cpu_input);

        const auto gpu_options = ArrayOption{.device = "gpu", .allocator = Allocator::MANAGED};
        const auto gpu_symmetry = cpu_symmetry.to(gpu_options);
        const auto gpu_input = cpu_input.to(gpu_options);
        const auto gpu_output = noa::empty_like(gpu_input);

        if constexpr (R == 2) {
            nx::symmetrize_2d(cpu_input, cpu_output, gpu_symmetry);
            nx::symmetrize_2d(gpu_input, gpu_output, gpu_symmetry);
        } else {
            nx::symmetrize_3d(cpu_input, cpu_output, gpu_symmetry);
            nx::symmetrize_3d(gpu_input, gpu_output, gpu_symmetry);
        }

        REQUIRE(test::allclose_abs_safe(cpu_input, cpu_output, 5e-4));
    });
}
