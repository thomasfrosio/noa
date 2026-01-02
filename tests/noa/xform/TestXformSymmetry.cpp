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
    const Path path_base = test::NOA_DATA_PATH / "xform";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_2d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32>({1, 1, 512, 512});
        const auto center = Vec{256., 256.};
        nx::draw({}, asset, nx::Rectangle{
            .center = center,
            .radius = Vec{64., 128.},
            .smoothness = 5.,
        }.draw());
        nx::draw(asset, asset, nx::Rectangle{
            .center = center + Vec{64., 128.},
            .radius = Vec{32., 32.},
            .smoothness = 3.,
        }.draw(), {}, noa::Plus{});
        nx::draw(asset, asset, nx::Rectangle{
            .center = center,
            .radius = Vec{2. ,2.},
            .smoothness = 0.,
        }.draw(), {}, noa::Plus{});
        noa::write_image(asset, input_filename);
    }

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{};
    for (size_t nb{}; nb < param["tests"].size(); ++nb) {
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
                symmetry = symmetry.to({device});

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = true}, options).data;
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(input);
            nx::symmetrize_2d(
                input, output, symmetry,
                {.symmetry_center = center, .interp = interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(output, expected_filename);
                continue;
            }

            const auto expected = noa::read_image<f32>(expected_filename, {.enforce_2d_stack = true}, options).data;
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = nx::Texture<f32>(input, device, interp, {
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

TEST_CASE("xform::transform_3d, symmetry", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "xform";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_3d"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32>({1, 150, 150, 150});
        constexpr auto center = Vec<f64, 3>::from_value(150 / 2);
        nx::draw({}, asset, nx::Rectangle{.center = center, .radius = Vec{34., 24., 24.}, .smoothness = 3.}.draw());
        noa::write_image(asset, path_base / param["input"][0].as<Path>());

        nx::draw(asset, asset, nx::Rectangle{
            .center = center + Vec{15., 15., 15.},
            .radius = Vec{15., 15., 15.},
            .smoothness = 3.
        }.draw(), {}, noa::Plus{});
        noa::write_image(asset, path_base / param["input"][1].as<Path>());
    }

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    size_t count{};
    for (size_t nb{}; nb < param["tests"].size(); ++nb) {
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
                symmetry = symmetry.to({device});

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = false}, options).data;
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                nx::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(input);

            nx::symmetrize_3d(
                input, output, symmetry,
                {.symmetry_center=center, .interp=interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(output, expected_filename);
                continue;
            }

            const auto expected = noa::read_image<f32>(expected_filename, {.enforce_2d_stack = false}, options).data;
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = nx::Texture<f32>(input, device, interp, {
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
