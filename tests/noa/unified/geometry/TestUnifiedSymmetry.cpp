#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/CubicBSplinePrefilter.hpp>
#include <noa/unified/geometry/Draw.hpp>
#include <noa/unified/geometry/Symmetry.hpp>
#include <noa/unified/geometry/Transform.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/Factory.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

namespace ng = ::noa::geometry;
using namespace ::noa::types;
using Interp = noa::Interp;

TEST_CASE("unified::geometry::symmetrize_2d", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_2d"];
    const auto input_filename = path_base / param["input"].as<Path>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32>({1, 1, 512, 512});
        const auto center = Vec{256., 256.};
        ng::draw({}, asset, ng::Rectangle{
            .center = center,
            .radius = Vec{64., 128.},
            .smoothness = 5.,
        }.draw());
        ng::draw(asset, asset, ng::Rectangle{
            .center = center + Vec{64., 128.},
            .radius = Vec{32., 32.},
            .smoothness = 3.,
        }.draw(), {}, noa::Plus{});
        ng::draw(asset, asset, ng::Rectangle{
            .center = center,
            .radius = Vec{2. ,2.},
            .smoothness = 0.,
        }.draw(), {}, noa::Plus{});
        noa::io::write(asset, input_filename);
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
        const auto center = test["center"].as<Vec2<f64>>();
        const auto interp = test["interp"].as<Interp>();
        const auto pre_shift = test["pre_shift"].as<Vec2<f64>>();
        const auto post_shift = test["post_shift"].as<Vec2<f64>>();

        const auto inverse_pre_matrix = ng::translate(-pre_shift).as<f32>();
        const auto inverse_post_matrix = (
            ng::translate(center + post_shift) *
            ng::linear2affine(ng::rotate(angle)) *
            ng::translate(-center)
        ).inverse().as<f32>();

        auto symmetry = ng::Symmetry<f32, 2>(test["symmetry"].as<std::string>());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (symmetry.device() != device)
                symmetry = symmetry.to({device});

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = true}, options);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(input);
            ng::symmetrize_2d(
                input, output, symmetry,
                {.symmetry_center = center, .interp = interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::write(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = true}, options);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = noa::Texture<f32>(input, device, interp, {
                .border = noa::Border::ZERO,
                .cvalue = 0.f,
                .prefilter = false,
            });

            noa::fill(output, 0); // erase
            ng::symmetrize_2d(
                input_texture, output, symmetry,
                {.symmetry_center=center},
                inverse_pre_matrix, inverse_post_matrix);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_3d, symmetry", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["symmetry_3d"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = noa::empty<f32>({1, 150, 150, 150});
        constexpr auto center = Vec<f64, 3>::from_value(150 / 2);
        ng::draw({}, asset, ng::Rectangle{.center = center, .radius = Vec{34., 24., 24.}, .smoothness = 3.}.draw());
        noa::io::write(asset, path_base / param["input"][0].as<Path>());

        ng::draw(asset, asset, ng::Rectangle{
            .center = center + Vec{15., 15., 15.},
            .radius = Vec{15., 15., 15.},
            .smoothness = 3.
        }.draw(), {}, noa::Plus{});
        noa::io::write(asset, path_base / param["input"][1].as<Path>());
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
        const auto shift = test["shift"].as<Vec3<f64>>();
        const auto angles = noa::deg2rad(test["angles"].as<Vec3<f64>>());
        const auto center = test["center"].as<Vec3<f64>>();
        const auto interp = test["interp"].as<Interp>();

        const auto inverse_pre_matrix = ng::translate(-shift).as<f32>();
        const auto inverse_post_matrix = (
            ng::translate(center) *
            ng::linear2affine(ng::euler2matrix(angles, {.axes="zyz"})) *
            ng::translate(-center)
        ).inverse().as<f32>();

        auto symmetry = ng::Symmetry<f32, 3>(test["symmetry"].as<std::string>());

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            if (symmetry.device() != device)
                symmetry = symmetry.to({device});

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = false}, options);
            if (interp.is_almost_any(Interp::CUBIC_BSPLINE))
                noa::cubic_bspline_prefilter(input, input);

            // With arrays:
            const auto output = noa::like(input);

            ng::symmetrize_3d(
                input, output, symmetry,
                {.symmetry_center=center, .interp=interp},
                inverse_pre_matrix, inverse_post_matrix);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::write(output, expected_filename);
                continue;
            }

            const auto expected = noa::io::read_data<f32>(expected_filename, {.enforce_2d_stack = false}, options);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4)); // usually around 2e-5, with some outliers...
            ++count;

            // With textures:
            const auto input_texture = noa::Texture<f32>(input, device, interp, {
                .border=noa::Border::ZERO,
                .cvalue=0.f,
                .prefilter=false,
            });
            noa::fill(output, 0.); // erase
            ng::symmetrize_3d(
                input_texture, output, symmetry,
                {.symmetry_center=center},
                inverse_pre_matrix, inverse_post_matrix);
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-4));
        }
    }
    REQUIRE(count == expected_count);
}
