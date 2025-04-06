#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/geometry/Project.hpp>
#include <noa/unified/IO.hpp>
#include <noa/core/utils/Zip.hpp>

#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/Reduce.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;
namespace ng = noa::geometry;

TEST_CASE("unified::geometry::project_3d, project sphere", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["project_3d"][0];
    const auto input_filename = path_base / param["input_images"].as<Path>();
    const auto output_volume_filename = path_base / param["output_volume"].as<Path>();
    const auto output_images_filename = path_base / param["output_images"].as<Path>();

    constexpr size_t n_images = 5;
    const auto tilts = std::array{-60., -30., 0., 30., 60.};
    auto shifts = std::array{
        Vec{23., 32.},
        Vec{33., -42.},
        Vec{-52., -22.},
        Vec{13., -18.},
        Vec{37., 17.},
    };

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto inverse_matrices = noa::empty<Mat<f64, 2, 3>>(n_images);
        for (auto&& [matrix, shift]: noa::zip(inverse_matrices.span_1d(), shifts))
            matrix = ng::translate(-shift).pop_back();

        constexpr auto circle = ng::Sphere{.center = Vec{128., 128.}, .radius = 32., .smoothness = 5.};
        const auto asset = noa::empty<f32>({n_images, 1, 256, 256});
        ng::draw_shape({}, asset, circle, inverse_matrices);
        noa::write(asset, input_filename);
    }

    constexpr auto volume_shape = Shape<i64, 3>{80, 256, 256};
    constexpr auto center = (volume_shape.vec / 2).as<f64>();

    auto backward_projection_matrices = noa::empty<Mat<f64, 2, 4>>(n_images);
    auto forward_projection_matrices = noa::empty<Mat<f64, 3, 4>>(n_images);
    i64 projection_window_size{};

    for (auto&& [backward_matrix, forward_matrix, shift, tilt]: noa::zip(
             backward_projection_matrices.span_1d(),
             forward_projection_matrices.span_1d(),
             shifts, tilts))
    {
        auto matrix =
            ng::translate((center.pop_front() + shift).push_front(0)) *
            ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
            ng::translate(-center);

        backward_matrix = matrix.filter_rows(1, 2);
        forward_matrix = matrix.inverse().pop_back();

        projection_window_size = noa::max(
            projection_window_size,
            ng::forward_projection_window_size(volume_shape, forward_matrix)
        );
    }

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};
        if (options.device != "cpu") {
            backward_projection_matrices = std::move(backward_projection_matrices).to(options);
            forward_projection_matrices = std::move(forward_projection_matrices).to(options);
        }

        // Backward project.
        {
            auto images = noa::read_data<f32>(input_filename, {}, options);
            auto volume = noa::empty<f32>(volume_shape.push_front(1), options);
            ng::backward_project_3d(images, volume, backward_projection_matrices);
            if constexpr (COMPUTE_ASSETS) {
                noa::write(volume, output_volume_filename);
            } else {
                auto asset_volume = noa::read_data<f32>(output_volume_filename, {}, options);
                REQUIRE(test::allclose_abs(volume, asset_volume, 1e-4));
            }
        }

        // Forward project.
        {
            const auto volume = noa::empty<f32>(volume_shape.push_front(1), options);
            ng::draw_shape({}, volume, ng::Sphere{.center = center, .radius = 32., .smoothness = 0.});

            auto images = noa::zeros<f32>({n_images, 1, 256, 256}, options);
            ng::forward_project_3d(
                volume, images, forward_projection_matrices,
                projection_window_size, {.add_to_output = true}
            );
            if constexpr (COMPUTE_ASSETS) {
                noa::write(images, output_images_filename);
            } else {
                auto asset_images = noa::read_data<f32>(output_images_filename, {}, options);
                REQUIRE(test::allclose_abs(images, asset_images, 1e-4));
            }
        }
    }
}

TEST_CASE("unified::geometry::project_3d, fused", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["project_3d"][1];
    const auto image_normal = path_base / param["image_normal"].as<Path>();
    const auto image_fused = path_base / param["image_fused"].as<Path>();

    constexpr bool COMPUTE_ASSETS = false;
    constexpr size_t n_images = 5;
    constexpr auto volume_shape = Shape<i64, 3>{80, 256, 256};
    constexpr auto center = (volume_shape.vec / 2).as<f64>();
    constexpr auto tilts = std::array{-60., -30., 0., 30., 60.};

    auto backward_projection_matrices = noa::empty<Mat<f64, 2, 4>>(n_images);
    for (auto&& [backward_matrix, tilt]: noa::zip(backward_projection_matrices.span_1d(), tilts)) {
        backward_matrix = (
            ng::translate((center.pop_front()).push_front(0)) *
            ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
            ng::translate(-center)
        ).filter_rows(1, 2);
    }

    auto forward_projection_matrix = (
        ng::translate((center.pop_front()).push_front(0)) *
        ng::linear2affine(ng::euler2matrix(noa::deg2rad(Vec{0., 40., 90.}), {.axes = "zyx", .intrinsic = false})) *
        ng::translate(-center)
    ).inverse().pop_back();
    i64 projection_window_size = ng::forward_projection_window_size(volume_shape, forward_projection_matrix);
    REQUIRE(projection_window_size == 259);

    constexpr auto circle = ng::Sphere{.center = Vec{128., 128.}, .radius = 32., .smoothness = 5.};
    auto images = noa::empty<f32>({n_images, 1, 256, 256});
    ng::draw_shape({}, images, circle);

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        auto options = ArrayOption{.device = device, .allocator = Allocator::MANAGED};

        auto projected_image = noa::zeros<f32>({1, 1, 256, 256}, options);
        if (options.device != Device()) {
            backward_projection_matrices = std::move(backward_projection_matrices).to(options);
            images = std::move(images).to(options);
        }

        { // normal
            auto volume = noa::empty<f32>(volume_shape.push_front(1), options);
            ng::backward_project_3d(images, volume, backward_projection_matrices, {.interp = noa::Interp::CUBIC});
            ng::forward_project_3d(
                volume, projected_image, forward_projection_matrix,
                projection_window_size, {.add_to_output = true}
            );
            if constexpr (COMPUTE_ASSETS) {
                noa::write(projected_image, image_normal);
            } else {
                auto asset_image = noa::read_data<f32>(image_normal, {}, options);
                REQUIRE(test::allclose_abs(projected_image, asset_image, 5e-4));
            }
        }

        { // fused
            ng::backward_and_forward_project_3d(
                images, projected_image, volume_shape,
                backward_projection_matrices, forward_projection_matrix,
                projection_window_size, {.interp = noa::Interp::CUBIC}
            );
            if constexpr (COMPUTE_ASSETS) {
                noa::write(projected_image, image_fused);
            } else {
                auto asset_image = noa::read_data<f32>(image_fused, {}, options);
                REQUIRE(test::allclose_abs(projected_image, asset_image, 5e-4));
            }
        }
    }
}

TEST_CASE("unified::geometry::project_3d, projection window", "[.]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";

    constexpr auto volume_shape = Shape<i64, 3>{64, 256, 256};
    constexpr auto center = (volume_shape.vec / 2).as<f64>();

    const auto volume = noa::ones<f32>(volume_shape.push_front(1));
    auto images = noa::zeros<f32>({1, 1, 256, 256});

    for (auto tilt: std::array{60.}) {
        auto forward_matrix = (
            ng::translate(center) *
            ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt + 180))) *
            ng::translate(-center)
        ).inverse().pop_back();

        auto projection_window_size = ng::forward_projection_window_size(volume_shape, forward_matrix);
        ng::forward_project_3d(volume, images, forward_matrix, projection_window_size, {.interp = noa::Interp::CUBIC});
        noa::write(images, path_base / "test_image.mrc");
    }
}
