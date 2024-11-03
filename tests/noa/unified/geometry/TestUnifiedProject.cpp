#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/geometry/Project.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/core/utils/Zip.hpp>
#include <catch2/catch.hpp>

#include "Assets.h"
#include "Utils.hpp"

using namespace noa::types;
namespace ng = noa::geometry;

TEST_CASE("unified::geometry::backward_project_3d, backproject sphere") {
    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["backward_project_3d"][0];
    const auto input_filename = path_base / param["input"].as<Path>();
    const auto output_filename = path_base / param["output"].as<Path>();

    constexpr size_t n_images = 5;
    auto shifts = Array<Vec<f64, 2>>::from_values(
        Vec{23., 32.},
        Vec{33., -42.},
        Vec{-52., -22.},
        Vec{13., -18.},
        Vec{37., 17.}
    );

    constexpr bool COMPUTE_ASSETS = true;
    if constexpr (COMPUTE_ASSETS) {
        const auto inverse_matrices = noa::empty<Mat<f64, 2, 3>>(n_images);
        for (auto&& [matrix, shift]: noa::zip(inverse_matrices.span_1d(), shifts.span_1d()))
            matrix = ng::translate(-shift).pop_back();

        constexpr auto circle = ng::Sphere{.center = Vec{128., 128.}, .radius = 32., .smoothness = 5.};
        const auto asset = noa::empty<f32>({n_images, 1, 256, 256});
        ng::draw_shape({}, asset, circle, inverse_matrices);
        noa::io::write(asset, input_filename);
    }

    constexpr i64 volume_depth = 80;
    constexpr auto center = Vec{40., 128., 128.};
    const auto tilts = Array<f64>::from_values(-60, -30, 0, 30, 60);

    auto backward_projection_matrices = noa::empty<Mat<f64, 2, 4>>(n_images);
    auto forward_projection_matrices = noa::empty<Mat<f64, 3, 4>>(n_images);
    for (auto&& [backward_matrix, forward_matrix, shift, tilt]: noa::zip(
             backward_projection_matrices.span_1d(),
             forward_projection_matrices.span_1d(),
             shifts.span_1d(),
             tilts.span_1d())) {
        auto matrix =
            ng::translate((center.pop_front() + shift).push_front(0)) *
            ng::linear2affine(ng::rotate_y(noa::deg2rad(tilt))) *
            ng::translate(-center);

        backward_matrix = matrix.filter(1, 2);
        forward_matrix = matrix.inverse().pop_back();
    }

    auto options = ArrayOption{.device="gpu"};
    if (options.device != Device()) {
        backward_projection_matrices = std::move(backward_projection_matrices).to(options);
        forward_projection_matrices = std::move(forward_projection_matrices).to(options);
    }

    // auto images = noa::io::read_data<f32>(input_filename, {}, options);
    // auto volume = noa::empty<f32>({1, volume_depth, 256, 256}, options);
    // ng::backward_project_3d(images, volume, backward_projection_matrices);
    // noa::io::write(volume, output_filename);


    const auto input_volume = noa::empty<f32>({1, volume_depth, 256, 256}, options);
    ng::draw_shape({}, input_volume, ng::Sphere{.center = Vec{40., 128., 128.}, .radius = 32., .smoothness = 0.});
    noa::io::write(input_volume, path_base / "input_volume.mrc");

    Stream::current("cpu").set_thread_limit(1);

    auto projected_images = noa::zeros<f32>({1, 1, 256, 256}, options);
    ng::forward_project_3d(input_volume, projected_images, ng::translate(Vec{100., 0., 0.}), {.add_to_output = true});
    noa::io::write(projected_images, output_filename);

    // std::vector<Device> devices{"cpu"};
    // if (not COMPUTE_ASSETS and Device::is_any_gpu())
    //     devices.emplace_back("gpu");
}
