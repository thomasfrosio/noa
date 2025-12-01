#include <noa/core/geometry/Transform.hpp>
#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/FourierProject.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/Factory.hpp>

#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace ng = noa::geometry;
using Interp = noa::Interp;

TEST_CASE("unified::geometry::extract_central_slices_3d", "[asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["extract_central_slices_3d"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset = tests["volumes"][0];
        const auto volume_filename = path / asset["filename"].as<Path>();
        const auto volume_shape = asset["shape"].as<Shape4<i64>>();

        const auto insert_inv_rotation_matrices = noa::empty<Mat33<f32>>(36);
        for (i64 i{}; auto& matrix: insert_inv_rotation_matrices.span_1d()) {
            matrix = ng::euler2matrix(
                noa::deg2rad(Vec{90.f, 0.f, static_cast<f32>(i++ * 10)}),
                {.axes = "xyz", .intrinsic = false}
            ).transpose();
        }
        const auto insert_slice_shape = Shape4<i64>{36, 1, volume_shape[2], volume_shape[3]};
        const Array volume_fft = noa::empty<f32>(volume_shape.rfft());

        const Array slice_fft = noa::linspace(insert_slice_shape.rfft(), noa::Linspace{1.f, 20.f});
        ng::insert_central_slices_3d<"hc2hc">(
            slice_fft, {}, insert_slice_shape, volume_fft, {}, volume_shape,
            {}, insert_inv_rotation_matrices, {.windowed_sinc = {-1, 0.1}});
        noa::write_image(volume_fft, volume_filename);
    }

    if constexpr (COMPUTE_ASSETS) {
        const auto asset_0 = tests["volumes"][1];
        const auto volume_filename = path / asset_0["filename"].as<Path>();
        const auto volume_shape = asset_0["shape"].as<Shape4<i64>>();
        const auto volume_fft = noa::empty<f32>(volume_shape.rfft());

        const auto slice_shape = Shape4<i64>{1, 1, volume_shape[2], volume_shape[3]};
        const auto slice_fft = noa::linspace(slice_shape.rfft(), noa::Linspace{1.f, 20.f});
        ng::insert_central_slices_3d<"hc2hc">(
            slice_fft, {}, slice_shape, volume_fft, {}, volume_shape,
            {}, Mat33<f32>::eye(1), {.windowed_sinc = {0.0234375, 1}});
        noa::write_image(volume_fft, volume_filename);
    }

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb{}; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);
        const YAML::Node& parameters = tests["tests"][nb];

        const auto volume_filename = path / parameters["volume_filename"].as<Path>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4<i64>>();

        const auto slice_filename = path / parameters["slice_filename"].as<Path>();
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto slice_scale = Vec2<f32>::from_value(parameters["slice_scale"].as<f32>());
        const auto slice_rotate = parameters["slice_rotate"].as<std::vector<f32>>();

        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f64>();
        const auto fftfreq_z_sinc = parameters["fftfreq_z_sinc"].as<f64>();
        const auto fftfreq_z_blackman = parameters["fftfreq_z_blackman"].as<f64>();
        const auto ews_radius = Vec2<f64>::from_value(parameters["ews_radius"].as<f64>());

        const Mat22<f32> inv_scaling_matrix = ng::scale(1 / slice_scale);
        auto fwd_rotation_matrices = noa::empty<Mat33<f32>>(static_cast<i64>(slice_rotate.size()));
        for (size_t i{}; auto& matrix: fwd_rotation_matrices.span_1d())
            matrix = ng::euler2matrix(noa::deg2rad(Vec3<f32>{0.f, slice_rotate[i++], 0.f}), {.axes = "zyx"});

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to({device});

            const Array volume_fft = noa::read_image<f32>(volume_filename, {}, options).data;
            const Array slice_fft = noa::empty<f32>(slice_shape.rfft(), options);

            // Forward project.
            ng::extract_central_slices_3d<"hc2hc">(
                volume_fft, {}, volume_shape, slice_fft, {}, slice_shape,
                inv_scaling_matrix, fwd_rotation_matrices, {
                    .w_windowed_sinc = {fftfreq_z_sinc, fftfreq_z_blackman},
                    .fftfreq_cutoff = fftfreq_cutoff,
                    .ews_radius = ews_radius
                });

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(slice_fft, slice_filename);
                continue;
            }

            const Array asset_slice_fft = noa::read_image<f32>(slice_filename).data;
            REQUIRE(test::allclose_abs_safe(asset_slice_fft, slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::extract_central_slices_3d, using texture API and remap", "", f32, c32) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto slice_shape = Shape4<i64>{20, 1, 128, 128};
    constexpr auto grid_shape = Shape4<i64>{1, 128, 128, 128};

    Array fwd_rotation_matrices = noa::empty<Mat33<f32>>(slice_shape[0]);
    for (i64 i{}; auto& fwd_rotation_matrix: fwd_rotation_matrices.span_1d())
        fwd_rotation_matrix = ng::rotate_y(noa::deg2rad(static_cast<f32>(i++ * 2)));

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption{device, Allocator::MANAGED};
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to({device});

        const Array grid_fft = noa::linspace(grid_shape.rfft(), noa::Linspace<TestType>{
            TestType{-50.}, TestType{50.}, true}, options);
        const Array slice_fft0 = noa::empty<TestType>(slice_shape.rfft(), options);
        const Array slice_fft1 = slice_fft0.copy();
        const Array slice_fft2 = slice_fft0.copy();

        const auto texture_grid_fft = Texture<TestType>(grid_fft, device, Interp::LINEAR);
        ng::extract_central_slices_3d<"hc2hc">(
            texture_grid_fft, {}, grid_shape,
            slice_fft0, {}, slice_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        ng::extract_central_slices_3d<"hc2h">(
            grid_fft, {}, grid_shape, slice_fft1, {}, slice_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        noa::fft::remap("h2hc", slice_fft1, slice_fft2, slice_shape);
        REQUIRE(test::allclose_abs_safe(slice_fft0, slice_fft2, 5e-5));
    }
}
