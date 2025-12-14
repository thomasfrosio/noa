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

TEST_CASE("unified::geometry::rasterize_central_slices_3d", "[asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["rasterize_central_slices_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb{}; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4>();
        const auto target_shape = parameters["target_shape"].as<Shape4>();
        const auto scale = Vec<f32, 2>::from_value(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f64>();
        const auto ews_radius = Vec<f64, 2>::from_value(parameters["ews_radius"].as<f64>());
        const auto volume_filename = path / parameters["volume_filename"].as<Path>();

        const auto inv_scaling_matrix = noa::geometry::scale(1 / scale);
        auto fwd_rotation_matrices = noa::empty<Mat33<f32>>(std::ssize(rotate));
        for (size_t i{}; auto& fwd_rotation_matrix: fwd_rotation_matrices.span_1d_contiguous())
            fwd_rotation_matrix = noa::geometry::euler2matrix(
                noa::deg2rad(Vec{0.f, rotate[i++], 0.f}), {.axes="zyx"});

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption{device, Allocator::MANAGED};
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to({device});

            // Backward project.
            const Array slice_fft = noa::linspace(slice_shape.rfft(), noa::Linspace<f32>{1, 10, true}, options);
            const Array volume_fft = noa::zeros<f32>(volume_shape.rfft(), options);
            noa::geometry::rasterize_central_slices_3d<"HC2HC">(
                slice_fft, {}, slice_shape, volume_fft, {}, volume_shape,
                inv_scaling_matrix, fwd_rotation_matrices,
                {fftfreq_cutoff, target_shape, ews_radius});

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(volume_fft, volume_filename);
            } else {
                const Array asset_volume_fft = noa::read_image<f32>(volume_filename).data;
                REQUIRE(test::allclose_abs_safe(asset_volume_fft, volume_fft, 1e-5));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::rasterize_central_slices_3d, remap", "", f32, c32) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto slice_shape = Shape4{20, 1, 64, 64};
    constexpr auto grid_shape = Shape4{1, 128, 128, 128};

    Array<Mat33<f32>> fwd_rotation_matrices(slice_shape[0]);
    for (size_t i{}; auto& matrix: fwd_rotation_matrices.span_1d_contiguous())
        matrix = noa::geometry::euler2matrix(
            noa::deg2rad(Vec<f32, 3>::from_values(0, i++ * 2, 0)), {.axes="zyx"});

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption{device, Allocator::MANAGED};
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to({device});

        const Array slice_fft = noa::linspace(slice_shape.rfft(), noa::Linspace{TestType{1}, TestType{10}, true}, options);
        const Array grid_fft0 = noa::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();
        const Array grid_fft2 = grid_fft0.copy();

        // With centered slices.
        ng::rasterize_central_slices_3d<"hc2hc">(
            slice_fft, {}, slice_shape, grid_fft0, {}, grid_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        ng::rasterize_central_slices_3d<"HC2H">(
            slice_fft, {}, slice_shape, grid_fft1, {}, grid_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        noa::fft::remap("h2hc", grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft2, 5e-5));

        // With non-centered slices.
        noa::fill(grid_fft0, TestType{});
        noa::fill(grid_fft1, TestType{});
        ng::rasterize_central_slices_3d<"h2hc">(
            slice_fft, {}, slice_shape, grid_fft0, {}, grid_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        ng::rasterize_central_slices_3d<"h2h">(
            slice_fft, {}, slice_shape, grid_fft1, {}, grid_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        noa::fft::remap("h2hc", grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft2, 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::rasterize_central_slices_3d, weights", "", f32, f64) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto slice_shape = Shape4{20, 1, 64, 64};
    constexpr auto grid_shape = Shape4{1, 64, 64, 64};

    auto fwd_rotation_matrices = noa::empty<Mat33<f64>>(slice_shape[0]);
    for (i64 i{}; auto& fwd_rotation_matrix: fwd_rotation_matrices.span_1d_contiguous())
        fwd_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec<f64, 3>::from_values(0, i * 2, 0)), {.axes="zyx"});

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption{device, Allocator::MANAGED};
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to({device});

        const Array slice_fft = noa::random(noa::Uniform<TestType>{-10, 10}, slice_shape.rfft(), options);
        const Array grid_fft0 = noa::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();

        ng::rasterize_central_slices_3d<"hc2hc">(
            slice_fft, slice_fft.copy(), slice_shape, grid_fft0, grid_fft1, grid_shape,
            {}, fwd_rotation_matrices, {.fftfreq_cutoff = 0.45});
        REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft1, 5e-5));
    }
}
