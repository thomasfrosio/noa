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

TEST_CASE("unified::geometry::insert_central_slices_3d", "[asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert_central_slices_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb{}; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4<i64>>();
        const auto target_shape = parameters["target_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>::from_value(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f64>();
        const auto ews_radius = Vec2<f64>::from_value(parameters["ews_radius"].as<f64>());
        const auto fftfreq_sinc = parameters["fftfreq_sinc"].as<f64>();
        const auto fftfreq_blackman = parameters["fftfreq_blackman"].as<f64>();
        const auto volume_filename = path / parameters["volume_filename"].as<Path>();

        const auto fwd_scaling_matrix = ng::scale(scale);
        auto inv_rotation_matrices = noa::empty<Mat33<f32>>(static_cast<i64>(rotate.size()));
        for (size_t i{}; auto& inv_rotation_matrix: inv_rotation_matrices.span_1d_contiguous())
            inv_rotation_matrix = ng::rotate_y(noa::deg2rad(-rotate[i++]));

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption{device, Allocator::MANAGED};
            INFO(device);

            if (inv_rotation_matrices.device() != device)
                inv_rotation_matrices = std::move(inv_rotation_matrices).to({device});

            // Backward project.
            const Array slice_fft = noa::linspace(slice_shape.rfft(), noa::Linspace{1.f, 10.f, true}, options);
            const Array volume_fft = noa::zeros<f32>(volume_shape.rfft(), options);
            ng::insert_central_slices_3d<"hc2hc">(
                slice_fft.eval(), {}, slice_shape, volume_fft, {}, volume_shape,
                fwd_scaling_matrix, inv_rotation_matrices, {
                    .interp = Interp::LINEAR,
                    .windowed_sinc = {fftfreq_sinc, fftfreq_blackman},
                    .fftfreq_cutoff = fftfreq_cutoff,
                    .target_shape = target_shape,
                    .ews_radius = ews_radius,
                });

            if constexpr (COMPUTE_ASSETS) {
                noa::write_image(volume_fft, volume_filename);
                continue;
            }

            const Array asset_volume_fft = noa::read_image<f32>(volume_filename).data;
            REQUIRE(test::allclose_abs_safe(asset_volume_fft, volume_fft, 5e-5));
        }
    }
}

// TEMPLATE_TEST_CASE("unified::geometry::insert_central_slices_3d, weights", "", f32, f64) {
//     std::vector<Device> devices{"cpu"};
//     if (Device::is_any_gpu())
//         devices.emplace_back("gpu");
//
//     constexpr auto slice_shape = Shape4<i64>{20, 1, 64, 64};
//     constexpr auto grid_shape = Shape4<i64>{1, 64, 64, 64};
//
//     auto fwd_rotation_matrices = noa::empty<Mat33<f32>>(slice_shape[0]);
//     for (i64 i{}; auto& fwd_rotation_matrix: fwd_rotation_matrices.span_1d_contiguous())
//         fwd_rotation_matrix = ng::rotate_y(noa::deg2rad(static_cast<f32>(i++ * 2)));
//
//     for (auto& device: devices) {
//         const auto stream = StreamGuard(device);
//         const auto options = ArrayOption{device, Allocator::MANAGED};
//         INFO(device);
//
//         if (fwd_rotation_matrices.device() != device)
//             fwd_rotation_matrices = fwd_rotation_matrices.to({device});
//
//         const Array slice_fft = noa::random(noa::Uniform<TestType>{-10, 10}, slice_shape.rfft(), options);
//         const Array grid_fft0 = noa::zeros<TestType>(grid_shape.rfft(), options);
//         const Array grid_fft1 = grid_fft0.copy();
//
//         ng::insert_central_slices_3d<"hc2hc">(
//             slice_fft, slice_fft.copy(), slice_shape,
//             grid_fft0, grid_fft1, grid_shape,
//             {}, fwd_rotation_matrices, {
//                 .windowed_sinc = {.fftfreq_sinc = 0.02, .fftfreq_blackman = 0.06},
//                 .fftfreq_cutoff = 0.45
//             });
//         REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft1, 5e-5));
//     }
// }
//
// TEMPLATE_TEST_CASE("unified::geometry::insert_central_slices_3d, texture/remap", "", f32, c32) {
//     std::vector<Device> devices{"cpu"};
//     if (Device::is_any_gpu())
//         devices.emplace_back("gpu");
//
//     const i64 slice_count = GENERATE(1, 20);
//     const auto slice_shape = Shape4<i64>{slice_count, 1, 64, 64};
//     constexpr auto grid_shape = Shape4<i64>{1, 128, 128, 128};
//     constexpr auto windowed_sinc = ng::WindowedSinc{0.02, 0.06};
//
//     auto inv_rotation_matrices = noa::empty<Mat33<f32>>(slice_shape[0]);
//     for (i64 i{}; auto& inv_rotation_matrix: inv_rotation_matrices.span_1d_contiguous())
//         inv_rotation_matrix = ng::rotate_y(noa::deg2rad(-static_cast<f32>(i++) * 2));
//
//     for (auto& device: devices) {
//         const auto stream = StreamGuard(device);
//         const auto options = ArrayOption{device, Allocator::MANAGED};
//         INFO(device);
//
//         if (inv_rotation_matrices.device() != device)
//             inv_rotation_matrices = inv_rotation_matrices.to({device});
//
//         const Array slice_fft = noa::linspace(slice_shape.rfft(), noa::Linspace<TestType>{1, 10, true}, options);
//         const Array grid_fft0 = noa::zeros<TestType>(grid_shape.rfft(), options);
//         const Array grid_fft1 = grid_fft0.copy();
//         const Array grid_fft2 = grid_fft0.copy();
//
//         { // Texture
//             const auto texture_slice_fft = Texture<TestType>{slice_fft, device, Interp::LINEAR};
//             ng::insert_central_slices_3d<"hc2h">(
//                 slice_fft, {}, slice_shape, grid_fft0, {}, grid_shape,
//                 {}, inv_rotation_matrices, {.windowed_sinc=windowed_sinc, .fftfreq_cutoff=0.45});
//             ng::insert_central_slices_3d<"hc2h">(
//                 texture_slice_fft, {}, slice_shape, grid_fft1, {}, grid_shape,
//                 {}, inv_rotation_matrices, {.windowed_sinc=windowed_sinc, .fftfreq_cutoff=0.45});
//
//             REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft1, 5e-5));
//         }
//
//         { // Remap
//             noa::fill(grid_fft0, {});
//             noa::fill(grid_fft1, {});
//             ng::insert_central_slices_3d<"hc2hc">(
//                 slice_fft, {}, slice_shape, grid_fft0, {}, grid_shape,
//                 {}, inv_rotation_matrices, {.windowed_sinc = windowed_sinc, .fftfreq_cutoff = 0.5});
//             ng::insert_central_slices_3d<"hc2h">(
//                 slice_fft, {}, slice_shape, grid_fft1, {}, grid_shape,
//                 {}, inv_rotation_matrices, {.windowed_sinc = windowed_sinc, .fftfreq_cutoff = 0.5});
//             noa::fft::remap("h2hc", grid_fft1, grid_fft2, grid_shape);
//             REQUIRE(test::allclose_abs_safe(grid_fft0, grid_fft2, 5e-5));
//         }
//     }
// }
