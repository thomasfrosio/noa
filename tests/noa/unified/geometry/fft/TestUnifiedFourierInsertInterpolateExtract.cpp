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

TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d", "[asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert_and_extract_central_slices_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb{}; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto input_slice_shape = parameters["input_slice_shape"].as<Shape4<i64>>();
        const auto output_slice_shape = parameters["output_slice_shape"].as<Shape4<i64>>();
        const auto input_rotate = parameters["input_rotate"].as<std::vector<f32>>();
        const auto output_rotate = parameters["output_rotate"].as<std::vector<f32>>();

        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f64>();
        const auto fftfreq_input_sinc = parameters["fftfreq_input_sinc"].as<f64>();
        const auto fftfreq_input_blackman = parameters["fftfreq_input_blackman"].as<f64>();
        const auto fftfreq_z_sinc = parameters["fftfreq_z_sinc"].as<f64>();
        const auto fftfreq_z_blackman = parameters["fftfreq_z_blackman"].as<f64>();
        const auto ews_radius = Vec2<f64>::from_value(parameters["ews_radius"].as<f64>());

        const auto output_slice_filename = path / parameters["output_slice_filename"].as<Path>();

        Array input_inv_rotation_matrices = noa::empty<Mat33<f32>>(static_cast<i64>(input_rotate.size()));
        Array output_fwd_rotation_matrices = noa::empty<Mat33<f32>>(static_cast<i64>(output_rotate.size()));
        for (size_t i{}; auto& input_inv_rotation_matrix: input_inv_rotation_matrices.span_1d())
            input_inv_rotation_matrix = ng::rotate_y(noa::deg2rad(-input_rotate[i++]));
        for (size_t i{}; auto& output_fwd_rotation_matrix: output_fwd_rotation_matrices.span_1d())
            output_fwd_rotation_matrix = ng::rotate_y(noa::deg2rad(output_rotate[i++]));

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption{device, Allocator::MANAGED};
            INFO(device);

            if (input_inv_rotation_matrices.device() != device)
                input_inv_rotation_matrices = input_inv_rotation_matrices.to({device});
            if (output_fwd_rotation_matrices.device() != device)
                output_fwd_rotation_matrices = output_fwd_rotation_matrices.to({device});

            const Array input_slice_fft = noa::linspace(input_slice_shape.rfft(), noa::Linspace{-50.f, 50.f, true}, options);
            const Array output_slice_fft = noa::empty<f32>(output_slice_shape.rfft(), options);

            ng::insert_and_extract_central_slices_3d<"hc2hc">(
                input_slice_fft, {}, input_slice_shape,
                output_slice_fft, {}, output_slice_shape,
                {}, input_inv_rotation_matrices,
                {}, output_fwd_rotation_matrices, {
                    .input_windowed_sinc = {fftfreq_input_sinc, fftfreq_input_blackman},
                    .w_windowed_sinc = {fftfreq_z_sinc, fftfreq_z_blackman},
                    .fftfreq_cutoff = fftfreq_cutoff,
                    .ews_radius = ews_radius,
                });

            if constexpr (COMPUTE_ASSETS) {
                noa::write(output_slice_fft, output_slice_filename);
                continue;
            }

            noa::write(output_slice_fft, path / "test_fourier_insert.mrc");
            const Array asset_output_slice_fft = noa::read_data<f32>(output_slice_filename);
            REQUIRE(test::allclose_abs_safe(asset_output_slice_fft, output_slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, texture/remap", "", f32, c32) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto input_slice_shape = Shape4<i64>{20, 1, 256, 256};
    constexpr auto output_slice_shape = Shape4<i64>{5, 1, 256, 256};

    Array<Mat33<f32>> input_inv_rotation_matrices(input_slice_shape[0]);
    for (i64 i{}; auto& input_inv_rotation_matrix: input_inv_rotation_matrices.span_1d_contiguous()) {
        input_inv_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec{0.f, -static_cast<f32>(i++ * 2), 0.f}), {.axes = "zyx"});
    }
    Array<Mat33<f32>> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (i64 i{}; auto& output_fwd_rotation_matrix: output_fwd_rotation_matrices.span_1d_contiguous()) {
        output_fwd_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec{0.f, static_cast<f32>(i++ * 2 + 1), 0.f}), {.axes = "zyx"});
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = input_inv_rotation_matrices.to({device});
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = output_fwd_rotation_matrices.to({device});

        const Array input_slice_fft = noa::linspace(input_slice_shape.rfft(), noa::Linspace{TestType{-50}, TestType{50}, true}, options);
        const Texture<TestType> texture_input_slice_fft(input_slice_fft, device, Interp::LINEAR);
        const Array output_slice_fft0 = noa::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft1 = noa::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft2 = noa::like(output_slice_fft1);

        ng::insert_and_extract_central_slices_3d<"hc2hc">(
            texture_input_slice_fft, {}, input_slice_shape,
            output_slice_fft0, {}, output_slice_shape,
            {}, input_inv_rotation_matrices,
            {}, output_fwd_rotation_matrices,
            {.input_windowed_sinc = {0.01}, .fftfreq_cutoff = 0.8});
        ng::insert_and_extract_central_slices_3d<"hc2h">(
            input_slice_fft, {}, input_slice_shape,
            output_slice_fft1, {}, output_slice_shape,
            {}, input_inv_rotation_matrices,
            {}, output_fwd_rotation_matrices,
            {.input_windowed_sinc = {0.01}, .fftfreq_cutoff = 0.8});
        noa::fft::remap("h2hc", output_slice_fft1, output_slice_fft2, output_slice_shape);
        REQUIRE(test::allclose_abs_safe(output_slice_fft0, output_slice_fft2, 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, weights", "", f32, f64) {
    std::vector<Device> devices = {"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto input_slice_shape = Shape4<i64>{20, 1, 256, 256};
    constexpr auto output_slice_shape = Shape4<i64>{5, 1, 256, 256};

    Array<Mat33<f32>> input_inv_rotation_matrices(input_slice_shape[0]);
    for (i64 i{}; auto& input_inv_rotation_matrix: input_inv_rotation_matrices.span_1d_contiguous()) {
        input_inv_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec{0.f, -static_cast<f32>(i++ * 2), 0.f}), {.axes = "zyx"});
    }
    Array<Mat33<f32>> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (i64 i{}; auto& output_fwd_rotation_matrix: output_fwd_rotation_matrices.span_1d_contiguous()) {
        output_fwd_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec{0.f, static_cast<f32>(i++ * 2 + 1), 0.f}), {.axes = "zyx"});
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = input_inv_rotation_matrices.to({device});
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = output_fwd_rotation_matrices.to({device});

        const Array input_slice_fft = noa::random(noa::Uniform<TestType>{-10, 10}, input_slice_shape.rfft(), options);
        const Array output_slice_fft0 = noa::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft1 = noa::empty<TestType>(output_slice_shape.rfft(), options);

        ng::insert_and_extract_central_slices_3d<"hc2hc">(
            input_slice_fft, input_slice_fft.copy(), input_slice_shape,
            output_slice_fft0, output_slice_fft1, output_slice_shape,
            {}, input_inv_rotation_matrices,
            {}, output_fwd_rotation_matrices,
            {.input_windowed_sinc = {0.01}, .fftfreq_cutoff = 0.8});
        REQUIRE(test::allclose_abs_safe(output_slice_fft0, output_slice_fft1, 5e-5));
    }
}

TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, test rotation", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto input_slice_shape = Shape4<i64>{1, 1, 256, 256};
    constexpr auto output_slice_shape = Shape4<i64>{11, 1, 256, 256};

    constexpr auto input_slice_rotation = Mat33<f32>::eye(1);
    const auto output_fwd_rotation_matrices = Array<Mat33<f32>>(output_slice_shape[0]);
    for (f32 angle = -20; auto& output_fwd_rotation_matrix: output_fwd_rotation_matrices.span_1d()) {
        output_fwd_rotation_matrix = ng::euler2matrix(
            noa::deg2rad(Vec{angle, 0.f, 0.f}), {.axes = "zyx", .intrinsic = false});
        angle += 4;
    }

    const auto directory = test::NOA_DATA_PATH / "geometry" / "fft";
    const auto output_filename = directory / YAML::LoadFile(directory / "tests.yaml")
                                 ["insert_and_extract_central_slices_3d_others"]
                                 ["test_rotation_filename"]
                                 .as<Path>();

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const Array input_slice_fft = noa::linspace(input_slice_shape.rfft(), noa::Linspace{-100.f, 100.f, true}, options);
        const Array output_slice_fft0 = noa::empty<f32>(output_slice_shape.rfft(), options);

        ng::insert_and_extract_central_slices_3d<"hc2hc">(
            input_slice_fft, {}, input_slice_shape,
            output_slice_fft0, {}, output_slice_shape,
            {}, input_slice_rotation,
            {}, output_fwd_rotation_matrices.to(options),
            {.input_windowed_sinc = {0.002}, .fftfreq_cutoff = 0.498});

        if constexpr (COMPUTE_ASSETS) {
            noa::write(output_slice_fft0, output_filename);
        } else {
            const auto asset = noa::read_data<f32>(output_filename);
            REQUIRE(test::allclose_abs_safe(asset, output_slice_fft0, 1e-4));
        }
    }
}

TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, weight/multiplicity", "[asset]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    constexpr auto input_slice_shape = Shape4<i64>{1, 1, 256, 256};
    constexpr auto output_slice_shape = Shape4<i64>{1, 1, 256, 256};

    Array<Mat33<f32>> input_inv_rotation_matrices(input_slice_shape[0]);
    Array<ng::Quaternion<f32>> input_inv_rotation_quaternion(input_slice_shape[0]);
    for (i64 i{}; i < input_slice_shape[0]; ++i) {
        const auto matrix =  ng::euler2matrix(noa::deg2rad(Vec{45.f, 0.f, 0.f}), {.axes = "zyx"});
        input_inv_rotation_matrices(0, 0, 0, i) = matrix;
        input_inv_rotation_quaternion(0, 0, 0, i) = ng::Quaternion<f32>::from_matrix(matrix).normalize();
    }

    Array<Mat33<f32>> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (auto& matrix: output_fwd_rotation_matrices.span_1d())
        matrix = ng::euler2matrix(noa::deg2rad(Vec{0.f, 0.f, 0.f}), {.axes="zyx"});

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = std::move(input_inv_rotation_matrices).to({device});
        if (input_inv_rotation_quaternion.device() != device)
            input_inv_rotation_quaternion = std::move(input_inv_rotation_quaternion).to({device});
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = std::move(output_fwd_rotation_matrices).to({device});

        const Array input_slice = noa::random(noa::Uniform{-50.f, 50.f}, input_slice_shape, options);
        const Array input_slice_rfft = noa::fft::remap("h2hc", noa::fft::r2c(input_slice), input_slice_shape);
        const Array output_slice_fft0 = noa::empty<c32>(output_slice_shape.rfft(), options);
        const Array output_slice_fft1 = noa::empty<c32>(output_slice_shape.rfft(), options);

        ng::insert_and_extract_central_slices_3d<"hc2hc">(
            input_slice_rfft, {}, input_slice_shape,
            output_slice_fft0, {}, output_slice_shape,
            {}, input_inv_rotation_matrices,
            {}, output_fwd_rotation_matrices,
            {.input_windowed_sinc = {0.01}, .fftfreq_cutoff = 0.8});
        ng::insert_and_extract_central_slices_3d<"hc2hc">(
            input_slice_rfft, {}, input_slice_shape,
            output_slice_fft1, {}, output_slice_shape,
            {}, input_inv_rotation_quaternion,
            {}, output_fwd_rotation_matrices,
            {.input_windowed_sinc = {0.01}, .fftfreq_cutoff = 0.8});

        // Ignore the usual tiny error along the line at Nyquist...
        noa::signal::lowpass<"hc2hc">(output_slice_fft0, output_slice_fft0, input_slice_shape, {.cutoff = 0.49, .width = 0});
        noa::signal::lowpass<"hc2hc">(output_slice_fft1, output_slice_fft1, input_slice_shape, {.cutoff = 0.49, .width = 0});
        REQUIRE(test::allclose_abs_safe(output_slice_fft0, output_slice_fft1, 5e-5));
    }
}

//TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, test recons 0deg", "[.]") {
//    const auto stack_path = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/tilt1_coarse_aligned.mrc");
//    const auto output_directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf");
//
//    auto file = noa::ImageFile(stack_path, noa::READ);
//    const auto n_slices = file.shape()[0];
//    const auto slice_shape = file.shape().set<0>(1);
//    const auto stack_shape = file.shape().set<0>(n_slices - 1);
//
//    // Load every image except the 0deg.
//    auto stack = noa::empty<f32>(stack_shape);
//    i64 index{0};
//    f32 angle = -57;
//    std::vector<f32> tilts;
//    for (i64 i = 0; i < n_slices; ++i, angle += 3) {
//        if (std::abs(angle) < 1e-6f)
//            continue;
//        file.read_slice(stack.view().subregion(index), i);
//        tilts.push_back(angle);
//        ++index;
//    }
//
//    // Input slices
//    const auto input_slice_rotation = noa::empty<Mat33<f32>>(stack_shape[0]);
//    for (i64 i = 0; i < input_slice_rotation.ssize(); ++i) {
//        input_slice_rotation(0, 0, 0, i) = ng::euler2matrix(
//                noa::math::deg2rad(Vec3<f32>{0, tilts[static_cast<size_t>(i)], 0}), "zyx", false).inverse();
//    }
//
//    // Output slice
//    const auto output_slice_rotation = ng::euler2matrix(
//            noa::math::deg2rad(Vec3<f32>{0, 0, 0}), "zyx", false);
//
//    auto stack_rfft = noa::fft::r2c(stack);
//    auto output_slice_rfft = noa::empty<c32>(slice_shape.rfft());
//    auto output_slice_rfft_weight = noa::empty<f32>(slice_shape.rfft());
//
//    noa::fft::remap(fft::H2HC, stack_rfft, stack_rfft, stack_shape);
//
//    const auto rotation_center = stack_shape.vec().filter(2,3).as<f32>() / 2;
//    noa::signal::fft::phase_shift_2d<fft::HC2HC>(stack_rfft, stack_rfft, stack_shape, -rotation_center);
//
//    const auto max_output_size = static_cast<f32>(noa::math::min(slice_shape.filter(2, 3)));
//    auto slice_z_radius =  1.f / max_output_size;
//    for (auto i: noa::irange(10)) {
//        ng::insert_and_extract_central_slices_3d<fft::HC2H>(
//                stack_rfft, stack_shape,
//                output_slice_rfft, slice_shape,
//                Float22{}, input_slice_rotation,
//                Float22{}, output_slice_rotation,
//                slice_z_radius);
//        ng::insert_and_extract_central_slices_3d<fft::HC2H>(
//                1.f, stack_shape,
//                output_slice_rfft_weight, slice_shape,
//                Float22{}, input_slice_rotation,
//                Float22{}, output_slice_rotation,
//                slice_z_radius);
//
//        noa::ewise_binary(output_slice_rfft, output_slice_rfft_weight, output_slice_rfft,
//                          [](c32 lhs, f32 rhs) {
//                              if (rhs > 1)
//                                  lhs /= rhs;
//                              return lhs;
//                          });
//        noa::signal::fft::phase_shift_2d<fft::H2H>(output_slice_rfft, output_slice_rfft, slice_shape, rotation_center);
//
//        // Save PS.
//        const auto output_slice_rfft_weight_full = noa::fft::remap(
//                fft::Remap::H2FC, output_slice_rfft_weight, slice_shape);
//        const auto output_slice = noa::fft::c2r(output_slice_rfft, slice_shape);
//
//        const auto suffix = noa::string::format("{:.4f}.mrc", slice_z_radius);
//        noa::save(output_slice_rfft_weight_full,
//                      output_directory / noa::string::format("output_slice_ps_{}", suffix));
//        noa::save(output_slice, output_directory /  noa::string::format("output_slice_{}", suffix));
//
//        slice_z_radius *= 2;
//    }
//}
//
//TEST_CASE("unified::geometry::insert_and_extract_central_slices_3d, test recons volume", "[.]") {
//    const auto stack_path = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/tilt1_coarse_aligned.mrc");
//    const auto output_directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf");
//
//    auto file = noa::ImageFile(stack_path, noa::READ);
//    const auto slice_shape = file.shape().set<0>(1);
//
//    // Load every image except the 0deg.
//    auto slice_0deg = noa::empty<f32>(slice_shape);
//    file.read_slice(slice_0deg, 19);
//
//    // Input slices
//    const auto input_slice_rotation = ng::euler2matrix(
//            noa::math::deg2rad(Vec3<f32>{0, 35, 0}), "zyx", false);
//
//    // 3d rfft volume.
//    const auto volume_shape = slice_shape.set<1>(600);
//    const auto volume_rfft = noa::zeros<c32>(volume_shape.rfft());
//    const auto volume_rfft_weight = noa::like<f32>(volume_rfft);
//
//    // Output slice
//
//    auto slice_0deg_rfft = noa::fft::r2c(slice_0deg);
//    noa::fft::remap(fft::H2HC, slice_0deg_rfft, slice_0deg_rfft, slice_shape);
//
//    const auto rotation_center = slice_shape.vec().filter(2,3).as<f32>() / 2;
//    noa::signal::fft::phase_shift_2d<fft::HC2HC>(slice_0deg_rfft, slice_0deg_rfft, slice_shape, -rotation_center);
//
//    const auto max_output_size = static_cast<f32>(noa::math::min(slice_shape.filter(2, 3)));
//    auto slice_z_radius = 0.015f; // 1.f / max_output_size;
//    for (auto i: noa::irange(1)) {
//        noa::fill(volume_rfft, c32{0});
//        noa::fill(volume_rfft_weight, f32{0});
//
//        ng::insert_central_slices_3d<fft::HC2H>(
//                slice_0deg_rfft, slice_shape,
//                volume_rfft, volume_shape,
//                Float22{}, input_slice_rotation,
//                slice_z_radius);
//        ng::insert_central_slices_3d<fft::HC2H>(
//                1.f, slice_shape,
//                volume_rfft_weight, volume_shape,
//                Float22{}, input_slice_rotation,
//                slice_z_radius);
//
////        noa::ewise_binary(volume_rfft, volume_rfft_weight, volume_rfft,
////                          [](c32 lhs, f32 rhs) {
////                              if (rhs > 1)
////                                  lhs /= rhs;
////                              return lhs;
////                          });
//        noa::signal::fft::phase_shift_2d<fft::H2H>(volume_rfft, volume_rfft, volume_shape, rotation_center);
//
//        // Save PS.
//        const auto volume_weight_full = noa::fft::remap(
//                fft::Remap::H2FC, volume_rfft_weight, volume_shape);
//        const auto volume = noa::fft::c2r(volume_rfft, volume_shape);
//        ng::fourier_gridding_correction(volume, volume, true);
//
//        const auto suffix = noa::string::format("{:.4f}.mrc", slice_z_radius);
//        noa::save(volume_weight_full,
//                      output_directory / noa::string::format("output_volume_weight_{}", suffix));
//        noa::save(volume, output_directory /  noa::string::format("output_volume_{}", suffix));
//
//        slice_z_radius *= 2;
//    }
//}
