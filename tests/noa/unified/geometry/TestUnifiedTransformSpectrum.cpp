#include <noa/core/geometry/Transform.hpp>
#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/geometry/TransformSpectrum.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/Texture.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/signal/PhaseShift.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace ng = noa::geometry;
namespace ns = noa::signal;
using Interp = noa::Interp;

// TODO Surprisingly this is quite low precision. The images look good, but the errors can get quite
//      large. Of course, we are measuring a round of r2c, phase-shift x2 and transformation... Instead,
//      we should measure the transformation step and only the transformation step. Also, there's still
//      this bug in the library at Nyquist, but here we don't even measure that because of the 0.45 cutoff.

TEST_CASE("unified::geometry::transform_spectrum_2d, vs scipy", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_spectrum_2d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{};
    for (size_t nb{}; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto scale = test["scale"].as<Vec<f64, 2>>();
        const auto rotate = noa::deg2rad(test["rotate"].as<f64>());
        const auto center = test["center"].as<Vec<f32, 2>>();
        const auto shift = test["shift"].as<Vec<f32, 2>>();
        const auto cutoff = test["cutoff"].as<f64>();
        const auto interp = test["interp"].as<Interp>();
        constexpr auto FFT_NORM = noa::fft::Norm::ORTHO;

        const auto matrix = (ng::rotate(rotate) * ng::scale(1 / scale)).inverse().as<f32>();

        for (auto& device: devices) {
            auto stream = noa::StreamGuard(device, Stream::SYNC);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = true}, options).data;
            const auto expected = noa::read_image<f32>(expected_filename, {.enforce_2d_stack = true}, options).data;
            const auto output = noa::like(expected);
            const auto input_fft = noa::empty<c32>(input.shape().rfft(), options);
            const auto input_fft_centered = noa::like(input_fft);
            const auto output_fft = noa::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, {.norm=FFT_NORM});
            ns::phase_shift_2d<"H2HC">(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            ng::transform_spectrum_2d<"HC2H">(
                input_fft_centered, output_fft, input.shape(),
                matrix, center + shift, {interp, cutoff});

            noa::fft::c2r(output_fft, output, {.norm=FFT_NORM});
            REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));

            ++count;

            // With textures:
            noa::fill(output, 0); // erase
            const auto input_fft_centered_texture = Texture<c32>(input_fft_centered, device, interp);
            ng::transform_spectrum_2d<"HC2H">(
                input_fft_centered_texture, output_fft, input.shape(),
                matrix, center + shift, {.fftfreq_cutoff = cutoff});
            noa::fft::c2r(output_fft, output, {.norm=FFT_NORM});
            REQUIRE(test::allclose_abs_safe(expected, output, 1e-4f));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::transform_spectrum_3d, vs scipy", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_spectrum_3d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{};
    for (size_t nb{}; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto scale = test["scale"].as<Vec<f64, 3>>();
        const auto rotate = noa::deg2rad(test["rotate"].as<Vec<f64, 3>>());
        const auto center = test["center"].as<Vec<f32, 3>>();
        const auto shift = test["shift"].as<Vec<f32, 3>>();
        const auto cutoff = test["cutoff"].as<f64>();
        const auto interp = test["interp"].as<Interp>();
        constexpr auto FFT_NORM = noa::fft::Norm::ORTHO;

        const auto matrix = (ng::euler2matrix(rotate) * ng::scale(1 / scale)).inverse().as<f32>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::read_image<f32>(input_filename, {}, options).data;
            const auto expected = noa::read_image<f32>(expected_filename, {}, options).data;
            const auto output = noa::like(expected);
            const auto input_fft = noa::empty<c32>(input.shape().rfft(), options);
            const auto input_fft_centered = noa::like(input_fft);
            const auto output_fft = noa::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, {.norm=FFT_NORM});
            ns::phase_shift_3d<"H2HC">(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            ng::transform_spectrum_3d<"HC2H">(
                input_fft_centered, output_fft, input.shape(),
                matrix, center + shift, {interp, cutoff});
            noa::fft::c2r(output_fft, output, {.norm=FFT_NORM});
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-3f)); // it was MATCH_ABS at 2e-4

            ++count;

            // With textures:
            noa::fill(output, 0); // erase
            const auto input_fft_centered_texture = Texture<c32>(input_fft_centered, device, interp);
            ng::transform_spectrum_3d<"HC2H">(
                input_fft_centered_texture, output_fft, input.shape(),
                matrix, center + shift, {.fftfreq_cutoff = cutoff});
            noa::fft::c2r(output_fft, output, {.norm=FFT_NORM});
            REQUIRE(test::allclose_abs_safe(expected, output, 5e-3f));
        }
    }
    REQUIRE(count == expected_count);
}

TEMPLATE_TEST_CASE("unified::geometry::transform_spectrum_2d(), remap", "", f32, f64) {
    const auto shape = test::random_shape_batched(2);
    constexpr f64 cutoff = 0.5;
    constexpr auto interp = Interp::LINEAR;

    test::Randomizer<f64> randomizer(-3, 3);
    auto transforms = Array<Mat22<f32>>(shape[0]);
    auto shifts = Array<Vec<f32, 2>>(shape[0]);

    for (auto& transform: transforms.span_1d()) {
        const auto scale = Vec{0.9, 1.1};
        const f64 angle = randomizer.get();
        transform = (ng::rotate(angle) * ng::scale(1 / scale)).inverse().as<f32>();
    }
    for (auto& shift: shifts.span_1d())
        shift = Vec{randomizer.get() * 10, randomizer.get() * 10}.as<f32>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (device != transforms.device()) {
            transforms = transforms.to({device});
            shifts = shifts.to({device});
        }

        const auto input = noa::random(noa::Uniform<TestType>{-3, 3}, shape, options);
        const auto input_rfft = noa::fft::r2c(input);

        using complex_t = Complex<TestType>;
        const auto output_rfft = noa::empty<complex_t>(shape.rfft(), options);
        ng::transform_spectrum_2d<"HC2H">(
            input_rfft, output_rfft, shape, transforms, shifts, {interp, cutoff});

        if constexpr (std::is_same_v<TestType, f64>) {
            const auto output_rfft_centered = noa::like(output_rfft);
            ng::transform_spectrum_2d<"HC2HC">(
                input_rfft, output_rfft_centered, shape,
                transforms, shifts, {interp, cutoff});
            const auto output_rfft_final = noa::fft::remap("HC2H", output_rfft_centered, shape);
            REQUIRE(test::allclose_abs(output_rfft, output_rfft_final, 1e-7));
        } else {
            const auto input_rfft_texture = Texture<c32>(input_rfft, device, interp);
            const auto output_rfft_centered = noa::like(output_rfft);
            ng::transform_spectrum_2d<"HC2HC">(
                input_rfft, output_rfft_centered, shape,
                transforms, shifts, {interp, cutoff});
            const auto output_rfft_final = noa::fft::remap("HC2H", output_rfft_centered, shape);
            REQUIRE(test::allclose_abs(output_rfft, output_rfft_final, 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::transform_spectrum_3d(), remap", "", f32, f64) {
    const auto shape = test::random_shape(3);
    constexpr f64 cutoff = 0.5;
    constexpr auto interp = Interp::LINEAR;

    test::Randomizer<f64> randomizer(-3, 3);
    auto transforms = Array<Mat33<f32>>(shape[0]);
    auto shifts = Array<Vec<f32, 3>>(shape[0]);

    for (auto& transform: transforms.span_1d()) {
        const auto scale = Vec{0.9, 1.1, 0.85};
        const auto angles = Vec{randomizer.get(), randomizer.get(), randomizer.get()};
        transform = (ng::euler2matrix(angles) * ng::scale(1 / scale)).inverse().as<f32>();
    }
    for (auto& shift: shifts.span_1d()) {
        shift = Vec{
            randomizer.get() * 10,
            randomizer.get() * 10,
            randomizer.get() * 10,
        }.as<f32>();
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (device != transforms.device()) {
            transforms = transforms.to({device});
            shifts = shifts.to({device});
        }

        const auto input = noa::random(noa::Uniform<TestType>{-3, 3}, shape, options);
        const auto input_fft = noa::fft::r2c(input);

        using complex_t = Complex<TestType>;
        const auto output_fft = noa::empty<complex_t>(shape.rfft(), options);
        ng::transform_spectrum_3d<"HC2H">(
            input_fft, output_fft, shape, transforms, shifts, {interp, cutoff});

        if constexpr (std::is_same_v<TestType, f64>) {
            const auto output_fft_centered = noa::like(output_fft);
            ng::transform_spectrum_3d<"HC2HC">(
                input_fft, output_fft_centered, shape,
                transforms, shifts, {interp, cutoff});
            const auto output_fft_final = noa::fft::remap("HC2H", output_fft_centered, shape);
            REQUIRE(test::allclose_abs(output_fft, output_fft_final, 1e-7));
        } else {
            const auto input_fft_texture = noa::Texture<c32>(input_fft, device, interp);
            const auto output_fft_centered = noa::like(output_fft);
            ng::transform_spectrum_3d<"HC2HC">(
                input_fft, output_fft_centered, shape,
                transforms, shifts, {interp, cutoff});
            const auto output_fft_final = noa::fft::remap("HC2H", output_fft_centered, shape);
            REQUIRE(test::allclose_abs(output_fft, output_fft_final, 1e-7));
        }
    }
}

// // The hermitian symmetry isn't broken by transform_2d and transform_3d.
// TEST_CASE("unified::geometry::transform_spectrum_2d, check redundancy", "[.]") {
//     const auto shape = Shape<i64, 4>{1, 1, 128, 128};
//     const Path output_path = test::NOA_DATA_PATH / "geometry" / "fft";
//     const auto option = ArrayOption("gpu", Allocator::MANAGED);
//
//     Array input = noa::linspace(shape, noa::Linspace<f32>{-10, 10, true}, option);
//     Array output0 = noa::fft::r2c(input);
//     noa::fft::remap("H2HC", output0, output0, shape);
//
//     Array output1 = noa::like(output0);
//     const Mat22 rotation = ng::rotate(noa::deg2rad(45.f));
//     ng::transform_spectrum_2d<"HC2HC">(output0, output1, shape, rotation);
//     noa::write_image(noa::real(output1), output_path / "test_output1_real.mrc");
//     noa::write_image(noa::imag(output1), output_path / "test_output1_imag.mrc");
// }
//
// TEST_CASE("unified::geometry::transform_spectrum_3d, check redundancy", "[.]") {
//     const auto shape = Shape<i64, 4>{1, 128, 128, 128};
//     const Path output_path = test::NOA_DATA_PATH / "geometry" / "fft";
//     const auto option = ArrayOption(Device("gpu"), Allocator::MANAGED);
//
//     Array input = noa::linspace(shape, noa::Linspace<f32>{-10, 10, true}, option);
//     Array output0 = noa::fft::r2c(input);
//     noa::fft::remap(noa::Remap::H2HC, output0, output0, shape);
//
//     Array output1 = noa::like(output0);
//     const Mat33 rotation = ng::euler2matrix(noa::deg2rad(Vec{45.f, 0.f, 0.f}), {.axes="zyx", .intrinsic = false});
//     ng::transform_spectrum_3d<"HC2HC">(output0, output1, shape, rotation, Vec<f32, 3>{});
//     noa::write_image(noa::real(output1), output_path / "test_output1_real.mrc");
//     noa::write_image(noa::imag(output1), output_path / "test_output1_imag.mrc");
// }
//
// TEST_CASE("unified::geometry::transform_spectrum_2d, 0", ".") {
//     constexpr auto shape = Shape<i64, 4>{1, 1, 256, 256};
//
//     auto data = noa::empty<f32>(shape);
//     ng::draw_shape({}, data, ng::Rectangle{
//                        .center = Vec{128., 128.},
//                        .radius = Vec{32., 44.},
//                        .smoothness = 80.,
//                    });
//     auto input_rfft = noa::fft::remap("FC2H", data, shape);
//
//     i64 batch = 5;
//     auto rotations = noa::empty<Mat22<f32>>(batch);
//     for (f32 i{5}; auto& rotation: rotations.span_1d_contiguous()) {
//         rotation = ng::rotate(noa::deg2rad(i));
//         i += 10;
//     }
//
//     auto output_rfft = noa::empty<f32>(shape.set<0>(batch).rfft());
//     ng::transform_spectrum_2d<"H2HC">(
//         input_rfft, output_rfft, shape,
//         rotations, {}, {.interp = Interp::CUBIC_FAST, .fftfreq_cutoff = 1});
//
//     noa::write_image(input_rfft, test::NOA_DATA_PATH / "geometry" / "test_input_rfft.mrc");
//     noa::write_image(output_rfft, test::NOA_DATA_PATH / "geometry" / "test_output_rfft.mrc");
// }
//
// TEST_CASE("unified::geometry::transform_spectrum_2d, 1", ".") {
//     constexpr auto shape = Shape<i64, 4>{5, 1, 256, 256};
//     constexpr auto center = Vec{128., 128.};
//
//     auto input = noa::empty<f32>(shape.set<0>(1));
//     ng::draw_shape({}, input, ng::Rectangle{
//         .center = center,
//         .radius = Vec{64., 64.},
//         .smoothness = 2.,
//     });
//     noa::write_image(input, test::NOA_DATA_PATH / "geometry" / "test_input.mrc");
//
//     auto rotations = noa::empty<Mat<f64, 2, 2>>(shape[0]);
//     for (f64 i{5}; auto& rotation: rotations.span_1d_contiguous()) {
//         rotation = ng::rotate(noa::deg2rad(i));
//         i += 10;
//     }
//
//     for (auto& interp: std::array{Interp::LINEAR, Interp::CUBIC, Interp::LANCZOS4, Interp::LANCZOS6, Interp::LANCZOS8}) {
//         auto input_rfft = noa::fft::r2c(input);
//         ns::phase_shift_2d<"h2h">(input_rfft, input_rfft, shape.set<0>(1), -center);
//
//         auto output_rfft = noa::empty<c32>(shape.rfft());
//         ng::transform_spectrum_2d<"h2h">(
//             input_rfft, output_rfft, shape, rotations, center, {
//                 .interp = interp,
//                 .fftfreq_cutoff = 0.5,
//             });
//
//         auto output = noa::fft::c2r(output_rfft, shape);
//         noa::write_image(output, test::NOA_DATA_PATH / "geometry" / fmt::format("test_output_{}.mrc", interp));
//     }
// }
