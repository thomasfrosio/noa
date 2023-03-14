#include <noa/core/geometry/Transform.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/geometry/fft/Transform.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/math/Complex.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/signal/fft/PhaseShift.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

// TODO Surprisingly this is quite low precision. The images look good, but the errors can get quite
//      large. Of course, we are measure a round of r2c, phase-shift x2 and transformation... Instead,
//      we should measure the transformation step and only the transformation step. Also, there's still
//      this bug in the library at Nyquist, but here we don't even measure that because of the 0.45 cutoff.

TEST_CASE("unified::geometry::fft::transform_2d, vs scipy", "[noa][unified][asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_2d"];

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto scale = test["scale"].as<Vec2<f64>>();
        const auto rotate = math::deg2rad(test["rotate"].as<f64>());
        const auto center = test["center"].as<Vec2<f32>>();
        const auto shift = test["shift"].as<Vec2<f32>>();
        const auto cutoff = test["cutoff"].as<f32>();
        const auto interp = test["interp"].as<InterpMode>();
        constexpr auto FFT_NORM = noa::fft::Norm::ORTHO;

        const auto matrix = (geometry::rotate(rotate) * geometry::scale(1 / scale)).inverse().as<f32>();

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, true, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, true, options);
            const auto output = noa::memory::like(expected);
            const auto input_fft = noa::memory::empty<c32>(input.shape().fft(), options);
            const auto input_fft_centered = noa::memory::like(input_fft);
            const auto output_fft = noa::memory::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, FFT_NORM);
            noa::signal::fft::phase_shift_2d<fft::H2HC>(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            noa::geometry::fft::transform_2d<fft::HC2H>(
                    input_fft_centered, output_fft, input.shape(), matrix, center + shift, cutoff, interp);
            noa::fft::c2r(output_fft, output, FFT_NORM);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));

            ++count;

            // With textures:
            noa::memory::fill(output, 0.f); // erase
            const Texture<c32> input_fft_centered_texture(input_fft_centered, device, interp, BorderMode::ZERO);
            noa::geometry::fft::transform_2d<fft::HC2H>(
                    input_fft_centered_texture, output_fft, input.shape(), matrix, center + shift, cutoff);
            noa::fft::c2r(output_fft, output, FFT_NORM);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::fft::transform_3d, vs scipy", "[noa][unified][asset]") {
    const Path path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform_3d"];

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto expected_filename = path_base / test["expected"].as<Path>();
        const auto scale = test["scale"].as<Vec3<f64>>();
        const auto rotate = noa::math::deg2rad(test["rotate"].as<Vec3<f64>>());
        const auto center = test["center"].as<Vec3<f32>>();
        const auto shift = test["shift"].as<Vec3<f32>>();
        const auto cutoff = test["cutoff"].as<f32>();
        const auto interp = test["interp"].as<InterpMode>();
        constexpr auto FFT_NORM = noa::fft::Norm::ORTHO;

        const auto matrix = (noa::geometry::euler2matrix(rotate) * noa::geometry::scale(1 / scale)).inverse().as<f32>();

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, false, options);
            const auto expected = noa::io::load_data<f32>(expected_filename, false, options);
            const auto output = noa::memory::like(expected);
            const auto input_fft = noa::memory::empty<c32>(input.shape().fft(), options);
            const auto input_fft_centered = noa::memory::like(input_fft);
            const auto output_fft = noa::memory::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, FFT_NORM);
            noa::signal::fft::phase_shift_3d<fft::H2HC>(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            noa::geometry::fft::transform_3d<fft::HC2H>(
                    input_fft_centered, output_fft, input.shape(), matrix, center + shift, cutoff, interp);
            noa::fft::c2r(output_fft, output, FFT_NORM);
            noa::io::save(output, path_base / "test_fft_3d.mrc");
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-3f)); // it was MATCH_ABS at 2e-4

            ++count;

            // With textures:
            noa::memory::fill(output, 0.f); // erase
            const noa::Texture<c32> input_fft_centered_texture(input_fft_centered, device, interp, BorderMode::ZERO);
            noa::geometry::fft::transform_3d<fft::HC2H>(
                    input_fft_centered_texture, output_fft, input.shape(), matrix, center + shift, cutoff);
            noa::fft::c2r(output_fft, output, FFT_NORM);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-3f));
        }
    }
    REQUIRE(count == expected_count);
}

TEMPLATE_TEST_CASE("unified::geometry::fft::transform_2d(), remap", "[noa][unified]", f32, f64) {
    const auto shape = test::get_random_shape4_batched(2);
    const float cutoff = 0.5;
    const auto interp = InterpMode::LINEAR;

    test::Randomizer<f32> randomizer(-3, 3);
    auto transforms = Array<Float22>(shape[0]);
    auto shifts = Array<Vec2<f32>>(shape[0]);
    for (i64 batch = 0; batch < shape[0]; ++batch) {
        const auto scale = Vec2<f32>{0.9, 1.1};
        const float angle = randomizer.get();
        transforms(0, 0, 0, batch) = (geometry::rotate(angle) * geometry::scale(1 / scale)).inverse();
        shifts(0, 0, 0, batch) = {randomizer.get() * 10,
                                  randomizer.get() * 10};
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        if (device != transforms.device()) {
            transforms = transforms.to(device);
            shifts = shifts.to(device);
        }

        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -3, 3, options);
        const auto input_fft = noa::fft::r2c(input);

        using complex_t = Complex<TestType>;
        const auto output_fft = noa::memory::empty<complex_t>(shape.fft(), options);
        noa::geometry::fft::transform_2d<fft::HC2H>(input_fft, output_fft, shape, transforms, shifts, cutoff, interp);

        if constexpr (std::is_same_v<TestType, f64>) {
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform_2d<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        } else {
            const auto input_fft_texture = noa::Texture(input_fft, device, interp, BorderMode::ZERO, complex_t{0}, true);
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform_2d<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::transform_3d(), remap", "[noa][unified]", f32, f64) {
    const auto shape = test::get_random_shape4_batched(3);
    const float cutoff = 0.5;
    const auto interp = InterpMode::LINEAR;

    test::Randomizer<f32> randomizer(-3, 3);
    auto transforms = Array<Float33>(shape[0]);
    auto shifts = Array<Vec3<f32>>(shape[0]);
    for (i64 batch = 0; batch < shape[0]; ++batch) {
        const auto scale = Vec3<f32>{0.9, 1.1, 0.85};
        const auto angles = Vec3<f32>{randomizer.get(), randomizer.get(), randomizer.get()};
        transforms(0, 0, 0, batch) = (geometry::euler2matrix(angles) * geometry::scale(1 / scale)).inverse();
        shifts(0, 0, 0, batch) = {randomizer.get() * 10,
                                  randomizer.get() * 10,
                                  randomizer.get() * 10};
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
        INFO(device);

        if (device != transforms.device()) {
            transforms = transforms.to(device);
            shifts = shifts.to(device);
        }

        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -3, 3, options);
        const auto input_fft = noa::fft::r2c(input);

        using complex_t = Complex<TestType>;
        const auto output_fft = noa::memory::empty<complex_t>(shape.fft(), options);
        noa::geometry::fft::transform_3d<fft::HC2H>(input_fft, output_fft, shape, transforms, shifts, cutoff, interp);

        if constexpr (std::is_same_v<TestType, f64>) {
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform_3d<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        } else {
            const auto input_fft_texture = noa::Texture<c32>(input_fft, device, interp, BorderMode::ZERO);
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform_3d<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        }
    }
}

// The hermitian symmetry isn't broken by transform_2d and transform_3d.
TEST_CASE("unified::geometry::fft::transform_2d, check redundancy", "[.]") {
    const auto shape = Shape4<i64>{1, 1, 128, 128};
    const Path output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    const auto option = ArrayOption(Device("gpu"), Allocator::MANAGED);

    Array input = noa::memory::linspace<f32>(shape, -10, 10, true, option);
    Array output0 = noa::fft::r2c(input);
    noa::fft::remap(noa::fft::H2HC, output0, output0, shape);

    Array output1 = noa::memory::like(output0);
    const Float22 rotation = noa::geometry::rotate(noa::math::deg2rad(45.f));
    noa::geometry::fft::transform_2d<fft::HC2HC>(output0, output1, shape, rotation);
    noa::io::save(noa::math::real(output1), output_path / "test_output1_real.mrc");
    noa::io::save(noa::math::imag(output1), output_path / "test_output1_imag.mrc");
}

TEST_CASE("unified::geometry::fft::transform_3d, check redundancy", "[.]") {
    const auto shape = Shape4<i64>{1, 128, 128, 128};
    const Path output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    const auto option = ArrayOption(Device("gpu"), Allocator::MANAGED);

    Array input = noa::memory::linspace<f32>(shape, -10, 10, true, option);
    Array output0 = noa::fft::r2c(input);
    noa::fft::remap(noa::fft::H2HC, output0, output0, shape);

    Array output1 = memory::like(output0);
    const Float33 rotation = noa::geometry::euler2matrix(noa::math::deg2rad(Vec3<f32>{45.f, 0, 0}), "zyx", false);
    noa::geometry::fft::transform_3d<fft::HC2HC>(output0, output1, shape, rotation, Vec3<f32>{});
    noa::io::save(noa::math::real(output1), output_path / "test_output1_real.mrc");
    noa::io::save(noa::math::imag(output1), output_path / "test_output1_imag.mrc");
}
