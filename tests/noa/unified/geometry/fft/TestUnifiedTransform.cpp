#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/Signal.h>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

// TODO Surprisingly this is quite low precision. The images look good, but the errors can get quite
//      large. Of course, we are measure a round of r2c, phase-shift x2 and transformation... Instead,
//      we should measure the transformation step and only the transformation step. Also, there's still
//      this bug in the library at Nyquist, but here we don't even measure that because of the 0.45 cutoff.

TEST_CASE("unified::geometry::fft::transform2D, vs scipy", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D"];

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<path_t>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto scale = test["scale"].as<double2_t>();
        const auto rotate = math::deg2rad(test["rotate"].as<double>());
        const auto center = test["center"].as<float2_t>();
        const auto shift = test["shift"].as<float2_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        const auto matrix = float22_t(math::inverse(geometry::rotate(rotate) *
                                                    geometry::scale(1 / scale)));

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, true, options);
            const auto expected = noa::io::load<float>(expected_filename, true, options);
            const auto output = noa::memory::like(expected);
            const auto input_fft = noa::memory::empty<cfloat_t>(input.shape().fft(), options);
            const auto input_fft_centered = noa::memory::like(input_fft);
            const auto output_fft = noa::memory::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, noa::fft::NORM_ORTHO);
            noa::signal::fft::shift2D<fft::H2HC>(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            noa::geometry::fft::transform2D<fft::HC2H>(
                    input_fft_centered, output_fft, input.shape(), matrix, center + shift, cutoff, interp);
            noa::fft::c2r(output_fft, output, fft::NORM_ORTHO);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));

            ++count;

            // With textures:
            noa::memory::fill(output, 0.f); // erase
            const auto input_fft_centered_texture = noa::Texture(input_fft_centered, device, INTERP_LINEAR, BORDER_ZERO);
            noa::geometry::fft::transform2D<fft::HC2H>(
                    input_fft_centered_texture, output_fft, input.shape(), matrix, center + shift, cutoff);
            noa::fft::c2r(output_fft, output, fft::NORM_ORTHO);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-4f));
        }
    }
    REQUIRE(count == expected_count);
}

TEST_CASE("unified::geometry::fft::transform3D, vs scipy", "[noa][unified][assets]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D"];

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const size_t expected_count = param["tests"].size() * devices.size();
    REQUIRE(expected_count > 1);
    size_t count{0};
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto input_filename = path_base / test["input"].as<path_t>();
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto scale = test["scale"].as<double3_t>();
        const auto rotate = math::deg2rad(test["rotate"].as<double3_t>());
        const auto center = test["center"].as<float3_t>();
        const auto shift = test["shift"].as<float3_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        const auto matrix = float33_t(math::inverse(geometry::euler2matrix(rotate) *
                                          geometry::scale(1 / scale)));

        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load<float>(input_filename, false, options);
            const auto expected = noa::io::load<float>(expected_filename, false, options);
            const auto output = noa::memory::like(expected);
            const auto input_fft = noa::memory::empty<cfloat_t>(input.shape().fft(), options);
            const auto input_fft_centered = noa::memory::like(input_fft);
            const auto output_fft = noa::memory::like(input_fft_centered);

            noa::fft::r2c(input, input_fft, noa::fft::NORM_ORTHO);
            noa::signal::fft::shift3D<fft::H2HC>(input_fft, input_fft_centered, input.shape(), -center, cutoff);

            // With arrays:
            noa::geometry::fft::transform3D<fft::HC2H>(
                    input_fft_centered, output_fft, input.shape(), matrix, center + shift, cutoff, interp);
            noa::fft::c2r(output_fft, output, fft::NORM_ORTHO);
            noa::io::save(output, path_base / "test_fft_3d.mrc");
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-3f)); // it was MATCH_ABS at 2e-4

            ++count;

            // With textures:
            noa::memory::fill(output, 0.f); // erase
            const auto input_fft_centered_texture = noa::Texture(input_fft_centered, device, interp, BORDER_ZERO);
            noa::geometry::fft::transform3D<fft::HC2H>(
                    input_fft_centered_texture, output_fft, input.shape(), matrix, center + shift, cutoff);
            noa::fft::c2r(output_fft, output, fft::NORM_ORTHO);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 5e-3f));
        }
    }
    REQUIRE(count == expected_count);
}

TEMPLATE_TEST_CASE("unified::geometry::fft::transform2D(), remap", "[noa][unified]", float, double) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;

    test::Randomizer<float> randomizer(-3, 3);
    auto transforms = Array<float22_t>(shape[0]);
    auto shifts = Array<float2_t>(shape[0]);
    for (dim_t batch = 0; batch < shape[0]; ++batch) {
        const auto scale = float2_t{0.9, 1.1};
        const float angle = randomizer.get();
        const auto matrix = float22_t(geometry::rotate(angle) *
                                      geometry::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10,
                         randomizer.get() * 10};
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
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
        noa::geometry::fft::transform2D<fft::HC2H>(input_fft, output_fft, shape, transforms, shifts, cutoff, interp);

        if constexpr (std::is_same_v<TestType, double>) {
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform2D<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        } else {
            const auto input_fft_texture = noa::Texture(input_fft, device, interp, BORDER_ZERO, complex_t{0}, true);
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform2D<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::transform3D(), remap", "[noa][unified]", float, double) {
    const dim4_t shape = test::getRandomShape(3);
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;

    test::Randomizer<float> randomizer(-3, 3);
    auto transforms = Array<float33_t>(shape[0]);
    auto shifts = Array<float3_t>(shape[0]);
    for (dim_t batch = 0; batch < shape[0]; ++batch) {
        const auto scale = float3_t{0.9, 1.1, 0.85};
        const auto angles = float3_t{randomizer.get(), randomizer.get(), randomizer.get()};
        const auto matrix = float33_t(geometry::euler2matrix(angles) *
                                      geometry::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10,
                         randomizer.get() * 10,
                         randomizer.get() * 10};
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
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
        noa::geometry::fft::transform3D<fft::HC2H>(input_fft, output_fft, shape, transforms, shifts, cutoff, interp);

        if constexpr (std::is_same_v<TestType, double>) {
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform3D<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        } else {
            const auto input_fft_texture = noa::Texture(input_fft, device, interp, BORDER_ZERO);
            const auto output_fft_centered = noa::memory::like(output_fft);
            noa::geometry::fft::transform3D<fft::HC2HC>(
                    input_fft, output_fft_centered, shape, transforms, shifts, cutoff, interp);
            const auto output_fft_final = noa::fft::remap(fft::HC2H, output_fft_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, output_fft, output_fft_final, 1e-7));
        }
    }
}

// The hermitian symmetry isn't broken by transform2D and transform3D.
TEST_CASE("unified::geometry::fft::transform2D, check redundancy", "[.]") {
    const dim4_t shape = {1, 1, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("gpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(shape, -10, 10, true, option);
    Array output0 = fft::r2c(input);
    fft::remap(fft::H2HC, output0, output0, shape);

    Array output1 = memory::like(output0);
    const float22_t rotation = geometry::rotate(math::deg2rad(45.f));
    geometry::fft::transform2D<fft::HC2HC>(output0, output1, shape, rotation, float2_t{});
    io::save(math::real(output1), output_path / "test_output1_real.mrc");
    io::save(math::imag(output1), output_path / "test_output1_imag.mrc");
}

TEST_CASE("unified::geometry::fft::transform3D, check redundancy", "[.]") {
    const dim4_t shape = {1, 128, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("gpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(shape, -10, 10, true, option);
    Array output0 = fft::r2c(input);
    fft::remap(fft::H2HC, output0, output0, shape);

    Array output1 = memory::like(output0);
    const float33_t rotation = geometry::euler2matrix(math::deg2rad(float3_t{45.f, 0, 0}), "ZYX", false);
    geometry::fft::transform3D<fft::HC2HC>(output0, output1, shape, rotation, float3_t{});
    io::save(math::real(output1), output_path / "test_output1_real.mrc");
    io::save(math::imag(output1), output_path / "test_output1_imag.mrc");
}
