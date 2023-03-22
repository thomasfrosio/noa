#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/signal/fft/Standardize.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("cpu::signal::fft::standardize_ifft(), half", "[noa][unified]") {
    const auto shape = Shape4<i64>{1, 1, 128, 128};

    fft::Norm norm = GENERATE(fft::Norm::FORWARD, fft::Norm::BACKWARD, fft::Norm::ORTHO, fft::Norm::NONE);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::math::random<f32>(noa::math::normal_t{}, shape, 2.4f, 4.1f, options);
        const auto input_fft = noa::fft::r2c(input, norm);

        const auto input_fft_centered = noa::memory::like(input_fft);
        noa::fft::remap(fft::H2HC, input_fft, input_fft_centered, shape);

        noa::signal::fft::standardize_ifft<fft::HC2HC>(input_fft_centered, input_fft_centered, shape, norm);

        noa::fft::remap(fft::HC2H, input_fft_centered, input_fft, shape);
        noa::fft::c2r(input_fft, input, norm);
        if (norm == fft::Norm::NONE)
            noa::ewise_binary(input, 1 / static_cast<f32>(shape.elements()), input, noa::multiply_t{});

        const auto mean = noa::math::mean(input);
        const auto std = noa::math::std(input, 0);
        REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
    }
}

TEST_CASE("cpu::signal::fft::standardize_ifft(), full", "[noa][cpu]") {
    const auto shape = Shape4<i64>{1, 1, 128, 128};

    fft::Norm norm = GENERATE(fft::Norm::FORWARD, fft::Norm::BACKWARD, fft::Norm::ORTHO, fft::Norm::NONE);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::math::random<f32>(noa::math::normal_t{}, shape, 2.4f, 4.1f, options);
        const auto input_fft = noa::fft::r2c(input, norm);

        const auto input_full_centered = noa::memory::empty<c32>(shape, options);
        noa::fft::remap(fft::H2FC, input_fft, input_full_centered, shape);

        noa::signal::fft::standardize_ifft<fft::FC2FC>(input_full_centered, input_full_centered, shape, norm);

        noa::fft::remap(fft::FC2H, input_full_centered, input_fft, shape);
        noa::fft::c2r(input_fft, input, norm);
        if (norm == fft::Norm::NONE)
            noa::ewise_binary(input, 1 / static_cast<f32>(shape.elements()), input, noa::multiply_t{});

        const auto mean = noa::math::mean(input);
        const auto std = noa::math::std(input, 0);
        REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-5));
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::standardize_ifft()", "[noa][unified]", f32, f64) {
    using real_t = TestType;

    const fft::Norm norm = GENERATE(fft::Norm::FORWARD, fft::Norm::BACKWARD, fft::Norm::ORTHO, fft::Norm::NONE);
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4(ndim);
    INFO(shape);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array image = noa::math::random<real_t>(noa::math::normal_t{}, shape, 2.5, 5.2, options);
        Array image_fft = noa::fft::r2c(image, norm);
        noa::signal::fft::standardize_ifft<fft::H2H>(image_fft, image_fft, shape, norm);
        noa::fft::c2r(image_fft, image, norm);

        if (norm == noa::fft::Norm::NONE)
            noa::ewise_binary(image, 1 / static_cast<real_t>(shape.elements()), image, noa::multiply_t{});

        const real_t mean = noa::math::mean(image);
        const real_t std = noa::math::std(image);
        REQUIRE_THAT(mean, Catch::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::WithinAbs(1, 1e-4));
    }
}
