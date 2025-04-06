#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/signal/Standardize.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
using Norm = noa::fft::Norm;
using Remap = noa::Remap;

TEST_CASE("unified::signal::standardize_ifft(), rfft") {
    constexpr auto shape = Shape4<i64>{1, 1, 128, 128};

    Norm norm = GENERATE(Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random(noa::Normal{2.4f, 4.1f}, shape, options);
        const auto input_fft = noa::fft::r2c(input, {.norm=norm});

        const auto input_fft_centered = noa::like(input_fft);
        noa::fft::remap(Remap::H2HC, input_fft, input_fft_centered, shape);

        noa::signal::standardize_ifft<Remap::HC2HC>(input_fft_centered, input_fft_centered, shape, norm);

        noa::fft::remap(Remap::HC2H, input_fft_centered, input_fft, shape);
        noa::fft::c2r(input_fft, input, {.norm=norm});
        if (norm == Norm::NONE)
            noa::ewise(noa::wrap(input, 1 / static_cast<f32>(shape.n_elements())), input, noa::Multiply{});

        const auto mean = noa::mean(input);
        const auto std = noa::stddev(input, 0);
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::Matchers::WithinAbs(1, 1e-5));
    }
}

TEST_CASE("unified::signal::standardize_ifft(), fft") {
    constexpr auto shape = Shape4<i64>{1, 1, 128, 128};

    Norm norm = GENERATE(Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random(noa::Normal{2.4f, 4.1f}, shape, options);
        const auto input_fft = noa::fft::r2c(input, {.norm=norm});

        const auto input_full_centered = noa::empty<c32>(shape, options);
        noa::fft::remap(Remap::H2FC, input_fft, input_full_centered, shape);

        noa::signal::standardize_ifft<Remap::FC2FC>(input_full_centered, input_full_centered, shape, norm);

        noa::fft::remap(Remap::FC2H, input_full_centered, input_fft, shape);
        noa::fft::c2r(input_fft, input, {.norm=norm});
        if (norm == Norm::NONE)
            noa::ewise(noa::wrap(input, 1 / static_cast<f32>(shape.n_elements())), input, noa::Multiply{});

        const auto mean = noa::mean(input);
        const auto std = noa::stddev(input, 0);
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
        REQUIRE_THAT(std, Catch::Matchers::WithinAbs(1, 1e-5));
    }
}

TEMPLATE_TEST_CASE("unified::signal::standardize_ifft()", "", f32, f64) {
    const Norm norm = GENERATE(Norm::FORWARD, Norm::BACKWARD, Norm::ORTHO, Norm::NONE);
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::random_shape_batched(ndim);
    INFO(shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device, Stream::DEFAULT);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        Array image = noa::random<TestType>(noa::Normal{2.5, 5.2}, shape, options);
        Array image_fft = noa::fft::r2c(image, {.norm=norm});
        noa::signal::standardize_ifft<Remap::H2H>(image_fft, image_fft, shape, norm);
        noa::fft::c2r(image_fft, image, {.norm=norm});

        if (norm == Norm::NONE) {
            const auto n_elements_per_batch = static_cast<TestType>(shape.pop_front().n_elements());
            noa::ewise(noa::wrap(image, 1 / n_elements_per_batch), image, noa::Multiply{});
        }

        for (i64 batch: noa::irange(shape[0])) {
            const auto [mean, stddev] = noa::mean_stddev(image.subregion(batch));
            REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 1e-6));
            REQUIRE_THAT(stddev, Catch::Matchers::WithinAbs(1, 1e-4));
        }
    }
}
