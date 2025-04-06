#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/signal/Bandpass.hpp>
#include <noa/unified/Ewise.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;
using Remap = noa::Remap;

TEST_CASE("unified::signal::lowpass()", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["lowpass"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<f64>();
        const auto width = test["width"].as<f64>();
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::empty<f32>(no_batch_shape.rfft());
            noa::signal::lowpass<"h2h">({}, filter_expected, no_batch_shape, {cutoff, width});
            noa::io::write(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so broadcast batch dimension.
        auto filter_expected = noa::indexing::broadcast(noa::io::read_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const Array filter_result = noa::empty<f32>(shape.rfft(), options);
            noa::signal::lowpass<"h2h">({}, filter_result, shape, {cutoff, width});
            noa::io::write(filter_result, path_base / "test.mrc");
            REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const Array input = noa::random(noa::Uniform<f32>{-5, 5}, shape.rfft(), options);
            const Array expected = noa::like(input);
            noa::ewise(noa::wrap(input, filter_expected), expected, noa::Multiply{});

            const Array result = input.copy();
            noa::signal::lowpass<"h2h">(result, result, shape, {cutoff, width});

            REQUIRE(test::allclose_abs(expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::lowpass(), remap", "", f16, f32, f64) {
    const auto shape = test::random_shape_batched(3);
    constexpr f64 cutoff = 0.4;
    constexpr f64 width = 0.1;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::like(filter_expected);
        const auto filter_remapped = noa::like(filter_expected);

        // H2HC
        noa::signal::lowpass<"h2h">({}, filter_expected, shape, {cutoff, width});
        noa::signal::lowpass<"h2hc">({}, filter_result, shape, {cutoff, width});
        noa::fft::remap("hc2h", filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::lowpass<"h2h">({}, filter_expected, shape, {cutoff, width});
        noa::signal::lowpass<"hc2hc">({}, filter_result, shape, {cutoff, width});
        noa::fft::remap("hc2h", filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::lowpass<"h2h">({}, filter_expected, shape, {cutoff, width});
        noa::signal::lowpass<"hc2h">({}, filter_result, shape, {cutoff, width});
        REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));
    }
}

TEST_CASE("unified::signal::highpass()", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["highpass"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<f64>();
        const auto width = test["width"].as<f64>();
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::empty<f32>(no_batch_shape.rfft());
            noa::signal::highpass<"h2h">({}, filter_expected, no_batch_shape, {cutoff, width});
            noa::io::write(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so copy to all batches.
        auto filter_expected = noa::indexing::broadcast(noa::io::read_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const auto filter_result = noa::empty<f32>(shape.rfft(), options);
            noa::signal::highpass<"h2h">({}, filter_result, shape, {cutoff, width});
            REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const auto input = noa::random(noa::Uniform<f32>{-5, 5}, shape.rfft(), options);
            const auto expected = noa::like(input);
            noa::ewise(noa::wrap(input, filter_expected), expected, noa::Multiply{});

            const auto result = input.copy();
            noa::signal::highpass<"h2h">(result, result, shape, {cutoff, width});

            REQUIRE(test::allclose_abs(expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::highpass(), remap", "", f16, f32, f64) {
    const auto shape = test::random_shape_batched(3);
    constexpr f64 cutoff = 0.4;
    constexpr f64 width = 0.1;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::like(filter_expected);
        const auto filter_remapped = noa::like(filter_expected);

        // H2HC
        noa::signal::highpass<Remap::H2H>({}, filter_expected, shape, {cutoff, width});
        noa::signal::highpass<Remap::H2HC>({}, filter_result, shape, {cutoff, width});
        noa::fft::remap(Remap::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::highpass<Remap::H2H>({}, filter_expected, shape, {cutoff, width});
        noa::signal::highpass<Remap::HC2HC>({}, filter_result, shape, {cutoff, width});
        noa::fft::remap(Remap::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::highpass<Remap::H2H>({}, filter_expected, shape, {cutoff, width});
        noa::signal::highpass<Remap::HC2H>({}, filter_result, shape, {cutoff, width});
        REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));
    }
}

TEST_CASE("unified::signal::bandpass()", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["bandpass"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<std::vector<f64>>();
        const auto width = test["width"].as<std::vector<f64>>();
        const auto bandpass_options = noa::signal::Bandpass{cutoff[0], width[0], cutoff[1], width[1]};
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::empty<f32>(no_batch_shape.rfft());
            noa::signal::bandpass<"h2h">({}, filter_expected, no_batch_shape, bandpass_options);
            noa::io::write(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so copy to all batches.
        auto filter_expected = noa::indexing::broadcast(noa::io::read_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const auto filter_result = noa::empty<f32>(shape.rfft(), options);
            noa::signal::bandpass<"h2h">({}, filter_result, shape, bandpass_options);
            REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const auto input = noa::random(noa::Uniform<f32>{-5, 5}, shape.rfft(), options);
            const auto expected = noa::like(input);
            noa::ewise(noa::wrap(input, filter_expected), expected, noa::Multiply{});

            const auto result = input.copy();
            noa::signal::bandpass<"h2h">(result, result, shape, bandpass_options);

            REQUIRE(test::allclose_abs(expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::bandpass(), remap", "", f16, f32, f64) {
    const auto shape = test::random_shape_batched(3);
    constexpr auto bandpass = noa::signal::Bandpass{
        .highpass_cutoff=0.1,
        .highpass_width=0.1,
        .lowpass_cutoff=0.4,
        .lowpass_width=0.1,
    };

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::like(filter_expected);
        const auto filter_remapped = noa::like(filter_expected);

        // H2HC
        noa::signal::bandpass<Remap::H2H>({}, filter_expected, shape, bandpass);
        noa::signal::bandpass<Remap::H2HC>({}, filter_result, shape, bandpass);
        noa::fft::remap(Remap::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::bandpass<Remap::H2H>({}, filter_expected, shape, bandpass);
        noa::signal::bandpass<Remap::HC2HC>({}, filter_result, shape, bandpass);
        noa::fft::remap(Remap::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::allclose_abs(filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::bandpass<Remap::H2H>({}, filter_expected, shape, bandpass);
        noa::signal::bandpass<Remap::HC2H>({}, filter_result, shape, bandpass);
        REQUIRE(test::allclose_abs(filter_expected, filter_result, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::signal::bandpass(), cpu vs gpu", "", f32, f64) {
    if (not Device::is_any_gpu())
        return;

    const auto shape = test::random_shape_batched(3);
    constexpr f64 cutoff = 0.4;
    constexpr f64 width = 0.1;

    const auto cpu_output = noa::empty<TestType>(shape.rfft());
    const auto gpu_output = noa::empty<TestType>(shape.rfft(), {.device="gpu", .allocator="pitched"});

    noa::signal::lowpass<"h2h">({}, cpu_output, shape, {cutoff, width});
    noa::signal::lowpass<"h2h">({}, gpu_output, shape, {cutoff, width});
    REQUIRE(test::allclose_abs(cpu_output, gpu_output.to_cpu(), 5e-6));

    noa::signal::highpass<"h2h">({}, cpu_output, shape, {cutoff, width});
    noa::signal::highpass<"h2h">({}, gpu_output, shape, {cutoff, width});
    REQUIRE(test::allclose_abs(cpu_output, gpu_output.to_cpu(), 5e-6));

    noa::signal::bandpass<"h2h">({}, cpu_output, shape, {0.1, 0.1, 0.45, 0.05});
    noa::signal::bandpass<"h2h">({}, gpu_output, shape, {0.1, 0.1, 0.45, 0.05});
    REQUIRE(test::allclose_abs(cpu_output, gpu_output.to_cpu(), 5e-6));
}
