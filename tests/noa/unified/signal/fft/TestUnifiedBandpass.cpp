#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/signal/fft/Bandpass.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace noa;

TEST_CASE("unified::signal::fft::lowpass()", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["lowpass"];

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<f32>();
        const auto width = test["width"].as<f32>();
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::memory::empty<f32>(no_batch_shape.rfft());
            noa::signal::fft::lowpass<fft::H2H>({}, filter_expected, no_batch_shape, cutoff, width);
            noa::io::save(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so copy to all batches.
        auto filter_expected = noa::indexing::broadcast(noa::io::load_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const auto filter_result = noa::memory::empty<f32>(shape.rfft(), options);
            noa::signal::fft::lowpass<fft::H2H>({}, filter_result, shape, cutoff, width);
            REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape.rfft(), -5, 5, options);
            const auto expected = noa::memory::like(input);
            noa::ewise_binary(filter_expected, input, expected, noa::multiply_t{});

            const auto result = input.copy();
            noa::signal::fft::lowpass<fft::H2H>(result, result, shape, cutoff, width);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::lowpass(), remap", "[noa][unified]", f16, f32, f64) {
    const auto shape = test::get_random_shape4_batched(3);
    const f32 cutoff = 0.4f;
    const f32 width = 0.1f;

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::memory::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::memory::like(filter_expected);
        const auto filter_remapped = noa::memory::like(filter_expected);

        // H2HC
        noa::signal::fft::lowpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::lowpass<fft::H2HC>({}, filter_result, shape, cutoff, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::fft::lowpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::lowpass<fft::HC2HC>({}, filter_result, shape, cutoff, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::fft::lowpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::lowpass<fft::HC2H>({}, filter_result, shape, cutoff, width);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));
    }
}

TEST_CASE("unified::signal::fft::highpass()", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["highpass"];

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<f32>();
        const auto width = test["width"].as<f32>();
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::memory::empty<f32>(no_batch_shape.rfft());
            noa::signal::fft::highpass<fft::H2H>({}, filter_expected, no_batch_shape, cutoff, width);
            noa::io::save(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so copy to all batches.
        auto filter_expected = noa::indexing::broadcast(noa::io::load_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const auto filter_result = noa::memory::empty<f32>(shape.rfft(), options);
            noa::signal::fft::highpass<fft::H2H>({}, filter_result, shape, cutoff, width);
            REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape.rfft(), -5, 5, options);
            const auto expected = noa::memory::like(input);
            noa::ewise_binary(filter_expected, input, expected, noa::multiply_t{});

            const auto result = input.copy();
            noa::signal::fft::highpass<fft::H2H>(result, result, shape, cutoff, width);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::highpass(), remap", "[noa][unified]", f16, f32, f64) {
    const auto shape = test::get_random_shape4_batched(3);
    const f32 cutoff = 0.4f;
    const f32 width = 0.1f;

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::memory::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::memory::like(filter_expected);
        const auto filter_remapped = noa::memory::like(filter_expected);

        // H2HC
        noa::signal::fft::highpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::highpass<fft::H2HC>({}, filter_result, shape, cutoff, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::fft::highpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::highpass<fft::HC2HC>({}, filter_result, shape, cutoff, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::fft::highpass<fft::H2H>({}, filter_expected, shape, cutoff, width);
        noa::signal::fft::highpass<fft::HC2H>({}, filter_result, shape, cutoff, width);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));
    }
}

TEST_CASE("unified::signal::fft::bandpass()", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["bandpass"];

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto cutoff = test["cutoff"].as<std::vector<f32>>();
        const auto width = test["width"].as<std::vector<f32>>();
        const auto filename_expected = path_base / test["path"].as<Path>();

        if (COMPUTE_ASSETS) {
            auto no_batch_shape = shape;
            no_batch_shape[0] = 1;
            const auto filter_expected = noa::memory::empty<f32>(no_batch_shape.rfft());
            noa::signal::fft::bandpass<fft::H2H>(
                    {}, filter_expected, no_batch_shape, cutoff[0], cutoff[1], width[0], width[1]);
            noa::io::save(filter_expected, filename_expected);
            continue;
        }

        // Get expected filter. Asset is not batched so copy to all batches.
        auto filter_expected = noa::indexing::broadcast(noa::io::load_data<f32>(filename_expected), shape.rfft());

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (device != filter_expected.device())
                filter_expected = filter_expected.to(options);

            // Test saving the mask.
            const auto filter_result = noa::memory::empty<f32>(shape.rfft(), options);
            noa::signal::fft::bandpass<fft::H2H>({}, filter_result, shape, cutoff[0], cutoff[1], width[0], width[1]);
            REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));

            // Test on-the-fly, in-place.
            const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape.rfft(), -5, 5, options);
            const auto expected = noa::memory::like(input);
            noa::ewise_binary(filter_expected, input, expected, noa::multiply_t{});

            const auto result = input.copy();
            noa::signal::fft::bandpass<fft::H2H>(result, result, shape, cutoff[0], cutoff[1], width[0], width[1]);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-6));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::bandpass(), remap", "[noa][unified]", f16, f32, f64) {
    const auto shape = test::get_random_shape4_batched(3);
    const f32 cutoff_high = 0.1f, cutoff_low = 0.4f;
    const f32 width = 0.1f;

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto filter_expected = noa::memory::empty<TestType>(shape.rfft(), options);
        const auto filter_result = noa::memory::like(filter_expected);
        const auto filter_remapped = noa::memory::like(filter_expected);

        // H2HC
        noa::signal::fft::bandpass<fft::H2H>({}, filter_expected, shape, cutoff_high, cutoff_low, width, width);
        noa::signal::fft::bandpass<fft::H2HC>({}, filter_result, shape, cutoff_high, cutoff_low, width, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2HC
        noa::signal::fft::bandpass<fft::H2H>({}, filter_expected, shape, cutoff_high, cutoff_low, width, width);
        noa::signal::fft::bandpass<fft::HC2HC>({}, filter_result, shape, cutoff_high, cutoff_low, width, width);
        noa::fft::remap(fft::HC2H, filter_result, filter_remapped, shape);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_remapped, 1e-6));

        // HC2H
        noa::signal::fft::bandpass<fft::H2H>({}, filter_expected, shape, cutoff_high, cutoff_low, width, width);
        noa::signal::fft::bandpass<fft::HC2H>({}, filter_result, shape, cutoff_high, cutoff_low, width, width);
        REQUIRE(test::Matcher(test::MATCH_ABS, filter_expected, filter_result, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::bandpass(), cpu vs gpu", "[noa][unified]", f32, f64) {
    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto shape = test::get_random_shape4_batched(3);
    const f32 cutoff = 0.4f;
    const f32 width = 0.1f;

    const auto cpu_output = noa::memory::empty<TestType>(shape.rfft());
    const auto gpu_output = noa::memory::empty<TestType>(shape.rfft(), ArrayOption(Device("gpu"), Allocator::PITCHED));

    noa::signal::fft::lowpass<fft::H2H>({}, cpu_output, shape, cutoff, width);
    noa::signal::fft::lowpass<fft::H2H>({}, gpu_output, shape, cutoff, width);
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_output, gpu_output.to_cpu(), 5e-6));

    noa::signal::fft::highpass<fft::H2H>({}, cpu_output, shape, cutoff, width);
    noa::signal::fft::highpass<fft::H2H>({}, gpu_output, shape, cutoff, width);
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_output, gpu_output.to_cpu(), 5e-6));

    noa::signal::fft::bandpass<fft::H2H>({}, cpu_output, shape, 0.1f, 0.45f, 0.1f, 0.05f);
    noa::signal::fft::bandpass<fft::H2H>({}, gpu_output, shape, 0.1f, 0.45f, 0.1f, 0.05f);
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_output, gpu_output.to_cpu(), 5e-6));
}
