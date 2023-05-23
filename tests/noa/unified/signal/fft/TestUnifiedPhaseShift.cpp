#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/signal/fft/PhaseShift.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace noa;

TEST_CASE("unified::signal::fft::phase_shift_2d()", "[assets][noa][unified]") {
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["2D"];

    std::vector<Device> devices{Device{"cpu"}};
        if (Device::is_any(DeviceType::GPU))
            devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        INFO("test=" << i);
        const auto shape = param["shape"].as<Shape4<i64>>();
        const auto shift = param["shift"].as<Vec2<f32>>();
        const auto cutoff = param["cutoff"].as<f32>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::io::load_data<c32>(path_output, false, options);

            if (path_input.filename().empty()) {
                const auto output = noa::memory::empty<c32>(shape.rfft(), options);
                noa::signal::fft::phase_shift_2d<fft::H2H>({}, output, shape, shift, cutoff);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-4f));

            } else {
                const auto input = noa::io::load_data<c32>(path_input, false, options);
                noa::signal::fft::phase_shift_2d<fft::H2H>(input, input, shape, shift, cutoff);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, input, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::phase_shift_2d(), remap", "[noa][unified]", c32, c64) {
    const auto shape = test::get_random_shape4_batched(2, true); // even sizes for inplace remap
    const auto shift = Vec2<f32>{31.5, -15.2};
    const f32 cutoff = 0.5f;

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape.rfft(), -1, 2, options);

        AND_THEN("h2hc") {
            const auto output = noa::memory::like(input);
            noa::signal::fft::phase_shift_2d<fft::H2H>(input, output, shape, shift, cutoff);
            noa::fft::remap(fft::H2HC, output, output, shape);

            const auto output_remap = noa::memory::like(input);
            noa::signal::fft::phase_shift_2d<fft::H2HC>(input, output_remap, shape, shift, cutoff);

            REQUIRE(test::Matcher(test::MATCH_ABS, output, output_remap, 1e-4f));
        }

        AND_THEN("hc2h") {
            const auto output = noa::memory::like(input);
            noa::signal::fft::phase_shift_2d<fft::H2H>(input, output, shape, shift, cutoff);

            const auto output_remap = noa::memory::like(input);
            noa::fft::remap(fft::H2HC, input, input, shape);
            noa::signal::fft::phase_shift_2d<fft::HC2H>(input, output_remap, shape, shift, cutoff);

            REQUIRE(test::Matcher(test::MATCH_ABS, output, output_remap, 1e-4f));
        }
    }
}

TEST_CASE("unified::signal::fft::phase_shift_3d()", "[assets][noa][unified]") {
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["3D"];

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<Shape4<i64>>();
        const auto shift = param["shift"].as<Vec3<f32>>();
        const auto cutoff = param["cutoff"].as<f32>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::io::load_data<c32>(path_output, false, options);

            if (path_input.filename().empty()) {
                const auto output = noa::memory::empty<c32>(shape.rfft(), options);
                noa::signal::fft::phase_shift_3d<fft::H2H>({}, output, shape, shift, cutoff);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-4f));

            } else {
                const auto input = noa::io::load_data<c32>(path_input, false, options);
                noa::signal::fft::phase_shift_3d<fft::H2H>(input, input, shape, shift, cutoff);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, input, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::phase_shift_3d(), remap", "[noa][unified]", c32, c64) {
    const auto shape = test::get_random_shape4_batched(3, true); // even sizes for inplace remap
    const auto shift = Vec3<f32>{31.5, -15.2, 25.8};
    const f32 cutoff = 0.5f;

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape.rfft(), -1, 2, options);

        AND_THEN("h2hc") {
            const auto output = noa::memory::like(input);
            noa::signal::fft::phase_shift_3d<fft::H2H>(input, output, shape, shift, cutoff);
            noa::fft::remap(fft::H2HC, output, output, shape);

            const auto output_remap = noa::memory::like(input);
            noa::signal::fft::phase_shift_3d<fft::H2HC>(input, output_remap, shape, shift, cutoff);

            REQUIRE(test::Matcher(test::MATCH_ABS, output, output_remap, 1e-4f));
        }

        AND_THEN("hc2h") {
            const auto output = noa::memory::like(input);
            noa::signal::fft::phase_shift_3d<fft::H2H>(input, output, shape, shift, cutoff);

            const auto output_remap = noa::memory::like(input);
            noa::fft::remap(fft::H2HC, input, input, shape);
            noa::signal::fft::phase_shift_3d<fft::HC2H>(input, output_remap, shape, shift, cutoff);

            REQUIRE(test::Matcher(test::MATCH_ABS, output, output_remap, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft::phase_shift{2|3}d(), cpu vs gpu", "[noa][unified]", c32, c64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4_batched(ndim);
    const auto shift = Vec3<f32>{31.5, -15.2, -21.1};
    const f32 cutoff = test::Randomizer<f32>(0.2, 0.5).get();
    INFO(shape);
    INFO(shift);
    INFO(cutoff);

    const auto cpu_input = noa::math::random<TestType>(
            noa::math::uniform_t{}, shape.rfft(), -TestType{2, 2}, TestType{2, 2});
    const auto gpu_input = cpu_input.to(ArrayOption(Device("gpu"), Allocator::PITCHED));

    const fft::Remap remap = GENERATE(fft::H2H, fft::H2HC, fft::HC2HC, fft::HC2H);
    const auto cpu_output = noa::memory::like(cpu_input);
    const auto gpu_output = noa::memory::like(gpu_input);

    INFO(remap);
    if (ndim == 2) {
        const auto shift_2d = shift.pop_back();
        switch (remap) {
            case fft::H2H: {
                noa::signal::fft::phase_shift_2d<fft::H2H>(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::fft::phase_shift_2d<fft::H2H>(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case fft::H2HC: {
                noa::signal::fft::phase_shift_2d<fft::H2HC>(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::fft::phase_shift_2d<fft::H2HC>(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case fft::HC2H: {
                noa::signal::fft::phase_shift_2d<fft::HC2H>(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::fft::phase_shift_2d<fft::HC2H>(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case fft::HC2HC: {
                noa::signal::fft::phase_shift_2d<fft::HC2HC>(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::fft::phase_shift_2d<fft::HC2HC>(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            default:
                REQUIRE(false);
        }
    } else {
        switch (remap) {
            case fft::H2H: {
                noa::signal::fft::phase_shift_3d<fft::H2H>(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::fft::phase_shift_3d<fft::H2H>(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case fft::H2HC: {
                noa::signal::fft::phase_shift_3d<fft::H2HC>(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::fft::phase_shift_3d<fft::H2HC>(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case fft::HC2H: {
                noa::signal::fft::phase_shift_3d<fft::HC2H>(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::fft::phase_shift_3d<fft::HC2H>(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case fft::HC2HC: {
                noa::signal::fft::phase_shift_3d<fft::HC2HC>(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::fft::phase_shift_3d<fft::HC2HC>(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            default:
                REQUIRE(false);
        }
    }
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_output, gpu_output.to_cpu(), 8e-5));
}
