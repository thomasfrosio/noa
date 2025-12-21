#include <noa/runtime/Factory.hpp>
#include <noa/runtime/Random.hpp>

#include <noa/io/IO.hpp>
#include <noa/fft/Remap.hpp>
#include <noa/signal/PhaseShift.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;
namespace nf = noa::fft;

TEST_CASE("signal::phase_shift_2d()", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["2d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        INFO("test=" << i);
        const auto shape = param["shape"].as<Shape4>();
        const auto shift = param["shift"].as<Vec<f32, 2>>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::read_image<c32>(path_output, {}, options).data;

            if (path_input.filename().empty()) {
                const auto output = noa::empty<c32>(shape.rfft(), options);
                noa::signal::phase_shift_2d<"h2h">({}, output, shape, shift);
                REQUIRE(test::allclose_abs(expected, output, 1e-4f));

            } else {
                const auto input = noa::read_image<c32>(path_input, {}, options).data;
                noa::signal::phase_shift_2d<"h2h">(input, input, shape, shift);
                REQUIRE(test::allclose_abs(expected, input, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_2d(), remap", "", c32, c64) {
    const auto shape = test::random_shape_batched(2, {.batch_range = {2, 5}, .only_even_sizes = true}); // even sizes for inplace remap
    const auto shift = Vec{31.5, -15.2};
    const f64 cutoff = 0.5;

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 2>>(shape[0]);
    for (auto& e: shifts.span_1d())
        e = shift.as<real_t>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (shifts.device() != device)
            shifts = shifts.to({.device=device});

        const auto input = noa::random<TestType>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);

        {
            const auto output = noa::like(input);
            noa::signal::phase_shift_2d<"h2h">(input, output, shape, shifts, cutoff);
            noa::fft::remap("H2HC", output, output, shape);

            const auto output_remap = noa::like(input);
            noa::signal::phase_shift_2d<"h2hc">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
        {
            const auto output = noa::like(input);
            noa::signal::phase_shift_2d<"h2h">(input, output, shape, shifts, cutoff);

            const auto output_remap = noa::like(input);
            noa::fft::remap("H2HC", input, input, shape);
            noa::signal::phase_shift_2d<"hc2h">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_2d(), batch", "", c32, c64) {
    const auto shape = test::random_shape_batched(2);

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 2>>(shape[0]);
    auto randomizer = test::Randomizer<real_t>(-30, 30);
    for (auto& e: shifts.span_1d())
        e = {randomizer.get(), randomizer.get()};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random<TestType>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);

        const auto output0 = noa::like(input);
        for (i64 i{}; auto shift: shifts.span_1d()) {
            noa::signal::phase_shift_2d<"h2h">(input.subregion(i), output0.subregion(i), shape.set<0>(1), shift);
            ++i;
        }

        const auto output1 = noa::like(input);
        noa::signal::phase_shift_2d<"h2h">(input, output1, shape, shifts.to({device}));
        REQUIRE(test::allclose_abs(output0, output1, 1e-6f));
    }
}

TEST_CASE("signal::phase_shift_3d()", "[asset]") {
    const Path path_base = test::NOA_DATA_PATH / "signal" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["3d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<Shape4>();
        const auto shift = param["shift"].as<Vec<f32, 3>>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::read_image<c32>(path_output, {}, options).data;

            if (path_input.filename().empty()) {
                const auto output = noa::empty<c32>(shape.rfft(), options);
                noa::signal::phase_shift_3d<"h2h">({}, output, shape, shift);
                REQUIRE(test::allclose_abs(expected, output, 1e-4f));

            } else {
                const auto input = noa::read_image<c32>(path_input, {}, options).data;
                noa::signal::phase_shift_3d<"h2h">(input, input, shape, shift);
                REQUIRE(test::allclose_abs(expected, input, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_3d(), remap", "", c32, c64) {
    const auto shape = test::random_shape_batched(3, {.only_even_sizes = true}); // even sizes for inplace remap
    const auto shift = Vec{31.5, -15.2, 25.8};
    const f64 cutoff = 0.5;

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 3>>(shape[0]);
    for (auto& e: shifts.span_1d())
        e = shift.as<real_t>();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random<TestType>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);

        AND_THEN("h2hc") {
            const auto output = noa::like(input);
            noa::signal::phase_shift_3d<"h2h">(input, output, shape, shifts, cutoff);
            noa::fft::remap("H2HC", output, output, shape);

            const auto output_remap = noa::like(input);
            noa::signal::phase_shift_3d<"h2hc">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }

        AND_THEN("hc2h") {
            const auto output = noa::like(input);
            noa::signal::phase_shift_3d<"h2h">(input, output, shape, shifts, cutoff);

            const auto output_remap = noa::like(input);
            noa::fft::remap("H2HC", input, input, shape);
            noa::signal::phase_shift_3d<"hc2h">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_3d(), batch", "", c32, c64) {
    const auto shape = test::random_shape_batched(3);

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 3>>(shape[0]);
    auto randomizer = test::Randomizer<real_t>(-30, 30);
    for (auto& e: shifts.span_1d())
        e = {randomizer.get(), randomizer.get(), randomizer.get()};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random<TestType>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);
        const auto output0 = noa::like(input);
        for (i64 i{}; auto shift: shifts.span_1d()) {
            noa::signal::phase_shift_3d<"h2h">(input.subregion(i), output0.subregion(i), shape.set<0>(1), shift);
            ++i;
        }

        const auto output1 = noa::like(input);
        noa::signal::phase_shift_3d<"h2h">(input, output1, shape, shifts.to({device}));
        REQUIRE(test::allclose_abs(output0, output1, 1e-6f));
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift{2|3}d(), cpu vs gpu", "", c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::random_shape_batched(ndim);
    const auto shift = Vec{31.5, -15.2, -21.1};
    const f64 cutoff = test::Randomizer<f64>(0.2, 0.5).get();
    INFO(shape);
    INFO(shift);
    INFO(cutoff);

    const auto cpu_input = noa::random<TestType>(noa::Uniform{-TestType{2, 2}, TestType{2, 2}}, shape.rfft());
    const auto gpu_input = cpu_input.to(ArrayOption(Device("gpu"), Allocator::PITCHED));

    const nf::Layout remap = GENERATE(nf::Layout::H2H, nf::Layout::H2HC, nf::Layout::HC2HC, nf::Layout::HC2H);
    const auto cpu_output = noa::like(cpu_input);
    const auto gpu_output = noa::like(gpu_input);

    INFO(remap);
    if (ndim == 2) {
        const auto shift_2d = shift.pop_back();
        switch (remap) {
            case nf::Layout::H2H: {
                noa::signal::phase_shift_2d<"h2h">(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::phase_shift_2d<"h2h">(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case nf::Layout::H2HC: {
                noa::signal::phase_shift_2d<"h2hc">(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::phase_shift_2d<"h2hc">(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case nf::Layout::HC2H: {
                noa::signal::phase_shift_2d<"hc2h">(cpu_input, cpu_output, shape, shift_2d, cutoff);
                noa::signal::phase_shift_2d<"hc2h">(gpu_input, gpu_output, shape, shift_2d, cutoff);
                break;
            }
            case nf::Layout::HC2HC: {
                noa::signal::phase_shift_2d<"hc2hc">(cpu_input, cpu_output, shape, shift_2d.as<f32>(), cutoff);
                noa::signal::phase_shift_2d<"hc2hc">(gpu_input, gpu_output, shape, shift_2d.as<f32>(), cutoff);
                break;
            }
            default:
                REQUIRE(false);
        }
    } else {
        switch (remap) {
            case nf::Layout::H2H: {
                noa::signal::phase_shift_3d<"h2h">(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::phase_shift_3d<"h2h">(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case nf::Layout::H2HC: {
                noa::signal::phase_shift_3d<"h2hc">(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::phase_shift_3d<"h2hc">(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case nf::Layout::HC2H: {
                noa::signal::phase_shift_3d<"hc2h">(cpu_input, cpu_output, shape, shift, cutoff);
                noa::signal::phase_shift_3d<"hc2h">(gpu_input, gpu_output, shape, shift, cutoff);
                break;
            }
            case nf::Layout::HC2HC: {
                noa::signal::phase_shift_3d<"hc2hc">(cpu_input, cpu_output, shape, shift.as<f32>(), cutoff);
                noa::signal::phase_shift_3d<"hc2hc">(gpu_input, gpu_output, shape, shift.as<f32>(), cutoff);
                break;
            }
            default:
                REQUIRE(false);
        }
    }
    REQUIRE(test::allclose_abs(cpu_output, gpu_output.to_cpu(), 8e-5));
}
