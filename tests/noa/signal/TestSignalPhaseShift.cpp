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
namespace ns = noa::signal;

TEST_CASE("signal::phase_shift_2d()", "[asset]") {
    const Path path_base = test::noa_data_path() / "signal";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["2d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        INFO("test=" << i);
        const auto shape = param["shape"].as<Shape4>().pop_front<2>();
        const auto shift = param["shift"].as<Vec<f32, 2>>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::read_image<c32, 2>(path_output, {}, options).data;

            if (path_input.filename().empty()) {
                const auto output = noa::empty<c32, 2>(shape.rfft(), options);
                ns::phase_shift_2d<"h2h">({}, output, shape, shift);
                REQUIRE(test::allclose_abs(expected, output, 1e-4f));

            } else {
                const auto input = noa::read_image<c32, 2>(path_input, {}, options).data;
                ns::phase_shift_2d<"h2h">(input, input, shape, shift);
                REQUIRE(test::allclose_abs(expected, input, 1e-4f));
            }
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_2d(), remap", "", c32, c64) {
    const auto shape = test::random_shape_batched<isize, 3>(2, {
        .batch_range = {2, 5},
        .only_even_sizes = true,
    }); // even sizes for inplace remap
    const auto shift = Vec{31.5, -15.2};
    const f64 cutoff = 0.5;

    using real_t = noa::traits::value_type_t<TestType>;
    Array<Vec<real_t, 2>, 1> shifts = noa::empty<Vec<real_t, 2>, 1>(shape[0]);
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

        const auto input = noa::random<TestType, 3>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);
        {
            const auto output = noa::empty_like(input);
            ns::phase_shift_2d<"h2h">(input, output, shape, shifts, cutoff);
            nf::remap_2d("H2HC", output, output, shape);

            const auto output_remap = noa::empty_like(input);
            ns::phase_shift_2d<"h2hc">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
        {
            const auto output = noa::empty_like(input);
            ns::phase_shift_2d<"h2h">(input, output, shape, shifts, cutoff);

            const auto output_remap = noa::empty_like(input);
            nf::remap_2d("H2HC", input, input, shape);
            ns::phase_shift_2d<"hc2h">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_2d(), batch", "", c32, c64) {
    const auto shape = test::random_shape<i32, 2>(2).push_front(Vec{2, 3, 4}).as<isize>();

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 2>>(shape.filter(0, 1, 2));
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

        const auto input = noa::random<TestType, 5>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);

        const auto output0 = noa::empty_like(input);
        auto span = shifts.span();
        for (i64 i{}; i < shifts.shape()[0]; ++i) {
            for (i64 j{}; j < shifts.shape()[1]; ++j) {
                for (i64 k{}; k < shifts.shape()[2]; ++k) {
                    auto shift = span(i, j, k);
                    ns::phase_shift_2d<"h2h">(
                        input.subregion(i, j, k).template as_nd<2>(),
                        output0.subregion(i, j, k).template as_nd<2>(),
                        shape.pop_front<3>(), shift);
                }
            }
        }

        const auto output1 = noa::empty_like(input);
        ns::phase_shift_2d<"h2h">(input, output1, shape, shifts.to({device}));
        REQUIRE(test::allclose_abs(output0, output1, 1e-6f));
    }
}

TEST_CASE("signal::phase_shift_3d()", "[asset]") {
    const Path path_base = test::noa_data_path() / "signal";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["3d"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<Shape4>().pop_front();
        const auto shift = param["shift"].as<Vec<f32, 3>>();
        const auto path_output = path_base / param["output"].as<Path>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<Path>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto expected = noa::read_image<c32, 3>(path_output, {}, options).data;

            if (path_input.filename().empty()) {
                const auto output = noa::empty<c32, 3>(shape.rfft(), options);
                ns::phase_shift_3d<"h2h">({}, output, shape, shift);
                REQUIRE(test::allclose_abs(expected, output, 1e-4f));

            } else {
                const auto input = noa::read_image<c32, 3>(path_input, {}, options).data;
                ns::phase_shift_3d<"h2h">(input, input, shape, shift);
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
            const auto output = noa::empty_like(input);
            ns::phase_shift_3d<"h2h">(input, output, shape, shifts, cutoff);
            nf::remap_3d("H2HC", output, output, shape);

            const auto output_remap = noa::empty_like(input);
            ns::phase_shift_3d<"h2hc">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }

        AND_THEN("hc2h") {
            const auto output = noa::empty_like(input);
            ns::phase_shift_3d<"h2h">(input, output, shape, shifts, cutoff);

            const auto output_remap = noa::empty_like(input);
            nf::remap_3d("H2HC", input, input, shape);
            ns::phase_shift_3d<"hc2h">(input, output_remap, shape, shift.as<real_t>(), cutoff);

            REQUIRE(test::allclose_abs(output, output_remap, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift_3d(), batch", "", c32, c64) {
    const auto shape = test::random_shape_batched(3);

    using real_t = noa::traits::value_type_t<TestType>;
    auto shifts = noa::empty<Vec<real_t, 3>, 1>(shape[0]);
    auto randomizer = test::Randomizer<real_t>(-30, 30);
    for (auto& e: shifts.span())
        e = {randomizer.get(), randomizer.get(), randomizer.get()};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::random<TestType>(noa::Uniform<f32>{-1, 2}, shape.rfft(), options);
        const auto output0 = noa::empty_like(input);
        for (i64 i{}; auto shift: shifts.span()) {
            ns::phase_shift_3d<"h2h">(input.subregion(i), output0.subregion(i), shape.set<0>(1), shift);
            ++i;
        }

        const auto output1 = noa::empty_like(input);
        ns::phase_shift_3d<"h2h">(input, output1, shape, shifts.to({device}));
        REQUIRE(test::allclose_abs(output0, output1, 1e-6f));
    }
}

TEMPLATE_TEST_CASE("signal::phase_shift{2|3}d(), cpu vs gpu", "", c32, c64) {
    if (not Device::is_any_gpu())
        return;

    constexpr auto shape_options = test::RandomShapeOptions{
        .size_range = {32, 64},
        .batch_range = {1, 4},
    };
    const auto shapes = noa::make_tuple(
        test::random_shape_batched<isize, 2>(2, shape_options),
        test::random_shape_batched<isize, 3>(3, shape_options),
        test::random_shape_batched<isize, 4>(4, shape_options),
        test::random_shape_batched<isize, 5>(3, shape_options),
        test::random_shape_batched<isize, 6>(3, shape_options)
    );

    shapes.for_each([&]<usize N>(const Shape<isize, N>& shape) {
        const auto cpu_input = noa::random<TestType>(noa::Uniform{-TestType{2, 2}, TestType{2, 2}}, shape.rfft());
        const auto gpu_input = cpu_input.to(ArrayOption(Device("gpu"), Allocator::PITCHED_MANAGED));
        const auto cpu_output = noa::empty_like(cpu_input);
        const auto gpu_output = noa::empty_like(gpu_input);

        auto run = [&]<nf::Layout REMAP>(const auto& cpu_shifts, const auto& gpu_shifts) {
            const f64 cutoff = test::Randomizer<f64>(0.2, 0.5).get();
            ns::phase_shift<REMAP>(cpu_input, cpu_output, shape, cpu_shifts, cutoff);
            ns::phase_shift<REMAP>(gpu_input, gpu_output, shape, gpu_shifts, cutoff);
            REQUIRE(test::allclose_abs(cpu_output, gpu_output, 8e-5));
        };

        auto run_full = [&]<usize R, usize BR>(const Shape<isize, BR>& shape_br) {
            constexpr usize B = BR - R;
            const auto shape_b = shape_br.template pop_back<R>();
            const auto cpu_shifts = Array<Vec<f64, R>, B>(shape_b);

            auto randomizer = test::Randomizer<f64>(-15., 15.);
            for (auto& e: cpu_shifts.span_1d())
                for (usize i{}; i < R; ++i)
                    e[i] = randomizer.get();

            const auto gpu_shifts = cpu_shifts.to({gpu_output.device(), Allocator::ASYNC});

            run.template operator()<nf::Layout::H2H>(cpu_shifts, gpu_shifts);
            run.template operator()<nf::Layout::H2HC>(cpu_shifts, gpu_shifts);
            run.template operator()<nf::Layout::HC2HC>(cpu_shifts, gpu_shifts);
            run.template operator()<nf::Layout::HC2H>(cpu_shifts, gpu_shifts);
        };

        if constexpr (N > 1)
            run_full.template operator()<1>(shape);
        if constexpr (N > 2)
            run_full.template operator()<2>(shape);
        if constexpr (N > 3)
            run_full.template operator()<3>(shape);
    });
}
