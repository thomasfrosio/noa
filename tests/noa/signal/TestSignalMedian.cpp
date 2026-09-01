#include <noa/runtime/Factory.hpp>
#include <noa/runtime/Random.hpp>

#include <noa/io/IO.hpp>
#include <noa/signal/MedianFilter.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("signal::median_filter()", "[asset]") {
    const Path path_base = test::noa_data_path() / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        for (size_t nb = 0; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto window = test["window"].as<i32>();
            const auto dim = test["dim"].as<i32>();
            const auto border = test["border"].as<noa::Border>();
            const auto filename_expected = path_base / test["expected"].as<Path>();

            const auto input = noa::read_image<f32, 4>(filename_input, {}, options).data;
            const auto expected = noa::read_image<f32, 4>(filename_expected, {}, options).data;

            const auto result = noa::empty_like(input);
            if (dim == 1)
                noa::signal::median_filter_1d(input, result, {window, border});
            else if (dim == 2)
                noa::signal::median_filter_2d(input, result, {window, border});
            else if (dim == 3)
                noa::signal::median_filter_3d(input, result, {window, border});
            else
                FAIL("dim is not correct");

            REQUIRE(test::allclose_abs(result, expected, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("signal::median_filter(), cpu vs gpu", "", i32, f16, f32, f64) {
    if (not Device::is_any_gpu())
        return;

    constexpr auto shape_options = test::RandomShapeOptions{
        .size_range = {32, 64},
        .batch_range = {1, 4},
    };
    const auto shapes = noa::make_tuple(
        Pair{test::random_shape_batched<isize, 1>(1, shape_options), 1},
        Pair{test::random_shape_batched<isize, 2>(2, shape_options), 2},
        Pair{test::random_shape_batched<isize, 3>(1, shape_options), 1},
        Pair{test::random_shape_batched<isize, 3>(2, shape_options), 2},
        Pair{test::random_shape_batched<isize, 3>(3, shape_options), 3},
        Pair{test::random_shape_batched<isize, 5>(1, shape_options), 1},
        Pair{test::random_shape_batched<isize, 5>(2, shape_options), 2},
        Pair{test::random_shape_batched<isize, 5>(3, shape_options), 3}
    );
    shapes.for_each([&]<usize N>(const Pair<Shape<isize, N>, i32>& pair) {
        const auto& [shape, rank] = pair;
        const auto cpu_data = noa::random<TestType>(noa::Uniform<f32>{-128, 128}, shape);
        const auto gpu_data = cpu_data.to({.device = "gpu", .allocator = Allocator::PITCHED_MANAGED});

        const auto cpu_result = noa::empty_like(cpu_data);
        const auto gpu_result = noa::empty_like(gpu_data);

        for (noa::Border border: {noa::Border::ZERO, noa::Border::REFLECT}) {
            if constexpr (N >= 1) {
                if (rank == 1) {
                    auto window = test::Randomizer<i32>(2, 21).get();
                    if (noa::is_even(window))
                        window -= 1;
                    noa::signal::median_filter_1d(cpu_data, cpu_result, {.window_size = window, .border_mode = border});
                    noa::signal::median_filter_1d(gpu_data, gpu_result, {.window_size = window, .border_mode = border});
                    REQUIRE(test::allclose_abs(cpu_result, gpu_result, 1e-5));
                }
            }
            if constexpr (N >= 2) {
                if (rank == 2) {
                    auto window = test::Randomizer<i32>(2, 11).get();
                    if (noa::is_even(window))
                        window -= 1;
                    noa::signal::median_filter_2d(cpu_data, cpu_result, {.window_size = window, .border_mode = border});
                    noa::signal::median_filter_2d(gpu_data, gpu_result, {.window_size = window, .border_mode = border});
                    REQUIRE(test::allclose_abs(cpu_result, gpu_result, 1e-5));
                }
            }
            if constexpr (N >= 3) {
                if (rank == 3) {
                    auto window = test::Randomizer<i32>(2, 11).get();
                    if (noa::is_even(window))
                        window -= 1;
                    noa::signal::median_filter_2d(cpu_data, cpu_result, {.window_size = window, .border_mode = border});
                    noa::signal::median_filter_2d(gpu_data, gpu_result, {.window_size = window, .border_mode = border});
                    REQUIRE(test::allclose_abs(cpu_result, gpu_result, 1e-5));
                }
            }
        }
    });
}
