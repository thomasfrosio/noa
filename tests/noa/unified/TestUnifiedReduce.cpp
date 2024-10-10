#include <noa/unified/Array.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"
#include "Assets.h"

using namespace noa::types;

TEST_CASE("unified:: reductions vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<Shape4<i64>>();
    const auto input_filename = path / input["path"].as<Path>();
    const auto output_filename = path / tests["all"]["output_path"].as<Path>();

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const auto expected_max = expected["max"].as<f32>();
    const auto expected_min = expected["min"].as<f32>();
    const auto expected_median = expected["median"].as<f32>();
    const auto expected_mean = expected["mean"].as<f32>();
    const auto expected_norm = expected["norm"].as<f32>();
    const auto expected_std = expected["std"].as<f32>();
    const auto expected_sum = expected["sum"].as<f32>();
    const auto expected_var = expected["var"].as<f32>();

    auto data = noa::io::read_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto min = noa::min(data);
        const auto max = noa::max(data);
        const auto min_max = noa::min_max(data);
        const auto median = noa::median(data);
        const auto sum = noa::sum(data);
        const auto mean = noa::mean(data);
        const auto norm = noa::l2_norm(data);
        const auto var = noa::variance(data);
        const auto std = noa::stddev(data);
        const auto mean_var = noa::mean_variance(data);
        const auto mean_std = noa::mean_stddev(data);

        REQUIRE_THAT(min, Catch::WithinAbs(expected_min, 1e-6));
        REQUIRE_THAT(max, Catch::WithinAbs(expected_max, 1e-6));
        REQUIRE_THAT(min_max.first, Catch::WithinAbs(expected_min, 1e-6));
        REQUIRE_THAT(min_max.second, Catch::WithinAbs(expected_max, 1e-6));
        REQUIRE_THAT(median, Catch::WithinAbs(expected_median, 1e-6));
        REQUIRE_THAT(sum, Catch::WithinRel(expected_sum));
        REQUIRE_THAT(mean, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(norm, Catch::WithinRel(expected_norm));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
        REQUIRE_THAT(mean_var.first, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(mean_var.second, Catch::WithinRel(expected_var));
        REQUIRE_THAT(mean_std.first, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(mean_std.second, Catch::WithinRel(expected_std));
    }
}

TEST_CASE("unified:: reductions complex vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_complex"];
    const auto shape = tests["shape"].as<Shape4<i64>>();
    const auto input_filename = path / tests["input_path"].as<Path>();
    const auto output_filename = path / tests["output_path"].as<Path>();

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const c32 expected_sum{expected["sum_real"].as<f32>(),
                           expected["sum_imag"].as<f32>()};
    const c32 expected_mean{expected["mean_real"].as<f32>(),
                            expected["mean_imag"].as<f32>()};
    const auto expected_norm = expected["norm"].as<f32>();
    const auto expected_std = expected["std"].as<f32>();
    const auto expected_var = expected["var"].as<f32>();

    auto data = noa::io::read_data<c32>(input_filename);
    REQUIRE(noa::all(data.shape() == shape));

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto complex_sum = noa::sum(data);
        const auto complex_mean = noa::mean(data);
        const auto norm = noa::l2_norm(data);
        const auto var = noa::variance(data, 1);
        const auto std = noa::stddev(data, 1);

        REQUIRE_THAT(complex_sum.real, Catch::WithinRel(expected_sum.real));
        REQUIRE_THAT(complex_sum.imag, Catch::WithinRel(expected_sum.imag));
        REQUIRE_THAT(complex_mean.real, Catch::WithinRel(expected_mean.real));
        REQUIRE_THAT(complex_mean.imag, Catch::WithinRel(expected_mean.imag));
        REQUIRE_THAT(norm, Catch::WithinRel(expected_norm));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));

        const auto [mean_var0, mean_var1] = noa::mean_variance(data, 1);
        REQUIRE_THAT(mean_var0.real, Catch::WithinRel(expected_mean.real));
        REQUIRE_THAT(mean_var0.imag, Catch::WithinRel(expected_mean.imag));
        REQUIRE_THAT(mean_var1, Catch::WithinRel(expected_var));

        const auto [mean_std0, mean_std1] = noa::mean_stddev(data, 1);
        REQUIRE_THAT(mean_std0.real, Catch::WithinRel(expected_mean.real));
        REQUIRE_THAT(mean_std0.imag, Catch::WithinRel(expected_mean.imag));
        REQUIRE_THAT(mean_std1, Catch::WithinRel(expected_std));
    }
}

TEMPLATE_TEST_CASE("unified:: reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const auto pad = GENERATE(true, false);
    const auto subregion_shape = test::random_shape_batched(3) + Shape4<i64>{1, 164, 164, 164};
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }
    INFO(pad);

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.n_elements(), randomizer);
    auto gpu_data = cpu_data.to({.device="gpu", .allocator="managed"});

    cpu_data = cpu_data.subregion(
            noa::indexing::FullExtent{},
            noa::indexing::Slice{0, subregion_shape[1]},
            noa::indexing::Slice{0, subregion_shape[2]},
            noa::indexing::Slice{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
            noa::indexing::FullExtent{},
            noa::indexing::Slice{0, subregion_shape[1]},
            noa::indexing::Slice{0, subregion_shape[2]},
            noa::indexing::Slice{0, subregion_shape[3]});

    using real_t = noa::traits::value_type_t<TestType>;
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(5e-5) : static_cast<real_t>(1e-10);

    if constexpr (not noa::traits::complex<TestType>) {
        const auto cpu_min = noa::min(cpu_data);
        const auto cpu_max = noa::max(cpu_data);
        const auto cpu_min_max = noa::min_max(cpu_data);
        const auto cpu_median = noa::median(cpu_data);

        const auto gpu_min = noa::min(gpu_data);
        const auto gpu_max = noa::max(gpu_data);
        const auto gpu_min_max = noa::min_max(gpu_data);
        const auto gpu_median = noa::median(gpu_data);

        REQUIRE_THAT(gpu_min, Catch::WithinAbs(static_cast<f64>(cpu_min), 1e-6));
        REQUIRE_THAT(gpu_max, Catch::WithinAbs(static_cast<f64>(cpu_max), 1e-6));
        REQUIRE_THAT(gpu_min_max.first, Catch::WithinAbs(static_cast<f64>(cpu_min_max.first), 1e-6));
        REQUIRE_THAT(gpu_min_max.second, Catch::WithinAbs(static_cast<f64>(cpu_min_max.second), 1e-6));
        REQUIRE_THAT(gpu_median, Catch::WithinAbs(static_cast<f64>(cpu_median), 1e-6));
    }

    if constexpr (not noa::traits::integer<TestType>) {
        const auto cpu_norm = noa::l2_norm(cpu_data);
        const auto cpu_var = noa::variance(cpu_data);
        const auto cpu_std = noa::stddev(cpu_data);
        const auto cpu_mean_var = noa::mean_variance(cpu_data);

        const auto gpu_norm = noa::l2_norm(gpu_data);
        const auto gpu_var = noa::variance(gpu_data);
        const auto gpu_std = noa::stddev(gpu_data);
        const auto gpu_mean_var = noa::mean_variance(gpu_data);

        REQUIRE(test::allclose_abs_safe(&gpu_norm, &cpu_norm, 1, eps));
        REQUIRE(test::allclose_abs_safe(&gpu_var, &cpu_var, 1, eps));
        REQUIRE(test::allclose_abs_safe(&gpu_std, &cpu_std, 1, eps));
        REQUIRE(test::allclose_abs_safe(&gpu_mean_var.first, &cpu_mean_var.first, 1, eps));
        REQUIRE(test::allclose_abs_safe(&gpu_mean_var.second, &cpu_mean_var.second, 1, eps));
    }

    const auto cpu_sum = noa::sum(cpu_data);
    const auto cpu_mean = noa::mean(cpu_data);
    const auto gpu_sum = noa::sum(gpu_data);
    const auto gpu_mean = noa::mean(gpu_data);

    REQUIRE(test::allclose_abs_safe(&cpu_sum, &gpu_sum, 1, eps));
    REQUIRE(test::allclose_abs_safe(&cpu_mean, &gpu_mean, 1, eps));
}
