#include <noa/unified/Array.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("unified::reduce - vs numpy", "[asset]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<Shape4<i64>>();
    const auto input_filename = path / input["path"].as<Path>();
    const auto output_filename = path / tests["all"]["output_path"].as<Path>();

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const auto expected_max = expected["max"].as<f64>();
    const auto expected_min = expected["min"].as<f64>();
    const auto expected_median = expected["median"].as<f64>();
    const auto expected_mean = expected["mean"].as<f64>();
    const auto expected_norm = expected["norm"].as<f64>();
    const auto expected_std = expected["std"].as<f64>();
    const auto expected_sum = expected["sum"].as<f64>();
    const auto expected_var = expected["var"].as<f64>();

    auto data = noa::read_image<f64>(input_filename).data;
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

        REQUIRE(noa::allclose(min, expected_min));
        REQUIRE(noa::allclose(max, expected_max));
        REQUIRE(noa::allclose(min_max.first, expected_min));
        REQUIRE(noa::allclose(min_max.second, expected_max));
        REQUIRE(noa::allclose(median, expected_median));
        REQUIRE(noa::allclose(sum, expected_sum));
        REQUIRE(noa::allclose(mean, expected_mean));
        REQUIRE(noa::allclose(norm, expected_norm));
        REQUIRE(noa::allclose(var, expected_var));
        REQUIRE(noa::allclose(std, expected_std));
        REQUIRE(noa::allclose(mean_var.first, expected_mean));
        REQUIRE(noa::allclose(mean_var.second, expected_var));
        REQUIRE(noa::allclose(mean_std.first, expected_mean));
        REQUIRE(noa::allclose(mean_std.second, expected_std));
    }
}

TEST_CASE("unified::reduce - complex vs numpy", "[assets]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_complex"];
    const auto shape = tests["shape"].as<Shape4<i64>>();
    const auto input_filename = path / tests["input_path"].as<Path>();
    const auto output_filename = path / tests["output_path"].as<Path>();

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const c64 expected_sum{expected["sum_real"].as<f64>(), expected["sum_imag"].as<f64>()};
    const c64 expected_mean{expected["mean_real"].as<f64>(), expected["mean_imag"].as<f64>()};
    const auto expected_norm = expected["norm"].as<f64>();
    const auto expected_std = expected["std"].as<f64>();
    const auto expected_var = expected["var"].as<f64>();

    auto data = noa::read_image<c64>(input_filename).data;

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
        const auto var = noa::variance(data, {.ddof = 1});
        const auto std = noa::stddev(data, {.ddof = 1});

        REQUIRE(noa::allclose(complex_sum.real, expected_sum.real));
        REQUIRE(noa::allclose(complex_sum.imag, expected_sum.imag));
        REQUIRE(noa::allclose(complex_mean.real, expected_mean.real));
        REQUIRE(noa::allclose(complex_mean.imag, expected_mean.imag));
        REQUIRE(noa::allclose(norm, expected_norm));
        REQUIRE(noa::allclose(var, expected_var, 1e-5));
        REQUIRE(noa::allclose(std, expected_std, 1e-5));

        const auto [mean_var0, mean_var1] = noa::mean_variance(data, {.ddof = 1});
        REQUIRE(noa::allclose(mean_var0.real, expected_mean.real));
        REQUIRE(noa::allclose(mean_var0.imag, expected_mean.imag));
        REQUIRE(noa::allclose(mean_var1, expected_var, 1e-5));

        const auto [mean_std0, mean_std1] = noa::mean_stddev(data, {.ddof = 1});
        REQUIRE(noa::allclose(mean_std0.real, expected_mean.real));
        REQUIRE(noa::allclose(mean_std0.imag, expected_mean.imag));
        REQUIRE(noa::allclose(mean_std1, expected_std, 1e-5));
    }
}

TEMPLATE_TEST_CASE("unified::reduce - cpu vs gpu", "", f32) { // , c32, c64
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
        noa::indexing::Full{},
        noa::indexing::Slice{0, subregion_shape[1]},
        noa::indexing::Slice{0, subregion_shape[2]},
        noa::indexing::Slice{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
        noa::indexing::Full{},
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

        REQUIRE(noa::allclose(gpu_min, cpu_min));
        REQUIRE(noa::allclose(gpu_max, cpu_max));
        REQUIRE(noa::allclose(gpu_min_max.first, cpu_min_max.first));
        REQUIRE(noa::allclose(gpu_min_max.second, cpu_min_max.second));
        REQUIRE(noa::allclose(gpu_median, cpu_median));
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
