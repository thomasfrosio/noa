#include <noa/unified/Array.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/Factory.hpp>

#include <noa/unified/io/ImageFile.hpp>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

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

        REQUIRE_THAT(min, Catch::WithinAbs(static_cast<f64>(expected_min), 1e-6));
        REQUIRE_THAT(max, Catch::WithinAbs(static_cast<f64>(expected_max), 1e-6));
        REQUIRE_THAT(min_max.first, Catch::WithinAbs(static_cast<f64>(expected_min), 1e-6));
        REQUIRE_THAT(min_max.second, Catch::WithinAbs(static_cast<f64>(expected_max), 1e-6));
        REQUIRE_THAT(median, Catch::WithinAbs(static_cast<f64>(expected_median), 1e-6));
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
    REQUIRE(all(data.shape() == shape));

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
    const auto subregion_shape = test::get_random_shape4_batched(3) + Shape4<i64>{1, 164, 164, 164};
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }
    INFO(pad);

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.elements(), randomizer);
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

    if constexpr (!noa::traits::is_complex_v<TestType>) {
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

    if constexpr (!noa::traits::is_int_v<TestType>){
        const auto cpu_norm = noa::l2_norm(cpu_data);
        const auto cpu_var = noa::variance(cpu_data);
        const auto cpu_std = noa::stddev(cpu_data);
        const auto cpu_mean_var = noa::mean_variance(cpu_data);

        const auto gpu_norm = noa::l2_norm(gpu_data);
        const auto gpu_var = noa::variance(gpu_data);
        const auto gpu_std = noa::stddev(gpu_data);
        const auto gpu_mean_var = noa::mean_variance(gpu_data);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_norm, &cpu_norm, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_std, &cpu_std, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean_var.first, &cpu_mean_var.first, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean_var.second, &cpu_mean_var.second, 1, eps));
    }

    const auto cpu_sum = noa::sum(cpu_data);
    const auto cpu_mean = noa::mean(cpu_data);
    const auto gpu_sum = noa::sum(gpu_data);
    const auto gpu_mean = noa::mean(gpu_data);

    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &cpu_sum, &gpu_sum, 1, eps));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &cpu_mean, &gpu_mean, 1, eps));
}

TEST_CASE("unified:: batched reductions vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<Shape4<i64>>();
    const auto output_shape = tests["batch"]["output_shape"].as<Shape4<i64>>();
    const auto input_filename = path / input["path"].as<Path>();
    const auto output_filename = path / tests["batch"]["output_path"].as<Path>();

    auto data = noa::io::read_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    const YAML::Node expected = YAML::LoadFile(output_filename);
    std::vector<f32> expected_max, expected_min, expected_mean, expected_norm, expected_std, expected_sum, expected_var;
    for (size_t i = 0; i < expected.size(); i++) {
        expected_max.emplace_back(expected[i]["max"].as<f32>());
        expected_min.emplace_back(expected[i]["min"].as<f32>());
        expected_mean.emplace_back(expected[i]["mean"].as<f32>());
        expected_norm.emplace_back(expected[i]["norm"].as<f32>());
        expected_std.emplace_back(expected[i]["std"].as<f32>());
        expected_sum.emplace_back(expected[i]["sum"].as<f32>());
        expected_var.emplace_back(expected[i]["var"].as<f32>());
    }

   std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const Array<f32> results({7, 1, 1, output_shape.elements()}, options);
        const auto mins = results.subregion(0).reshape(output_shape);
        const auto maxs = results.subregion(1).reshape(output_shape);
        const auto sums = results.subregion(2).reshape(output_shape);
        const auto means = results.subregion(3).reshape(output_shape);
        const auto norms = results.subregion(4).reshape(output_shape);
        const auto vars = results.subregion(5).reshape(output_shape);
        const auto stds = results.subregion(6).reshape(output_shape);

        noa::min(data, mins);
        noa::max(data, maxs);
        noa::sum(data, sums);
        noa::mean(data, means);
        noa::l2_norm(data, norms);
        noa::variance(data, vars);
        noa::stddev(data, stds);
        data.eval();

        for (uint batch = 0; batch < shape[0]; ++batch) {
            REQUIRE_THAT(mins(batch, 0, 0, 0), Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
            REQUIRE_THAT(maxs(batch, 0, 0, 0), Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
            REQUIRE_THAT(sums(batch, 0, 0, 0), Catch::WithinRel(expected_sum[batch]));
            REQUIRE_THAT(means(batch, 0, 0, 0), Catch::WithinRel(expected_mean[batch]));
            REQUIRE_THAT(norms(batch, 0, 0, 0), Catch::WithinRel(expected_norm[batch]));
            REQUIRE_THAT(vars(batch, 0, 0, 0), Catch::WithinRel(expected_var[batch]));
            REQUIRE_THAT(stds(batch, 0, 0, 0), Catch::WithinRel(expected_std[batch]));
        }
    }
}

TEMPLATE_TEST_CASE("unified:: batched reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (!Device::is_any_gpu())
        return;

    const auto pad = GENERATE(true, false);
    const auto large = GENERATE(true, false);
    const auto subregion_shape =
            test::get_random_shape4_batched(3) +
            (large ? Shape4<i64>{1, 164, 164, 164} : Shape4 < i64 > {});
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }
    INFO("padded: " << pad);

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.elements(), randomizer);
    auto gpu_data = cpu_data.to(ArrayOption(Device("gpu"), "managed"));

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
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(1e-4) : static_cast<real_t>(1e-10);
    const auto output_shape = Shape4<i64>{subregion_shape.batch(), 1, 1, 1};

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
        const auto cpu_min = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_max = cpu_results.view().subregion(1).reshape(output_shape);
        noa::min(cpu_data, cpu_min);
        noa::max(cpu_data, cpu_max);

        const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
        const auto gpu_min = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_max = gpu_results.view().subregion(1).reshape(output_shape);
        noa::min(gpu_data, gpu_min);
        noa::max(gpu_data, gpu_max);

        REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, cpu_results, gpu_results, 1e-8));
    }

    if constexpr (!noa::traits::is_int_v<TestType>){
        const Array<real_t> cpu_results({3, 1, 1, output_shape.elements()});
        const auto cpu_norm = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_var = cpu_results.view().subregion(1).reshape(output_shape);
        const auto cpu_std = cpu_results.view().subregion(2).reshape(output_shape);

        const Array<real_t> gpu_results({3, 1, 1, output_shape.elements()}, gpu_data.options());
        const auto gpu_norm = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_var = gpu_results.view().subregion(1).reshape(output_shape);
        const auto gpu_std = gpu_results.view().subregion(2).reshape(output_shape);

        noa::l2_norm(cpu_data, cpu_norm);
        noa::l2_norm(gpu_data, gpu_norm);
        REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_norm.to_cpu(), gpu_norm.to_cpu(), eps));

        for (i64 ddof = 0; ddof < 2; ++ddof) {
            INFO("ddof: " << ddof);
            noa::variance(cpu_data, cpu_var, ddof);
            noa::stddev(cpu_data, cpu_std, ddof);

            noa::variance(gpu_data, gpu_var, ddof);
            noa::stddev(gpu_data, gpu_std, ddof);

            REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
        }
    }

    const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
    const auto cpu_sum = cpu_results.view().subregion(0).reshape(output_shape);
    const auto cpu_mean = cpu_results.view().subregion(1).reshape(output_shape);
    noa::sum(cpu_data, cpu_sum);
    noa::mean(cpu_data, cpu_mean);

    const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
    const auto gpu_sum = gpu_results.view().subregion(0).reshape(output_shape);
    const auto gpu_mean = gpu_results.view().subregion(1).reshape(output_shape);
    noa::sum(gpu_data, gpu_sum);
    noa::mean(gpu_data, gpu_mean);

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
}

TEST_CASE("unified:: axis reductions vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto input_filename = path / input["path"].as<Path>();
    const auto shape = input["shape"].as<Shape4<i64>>();

    auto data = noa::io::read_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

   std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (size_t i = 0; i < 4; ++i) {
        INFO(i);
        const std::string key = fmt::format("axis{}", i);
        const auto output_path_min = path / tests[key]["output_min"].as<Path>();
        const auto output_path_max = path / tests[key]["output_max"].as<Path>();
        const auto output_path_sum = path / tests[key]["output_sum"].as<Path>();
        const auto output_path_mean = path / tests[key]["output_mean"].as<Path>();
        const auto output_path_norm = path / tests[key]["output_norm"].as<Path>();
        const auto output_path_var = path / tests[key]["output_var"].as<Path>();
        const auto output_path_std = path / tests[key]["output_std"].as<Path>();
        const auto output_shape = tests[key]["output_shape"].as < Shape4 < i64 >> ();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, "managed");
            INFO(device);
            data = device == data.device() ? data : data.to(options);

            const Array<f32> output(output_shape, options);

            auto expected = noa::io::read_data<f32>(output_path_min);
            noa::min(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::read_data<f32>(output_path_max);
            noa::max(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::read_data<f32>(output_path_sum);
            noa::sum(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected.release(), output, 1e-5));

            const auto expected_std = noa::io::read_data<f32>(output_path_std);
            noa::stddev(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_std, output, 1e-5));

            const auto expected_mean = noa::io::read_data<f32>(output_path_mean);
            noa::mean(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_mean, output, 1e-5));

            const auto expected_norm = noa::io::read_data<f32>(output_path_norm);
            noa::l2_norm(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_norm, output, 1e-5));

            const auto expected_variance = noa::io::read_data<f32>(output_path_var);
            noa::variance(data, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_variance, output, 1e-5));

            const Array<f32> mean(output_shape, options);
            noa::mean_variance(data, mean, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_mean, mean, 1e-5));
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_variance, output, 1e-5));

            noa::mean_stddev(data, mean, output);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_mean, mean, 1e-5));
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, expected_std, output, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified:: axis reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (!Device::is_any_gpu())
        return;

    const auto pad = GENERATE(true, false);
    const auto large = GENERATE(true, false);
    const auto subregion_shape =
            test::get_random_shape4_batched(3) +
            (large ? Shape4<i64>{1, 64, 64, 64} : Shape4<i64>{});
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }
    INFO("padded: " << pad);
    INFO("large: " << large);

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.elements(), randomizer);
    auto gpu_data = cpu_data.to(ArrayOption(Device("gpu"), "managed"));

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

    for (i64 axis = 0; axis < 4; ++axis) {
        INFO("axis: " << axis);
        auto output_shape = subregion_shape;
        output_shape[axis] = 1;

        if constexpr (!noa::traits::is_complex_v<TestType>) {
            const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
            const auto cpu_min = cpu_results.view().subregion(0).reshape(output_shape);
            const auto cpu_max = cpu_results.view().subregion(1).reshape(output_shape);
            noa::min(cpu_data, cpu_min);
            noa::max(cpu_data, cpu_max);

            const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
            const auto gpu_min = gpu_results.view().subregion(0).reshape(output_shape);
            const auto gpu_max = gpu_results.view().subregion(1).reshape(output_shape);
            noa::min(gpu_data, gpu_min);
            noa::max(gpu_data, gpu_max);

            REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, cpu_results, gpu_results, 1e-8));
        }

        if constexpr (!noa::traits::is_int_v<TestType>) {
            auto cpu_norm = noa::l2_norm(cpu_data, ReduceAxes::from_shape(output_shape));
            auto gpu_norm = noa::l2_norm(gpu_data, ReduceAxes::from_shape(output_shape));
            REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_norm.release(), gpu_norm.release(), eps));

            for (i64 ddof = 0; ddof < 2; ++ddof) {
                INFO("ddof: " << ddof);
                const Array<real_t> cpu_results({2, 1, 1, output_shape.elements()});
                const auto cpu_var = cpu_results.view().subregion(0).reshape(output_shape);
                const auto cpu_std = cpu_results.view().subregion(1).reshape(output_shape);
                noa::variance(cpu_data, cpu_var, ddof);
                noa::stddev(cpu_data, cpu_std, ddof);

                const Array<real_t> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
                const auto gpu_var = gpu_results.view().subregion(0).reshape(output_shape);
                const auto gpu_std = gpu_results.view().subregion(1).reshape(output_shape);
                noa::variance(gpu_data, gpu_var, ddof);
                noa::stddev(gpu_data, gpu_std, ddof);

                REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));

                const Array<TestType> cpu_mean(output_shape);
                const Array<real_t> cpu_variance(output_shape);
                const Array<TestType> gpu_mean(output_shape, gpu_data.options());
                const Array<real_t> gpu_variance(output_shape, gpu_data.options());

                noa::mean_variance(cpu_data, cpu_mean, cpu_variance);
                noa::mean_variance(gpu_data, gpu_mean, gpu_variance);
                REQUIRE(test::Matcher<TestType>(test::MATCH_ABS_SAFE, cpu_mean, gpu_mean, eps));
                REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_variance, gpu_variance, eps));

                noa::mean_stddev(cpu_data, cpu_mean, cpu_variance);
                noa::mean_stddev(gpu_data, gpu_mean, gpu_variance);
                REQUIRE(test::Matcher<TestType>(test::MATCH_ABS_SAFE, cpu_mean, gpu_mean, eps));
                REQUIRE(test::Matcher<real_t>(test::MATCH_ABS_SAFE, cpu_variance, gpu_variance, eps));
            }
        }

        const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
        const auto cpu_sum = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_mean = cpu_results.view().subregion(1).reshape(output_shape);
        noa::sum(cpu_data, cpu_sum);
        noa::mean(cpu_data, cpu_mean);

        const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
        const auto gpu_sum = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_mean = gpu_results.view().subregion(1).reshape(output_shape);
        noa::sum(gpu_data, gpu_sum);
        noa::mean(gpu_data, gpu_mean);

        REQUIRE(test::Matcher<TestType>(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
    }
}

TEST_CASE("unified::argmax/argmin()", "[noa][unified]") {
    const bool small = GENERATE(true, false);
    const auto shape = small ? Shape4<i64>{3, 8, 41, 65} : Shape4<i64>{3, 256, 256, 300};
    const auto n_elements_per_batch = shape.pop_front().template as<u32>().elements();

    std::vector<Device> devices{Device{}};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    AND_THEN("reduce entire array") {
        Array<f32> input(shape);
        test::randomize(input.get(), input.elements(), test::Randomizer<f32>(-100., 100.));

        auto expected_min_offset = test::Randomizer<u32>(i64{}, input.elements() - 2).get();
        auto expected_max_offset = test::Randomizer<u32>(i64{}, input.elements() - 2).get();
        if (expected_min_offset == expected_max_offset)
            expected_max_offset += 1;

        input.span()[expected_min_offset] = -101;
        input.span()[expected_max_offset] = 101;
        for (auto& device: devices) {
            INFO(device);
            const auto options = ArrayOption{device, "managed"};
            input = input.device().is_cpu() ? input : input.to(options);

            const auto [min, min_offset] = noa::argmin(input);
            REQUIRE((min == -101 and min_offset == expected_min_offset));
            const auto [max, max_offset] = noa::argmax(input);
            REQUIRE((max == 101 and max_offset == expected_max_offset));
        }
    }

    AND_THEN("per batch") {
        Array<f32> input(shape);
        test::randomize(input.get(), input.elements(), test::Randomizer<f32>(-100., 100.));

        Array min_values = noa::empty<f32>({shape.batch(), 1, 1, 1});
        Array min_offsets = noa::like<i32>(min_values);
        Array expected_min_values = noa::like<f32>(min_values);
        Array expected_min_offsets = noa::like<i32>(min_values);

        test::randomize(expected_min_values.get(), expected_min_values.elements(),
                        test::Randomizer<f32>(-200, -101));
        test::randomize(expected_min_offsets.get(), expected_min_offsets.elements(),
                        test::Randomizer<i32>(0u, n_elements_per_batch - 1));

        const auto input_2d = input.reshape({shape.batch(), 1, 1, -1});
        for (i64 batch = 0; batch < shape.batch(); ++batch) {
            auto& offset = expected_min_offsets(batch, 0, 0, 0);
            input_2d(batch, 0, 0, offset) = expected_min_values(batch, 0, 0, 0);
        }

        for (auto& device: devices) {
            INFO(device);
            const auto options = ArrayOption{device, "managed"};
            if (input.device() != device) {
                input = input.to(options);
                min_values = min_values.to(options);
                min_offsets = min_offsets.to(options);
            }

            noa::argmin(input, min_values, min_offsets);
            REQUIRE(test::Matcher<f32>(test::MATCH_ABS, min_values, expected_min_values, 1e-7));

            // Offsets are not relative to each batch...
            for (i64 batch = 0; auto& value: expected_min_offsets.span()) {
                value += static_cast<i32>(batch * n_elements_per_batch);
                ++batch;
            }
            REQUIRE(test::Matcher<i32>(test::MATCH_ABS, min_offsets, expected_min_offsets, 1e-7));
        }
    }
}
