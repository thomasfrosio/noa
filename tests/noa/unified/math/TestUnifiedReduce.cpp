#include <noa/unified/Array.hpp>
#include <noa/unified/math/Reduce.hpp>

#include <noa/unified/io/ImageFile.hpp>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("unified::math:: reductions vs numpy", "[assets][noa][unified]") {
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
    const auto expected_std = expected["std"].as<f32>();
    const auto expected_sum = expected["sum"].as<f32>();
    const auto expected_var = expected["var"].as<f32>();

    auto data = noa::io::load_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto min = math::min(data);
        const auto max = math::max(data);
        const auto median = math::median(data);
        const auto sum = math::sum(data);
        const auto mean = math::mean(data);
        const auto var = math::var(data);
        const auto std = math::std(data);
        const auto mean_var = math::mean_var(data);

        REQUIRE_THAT(min, Catch::WithinAbs(static_cast<double>(expected_min), 1e-6));
        REQUIRE_THAT(max, Catch::WithinAbs(static_cast<double>(expected_max), 1e-6));
        REQUIRE_THAT(median, Catch::WithinAbs(static_cast<double>(expected_median), 1e-6));
        REQUIRE_THAT(sum, Catch::WithinRel(expected_sum));
        REQUIRE_THAT(mean, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
        REQUIRE_THAT(mean_var.first, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(mean_var.second, Catch::WithinRel(expected_var));
    }
}

TEST_CASE("unified::math:: reductions complex vs numpy", "[assets][noa][unified]") {
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
    const auto expected_std = expected["std"].as<f32>();
    const auto expected_var = expected["var"].as<f32>();

    auto data = noa::io::load_data<c32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto complex_sum = math::sum(data);
        const auto complex_mean = math::mean(data);
        const auto var = math::var(data, 1);
        const auto std = math::std(data, 1);

        REQUIRE_THAT(complex_sum.real, Catch::WithinRel(expected_sum.real));
        REQUIRE_THAT(complex_sum.imag, Catch::WithinRel(expected_sum.imag));
        REQUIRE_THAT(complex_mean.real, Catch::WithinRel(expected_mean.real));
        REQUIRE_THAT(complex_mean.imag, Catch::WithinRel(expected_mean.imag));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
    }
}

TEMPLATE_TEST_CASE("unified::math:: reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (!Device::is_any(DeviceType::GPU))
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
    auto gpu_data = cpu_data.to(ArrayOption(Device("gpu"), Allocator::DEFAULT_ASYNC));

    cpu_data = cpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});

    using real_t = noa::traits::value_type_t<TestType>;
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(5e-5) : static_cast<real_t>(1e-10);

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        const auto cpu_min = math::min(cpu_data);
        const auto cpu_max = math::max(cpu_data);
        const auto cpu_median = math::median(cpu_data);

        const auto gpu_min = math::min(gpu_data);
        const auto gpu_max = math::max(gpu_data);
        const auto gpu_median = math::median(gpu_data);

        REQUIRE_THAT(gpu_min, Catch::WithinAbs(static_cast<f64>(cpu_min), 1e-6));
        REQUIRE_THAT(gpu_max, Catch::WithinAbs(static_cast<f64>(cpu_max), 1e-6));
        REQUIRE_THAT(gpu_median, Catch::WithinAbs(static_cast<f64>(cpu_median), 1e-6));
    }

    if constexpr (!noa::traits::is_int_v<TestType>){
        const auto cpu_var = math::var(cpu_data);
        const auto cpu_std = math::std(cpu_data);
        const auto cpu_mean_var = math::mean_var(cpu_data);
        const auto gpu_var = math::var(gpu_data);
        const auto gpu_std = math::std(gpu_data);
        const auto gpu_mean_var = math::mean_var(gpu_data);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_std, &cpu_std, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean_var.first, &cpu_mean_var.first, 1, eps));
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean_var.second, &cpu_mean_var.second, 1, eps));
    }

    const auto cpu_sum = math::sum(cpu_data);
    const auto cpu_mean = math::mean(cpu_data);
    const auto gpu_sum = math::sum(gpu_data);
    const auto gpu_mean = math::mean(gpu_data);

    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &cpu_sum, &gpu_sum, 1, eps));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &cpu_mean, &gpu_mean, 1, eps));
}

TEST_CASE("unified::math:: batched reductions vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<Shape4<i64>>();
    const auto output_shape = tests["batch"]["output_shape"].as<Shape4<i64>>();
    const auto input_filename = path / input["path"].as<Path>();
    const auto output_filename = path / tests["batch"]["output_path"].as<Path>();

    auto data = noa::io::load_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    const YAML::Node expected = YAML::LoadFile(output_filename);
    std::vector<f32> expected_max, expected_min, expected_mean, expected_std, expected_sum, expected_var;
    for (size_t i = 0; i < expected.size(); i++) {
        expected_max.emplace_back(expected[i]["max"].as<f32>());
        expected_min.emplace_back(expected[i]["min"].as<f32>());
        expected_mean.emplace_back(expected[i]["mean"].as<f32>());
        expected_std.emplace_back(expected[i]["std"].as<f32>());
        expected_sum.emplace_back(expected[i]["sum"].as<f32>());
        expected_var.emplace_back(expected[i]["var"].as<f32>());
    }

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const Array<f32> results({6, 1, 1, output_shape.elements()}, options);
        const auto mins = results.subregion(0).reshape(output_shape);
        const auto maxs = results.subregion(1).reshape(output_shape);
        const auto sums = results.subregion(2).reshape(output_shape);
        const auto means = results.subregion(3).reshape(output_shape);
        const auto vars = results.subregion(4).reshape(output_shape);
        const auto stds = results.subregion(5).reshape(output_shape);

        noa::math::min(data, mins);
        noa::math::max(data, maxs);
        noa::math::sum(data, sums);
        noa::math::mean(data, means);
        noa::math::var(data, vars);
        noa::math::std(data, stds);
        data.eval();

        for (uint batch = 0; batch < shape[0]; ++batch) {
            REQUIRE_THAT(mins(batch, 0, 0, 0), Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
            REQUIRE_THAT(maxs(batch, 0, 0, 0), Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
            REQUIRE_THAT(sums(batch, 0, 0, 0), Catch::WithinRel(expected_sum[batch]));
            REQUIRE_THAT(means(batch, 0, 0, 0), Catch::WithinRel(expected_mean[batch]));
            REQUIRE_THAT(vars(batch, 0, 0, 0), Catch::WithinRel(expected_var[batch]));
            REQUIRE_THAT(stds(batch, 0, 0, 0), Catch::WithinRel(expected_std[batch]));
        }
    }
}

TEMPLATE_TEST_CASE("unified::math:: batched reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (!Device::is_any(DeviceType::GPU))
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
    auto gpu_data = cpu_data.to(ArrayOption(Device("gpu"), Allocator::MANAGED));

    cpu_data = cpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});

    using real_t = noa::traits::value_type_t<TestType>;
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(1e-4) : static_cast<real_t>(1e-10);
    const auto output_shape = Shape4<i64>{subregion_shape.batch(), 1, 1, 1};

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
        const auto cpu_min = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_max = cpu_results.view().subregion(1).reshape(output_shape);
        math::min(cpu_data, cpu_min);
        math::max(cpu_data, cpu_max);

        const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
        const auto gpu_min = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_max = gpu_results.view().subregion(1).reshape(output_shape);
        math::min(gpu_data, gpu_min);
        math::max(gpu_data, gpu_max);

        REQUIRE(test::Matcher(test::MATCH_ABS, cpu_results, gpu_results, 1e-8));
    }

    if constexpr (!noa::traits::is_int_v<TestType>){
        for (i64 ddof = 0; ddof < 2; ++ddof) {
            INFO("ddof: " << ddof);
            const Array<real_t> cpu_results({2, 1, 1, output_shape.elements()});
            const auto cpu_var = cpu_results.view().subregion(0).reshape(output_shape);
            const auto cpu_std = cpu_results.view().subregion(1).reshape(output_shape);
            math::var(cpu_data, cpu_var, ddof);
            math::std(cpu_data, cpu_std, ddof);

            const Array<real_t> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
            const auto gpu_var = gpu_results.view().subregion(0).reshape(output_shape);
            const auto gpu_std = gpu_results.view().subregion(1).reshape(output_shape);
            math::var(gpu_data, gpu_var, ddof);
            math::std(gpu_data, gpu_std, ddof);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
        }
    }

    const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
    const auto cpu_sum = cpu_results.view().subregion(0).reshape(output_shape);
    const auto cpu_mean = cpu_results.view().subregion(1).reshape(output_shape);
    math::sum(cpu_data, cpu_sum);
    math::mean(cpu_data, cpu_mean);

    const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
    const auto gpu_sum = gpu_results.view().subregion(0).reshape(output_shape);
    const auto gpu_mean = gpu_results.view().subregion(1).reshape(output_shape);
    math::sum(gpu_data, gpu_sum);
    math::mean(gpu_data, gpu_mean);

    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
}

TEST_CASE("unified::math:: axis reductions vs numpy", "[assets][noa][unified]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto input_filename = path / input["path"].as<Path>();
    const auto shape = input["shape"].as<Shape4<i64>>();

    auto data = noa::io::load_data<f32>(input_filename);
    REQUIRE(all(data.shape() == shape));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t i = 0; i < 4; ++i) {
        INFO(i);
        const std::string key = string::format("axis{}", i);
        const auto output_path_min = path / tests[key]["output_min"].as<Path>();
        const auto output_path_max = path / tests[key]["output_max"].as<Path>();
        const auto output_path_sum = path / tests[key]["output_sum"].as<Path>();
        const auto output_path_mean = path / tests[key]["output_mean"].as<Path>();
        const auto output_path_var = path / tests[key]["output_var"].as<Path>();
        const auto output_path_std = path / tests[key]["output_std"].as<Path>();
        const auto output_shape = tests[key]["output_shape"].as < Shape4 < i64 >> ();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);
            data = device == data.device() ? data : data.to(options);

            const Array<f32> output(output_shape, options);

            auto expected = noa::io::load_data<f32>(output_path_min);
            math::min(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::load_data<f32>(output_path_max);
            math::max(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::load_data<f32>(output_path_sum);
            math::sum(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::load_data<f32>(output_path_mean);
            math::mean(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::load_data<f32>(output_path_var);
            math::var(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));

            expected = noa::io::load_data<f32>(output_path_std);
            math::std(data, output);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::math:: axis reductions, cpu vs gpu", "[noa][unified]", i64, f32, f64, c32, c64) {
    if (!Device::is_any(DeviceType::GPU))
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

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.elements(), randomizer);
    auto gpu_data = cpu_data.to(ArrayOption(Device("gpu"), Allocator::MANAGED));

    cpu_data = cpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
            noa::indexing::full_extent_t{},
            noa::indexing::slice_t{0, subregion_shape[1]},
            noa::indexing::slice_t{0, subregion_shape[2]},
            noa::indexing::slice_t{0, subregion_shape[3]});

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
            math::min(cpu_data, cpu_min);
            math::max(cpu_data, cpu_max);

            const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
            const auto gpu_min = gpu_results.view().subregion(0).reshape(output_shape);
            const auto gpu_max = gpu_results.view().subregion(1).reshape(output_shape);
            math::min(gpu_data, gpu_min);
            math::max(gpu_data, gpu_max);

            REQUIRE(test::Matcher(test::MATCH_ABS, cpu_results, gpu_results, 1e-8));
        }

        if constexpr (!noa::traits::is_int_v<TestType>){
            for (i64 ddof = 0; ddof < 2; ++ddof) {
                INFO("ddof: " << ddof);
                const Array<real_t> cpu_results({2, 1, 1, output_shape.elements()});
                const auto cpu_var = cpu_results.view().subregion(0).reshape(output_shape);
                const auto cpu_std = cpu_results.view().subregion(1).reshape(output_shape);
                math::var(cpu_data, cpu_var, ddof);
                math::std(cpu_data, cpu_std, ddof);

                const Array<real_t> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
                const auto gpu_var = gpu_results.view().subregion(0).reshape(output_shape);
                const auto gpu_std = gpu_results.view().subregion(1).reshape(output_shape);
                math::var(gpu_data, gpu_var, ddof);
                math::std(gpu_data, gpu_std, ddof);

                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
            }
        }

        const Array<TestType> cpu_results({2, 1, 1, output_shape.elements()});
        const auto cpu_sum = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_mean = cpu_results.view().subregion(1).reshape(output_shape);
        math::sum(cpu_data, cpu_sum);
        math::mean(cpu_data, cpu_mean);

        const Array<TestType> gpu_results({2, 1, 1, output_shape.elements()}, gpu_data.options());
        const auto gpu_sum = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_mean = gpu_results.view().subregion(1).reshape(output_shape);
        math::sum(gpu_data, gpu_sum);
        math::mean(gpu_data, gpu_mean);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_results, gpu_results, eps));
    }
}
