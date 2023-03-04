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
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    WHEN("individual reduction") {
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
}

TEMPLATE_TEST_CASE("unified::math:: large reductions, cpu vs gpu", "[assets][noa][unified]",
                   i64, f32, f64, c32, c64) {
    if (!Device::any(DeviceType::GPU))
        return;

    const auto pad = GENERATE(true, false);
    const auto subregion_shape = Shape4<i64>{1, 300, 300, 300};
    auto shape = subregion_shape;
    if (pad) {
        shape[1] = 10;
        shape[2] = 11;
        shape[3] = 12;
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
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(1e-5) : static_cast<real_t>(1e-10);

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

//TEST_CASE("cpu::math::statistics() - batch", "[assets][noa][cpu][math]") {
//    const path_t path = test::NOA_DATA_PATH / "math";
//    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];
//
//    const YAML::Node& input = tests["input"];
//    const auto shape = input["shape"].as<size4_t>();
//    const auto output_shape = tests["batch"]["output_shape"].as<size4_t>();
//    const auto input_filename = path / input["path"].as<path_t>();
//    const auto output_filename = path / tests["batch"]["output_path"].as<path_t>();
//
//    const size4_t stride = shape.strides();
//    const size_t elements = shape.elements();
//    const size4_t output_stride = output_shape.strides();
//    const size_t output_elements = output_shape.elements();
//
//    io::MRCFile file(input_filename, io::READ);
//    cpu::memory::PtrHost<float> data(elements);
//    file.readAll(data.get());
//
//    const YAML::Node expected = YAML::LoadFile(output_filename);
//    std::vector<float> expected_max, expected_min, expected_mean, expected_std, expected_sum, expected_var;
//    for (size_t i = 0; i < expected.size(); i++) {
//        expected_max.emplace_back(expected[i]["max"].as<float>());
//        expected_min.emplace_back(expected[i]["min"].as<float>());
//        expected_mean.emplace_back(expected[i]["mean"].as<float>());
//        expected_std.emplace_back(expected[i]["std"].as<float>());
//        expected_sum.emplace_back(expected[i]["sum"].as<float>());
//        expected_var.emplace_back(expected[i]["var"].as<float>());
//    }
//
//    cpu::Stream stream(cpu::Stream::DEFAULT);
//    cpu::memory::PtrHost<float> results(output_elements * 6);
//    const auto mins = results.share();
//    const auto maxs = results.attach(results.get() + output_shape[0] * 1);
//    const auto sums = results.attach(results.get() + output_shape[0] * 2);
//    const auto means = results.attach(results.get() + output_shape[0] * 3);
//    const auto vars = results.attach(results.get() + output_shape[0] * 4);
//    const auto stds = results.attach(results.get() + output_shape[0] * 5);
//
//    cpu::math::min<float>(data.share(), stride, shape, mins, output_stride, output_shape, stream);
//    cpu::math::max<float>(data.share(), stride, shape, maxs, output_stride, output_shape, stream);
//    cpu::math::sum<float>(data.share(), stride, shape, sums, output_stride, output_shape, stream);
//    cpu::math::mean<float>(data.share(), stride, shape, means, output_stride, output_shape, stream);
//    cpu::math::var<float>(data.share(), stride, shape, vars, output_stride, output_shape, 0, stream);
//    cpu::math::std<float>(data.share(), stride, shape, stds, output_stride, output_shape, 0, stream);
//
//    for (uint batch = 0; batch < shape[0]; ++batch) {
//        REQUIRE_THAT(mins[batch], Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
//        REQUIRE_THAT(maxs[batch], Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
//        REQUIRE_THAT(sums[batch], Catch::WithinRel(expected_sum[batch]));
//        REQUIRE_THAT(means[batch], Catch::WithinRel(expected_mean[batch]));
//        REQUIRE_THAT(vars[batch], Catch::WithinRel(expected_var[batch]));
//        REQUIRE_THAT(stds[batch], Catch::WithinRel(expected_std[batch]));
//    }
//}
//
//TEST_CASE("cpu::math::statistics() - axes", "[assets][noa][cpu][math]") {
//    const path_t path = test::NOA_DATA_PATH / "math";
//    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];
//
//    const YAML::Node& input = tests["input"];
//    const auto input_path = path / input["path"].as<path_t>();
//    const auto shape = input["shape"].as<size4_t>();
//    const size4_t stride = shape.strides();
//    const size_t elements = shape.elements();
//
//    io::MRCFile file(input_path, io::READ);
//    cpu::memory::PtrHost<float> data(elements);
//    file.readAll(data.get());
//
//    cpu::Stream stream(cpu::Stream::DEFAULT);
//    for (size_t i = 0; i < 4; ++i) {
//        INFO(i);
//        std::string key = string::format("axis{}", i);
//        const auto output_path_min = path / tests[key]["output_min"].as<path_t>();
//        const auto output_path_max = path / tests[key]["output_max"].as<path_t>();
//        const auto output_path_sum = path / tests[key]["output_sum"].as<path_t>();
//        const auto output_path_mean = path / tests[key]["output_mean"].as<path_t>();
//        const auto output_path_var = path / tests[key]["output_var"].as<path_t>();
//        const auto output_path_std = path / tests[key]["output_std"].as<path_t>();
//        const auto output_shape = tests[key]["output_shape"].as<size4_t>();
//        const size4_t output_stride = output_shape.strides();
//        const size_t output_elements = output_shape.elements();
//
//        cpu::memory::PtrHost<float> output(output_elements);
//        cpu::memory::PtrHost<float> expected(output_elements);
//
//        file.open(output_path_min, io::READ);
//        file.readAll(expected.get());
//        cpu::math::min<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//
//        file.open(output_path_max, io::READ);
//        file.readAll(expected.get());
//        cpu::math::max<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//
//        file.open(output_path_sum, io::READ);
//        file.readAll(expected.get());
//        cpu::math::sum<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//
//        file.open(output_path_mean, io::READ);
//        file.readAll(expected.get());
//        cpu::math::mean<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//
//        file.open(output_path_var, io::READ);
//        file.readAll(expected.get());
//        cpu::math::var<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, 0, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//
//        file.open(output_path_std, io::READ);
//        file.readAll(expected.get());
//        cpu::math::std<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, 0, stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
//    }
//}
//
//TEST_CASE("cpu::math::statistics() - complex", "[assets][noa][cpu][math]") {
//    const path_t path = test::NOA_DATA_PATH / "math";
//    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_complex"];
//    const auto shape = tests["shape"].as<size4_t>();
//    const auto input_filename = path / tests["input_path"].as<path_t>();
//    const auto output_filename = path / tests["output_path"].as<path_t>();
//
//    const size4_t stride = shape.strides();
//    const size_t elements = shape.elements();
//
//    const YAML::Node expected = YAML::LoadFile(output_filename);
//    const cfloat_t expected_sum{expected["sum_real"].as<float>(),
//                                expected["sum_imag"].as<float>()};
//    const cfloat_t expected_mean{expected["mean_real"].as<float>(),
//                                 expected["mean_imag"].as<float>()};
//    const auto expected_std = expected["std"].as<float>();
//    const auto expected_var = expected["var"].as<float>();
//
//    io::MRCFile file(input_filename, io::READ);
//    cpu::memory::PtrHost<cfloat_t> data(elements);
//    file.readAll(data.get());
//
//    cpu::Stream stream(cpu::Stream::DEFAULT);
//
//    THEN("sum, mean") {
//        const auto complex_sum = cpu::math::sum<cfloat_t>(data.share(), stride, shape, stream);
//        const auto complex_mean = cpu::math::mean<cfloat_t>(data.share(), stride, shape, stream);
//
//        REQUIRE_THAT(complex_sum.real, Catch::WithinRel(expected_sum.real));
//        REQUIRE_THAT(complex_sum.imag, Catch::WithinRel(expected_sum.imag));
//        REQUIRE_THAT(complex_mean.real, Catch::WithinRel(expected_mean.real));
//        REQUIRE_THAT(complex_mean.imag, Catch::WithinRel(expected_mean.imag));
//    }
//
//    THEN("var, std") {
//        const float var = cpu::math::var<cfloat_t>(data.share(), stride, shape, 1, stream);
//        const float std = cpu::math::std<cfloat_t>(data.share(), stride, shape, 1, stream);
//        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
//        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
//    }
//}
