#include <noa/common/io/ImageFile.h>
#include <noa/common/io/TextFile.h>
#include <noa/common/string/Parse.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reduce.h>

#include "Helpers.h"
#include "Assets.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::math::statistics() - all", "[assets][noa][cpu][math]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<size4_t>();
    const auto input_filename = path / input["path"].as<path_t>();
    const auto output_filename = path / tests["all"]["output_path"].as<path_t>();

    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    io::ImageFile file(input_filename, io::READ);
    cpu::memory::PtrHost<float> data(elements);
    file.readAll(data.get());

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const auto expected_max = expected["max"].as<float>();
    const auto expected_min = expected["min"].as<float>();
    const auto expected_mean = expected["mean"].as<float>();
    const auto expected_std = expected["std"].as<float>();
    const auto expected_sum = expected["sum"].as<float>();
    const auto expected_var = expected["var"].as<float>();

    cpu::Stream stream(cpu::Stream::DEFAULT);

    WHEN("individual reduction") {
        const auto min = cpu::math::min<float>(data.share(), stride, shape, stream);
        const auto max = cpu::math::max<float>(data.share(), stride, shape, stream);
        const auto sum = cpu::math::sum<float>(data.share(), stride, shape, stream);
        const auto mean = cpu::math::mean<float>(data.share(), stride, shape, stream);
        const auto var = cpu::math::var<0, float>(data.share(), stride, shape, stream);
        const auto std = cpu::math::std<0, float>(data.share(), stride, shape, stream);

        REQUIRE_THAT(min, Catch::WithinAbs(static_cast<double>(expected_min), 1e-6));
        REQUIRE_THAT(max, Catch::WithinAbs(static_cast<double>(expected_max), 1e-6));
        REQUIRE_THAT(sum, Catch::WithinRel(expected_sum));
        REQUIRE_THAT(mean, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
    }

    WHEN("statistics") {
        const auto [sum, mean, var, std] = cpu::math::statistics<0, float>(data.share(), stride, shape, stream);
        REQUIRE_THAT(sum, Catch::WithinRel(expected_sum));
        REQUIRE_THAT(mean, Catch::WithinRel(expected_mean));
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
    }
}

TEST_CASE("cpu::math::statistics() - batch", "[assets][noa][cpu][math]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<size4_t>();
    const auto output_shape = tests["batch"]["output_shape"].as<size4_t>();
    const auto input_filename = path / input["path"].as<path_t>();
    const auto output_filename = path / tests["batch"]["output_path"].as<path_t>();

    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size4_t output_stride = output_shape.stride();
    const size_t output_elements = output_shape.elements();

    io::ImageFile file(input_filename, io::READ);
    cpu::memory::PtrHost<float> data(elements);
    file.readAll(data.get());

    const YAML::Node expected = YAML::LoadFile(output_filename);
    std::vector<float> expected_max, expected_min, expected_mean, expected_std, expected_sum, expected_var;
    for (size_t i = 0; i < expected.size(); i++) {
        expected_max.emplace_back(expected[i]["max"].as<float>());
        expected_min.emplace_back(expected[i]["min"].as<float>());
        expected_mean.emplace_back(expected[i]["mean"].as<float>());
        expected_std.emplace_back(expected[i]["std"].as<float>());
        expected_sum.emplace_back(expected[i]["sum"].as<float>());
        expected_var.emplace_back(expected[i]["var"].as<float>());
    }

    cpu::Stream stream(cpu::Stream::DEFAULT);
    cpu::memory::PtrHost<float> results(output_elements * 6);
    const auto mins = results.share();
    const auto maxs = results.attach(results.get() + output_shape[0] * 1);
    const auto sums = results.attach(results.get() + output_shape[0] * 2);
    const auto means = results.attach(results.get() + output_shape[0] * 3);
    const auto vars = results.attach(results.get() + output_shape[0] * 4);
    const auto stds = results.attach(results.get() + output_shape[0] * 5);

    cpu::math::min<float>(data.share(), stride, shape, mins, output_stride, output_shape, stream);
    cpu::math::max<float>(data.share(), stride, shape, maxs, output_stride, output_shape, stream);
    cpu::math::sum<float>(data.share(), stride, shape, sums, output_stride, output_shape, stream);
    cpu::math::mean<float>(data.share(), stride, shape, means, output_stride, output_shape, stream);
    cpu::math::var<0, float>(data.share(), stride, shape, vars, output_stride, output_shape, stream);
    cpu::math::std<0, float>(data.share(), stride, shape, stds, output_stride, output_shape, stream);

    for (uint batch = 0; batch < shape[0]; ++batch) {
        REQUIRE_THAT(mins[batch], Catch::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
        REQUIRE_THAT(maxs[batch], Catch::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
        REQUIRE_THAT(sums[batch], Catch::WithinRel(expected_sum[batch]));
        REQUIRE_THAT(means[batch], Catch::WithinRel(expected_mean[batch]));
        REQUIRE_THAT(vars[batch], Catch::WithinRel(expected_var[batch]));
        REQUIRE_THAT(stds[batch], Catch::WithinRel(expected_std[batch]));
    }
}

TEST_CASE("cpu::math::statistics() - axes", "[assets][noa][cpu][math]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto input_path = path / input["path"].as<path_t>();
    const auto shape = input["shape"].as<size4_t>();
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    io::ImageFile file(input_path, io::READ);
    cpu::memory::PtrHost<float> data(elements);
    file.readAll(data.get());

    cpu::Stream stream(cpu::Stream::DEFAULT);
    for (size_t i = 0; i < 4; ++i) {
        INFO(i);
        std::string key = string::format("axis{}", i);
        const auto output_path_min = path / tests[key]["output_min"].as<path_t>();
        const auto output_path_max = path / tests[key]["output_max"].as<path_t>();
        const auto output_path_sum = path / tests[key]["output_sum"].as<path_t>();
        const auto output_path_mean = path / tests[key]["output_mean"].as<path_t>();
        const auto output_path_var = path / tests[key]["output_var"].as<path_t>();
        const auto output_path_std = path / tests[key]["output_std"].as<path_t>();
        const auto output_shape = tests[key]["output_shape"].as<size4_t>();
        const size4_t output_stride = output_shape.stride();
        const size_t output_elements = output_shape.elements();

        cpu::memory::PtrHost<float> output(output_elements);
        cpu::memory::PtrHost<float> expected(output_elements);

        file.open(output_path_min, io::READ);
        file.readAll(expected.get());
        cpu::math::min<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));

        file.open(output_path_max, io::READ);
        file.readAll(expected.get());
        cpu::math::max<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));

        file.open(output_path_sum, io::READ);
        file.readAll(expected.get());
        cpu::math::sum<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));

        file.open(output_path_mean, io::READ);
        file.readAll(expected.get());
        cpu::math::mean<float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));

        file.open(output_path_var, io::READ);
        file.readAll(expected.get());
        cpu::math::var<0, float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));

        file.open(output_path_std, io::READ);
        file.readAll(expected.get());
        cpu::math::std<0, float>(data.share(), stride, shape, output.share(), output_stride, output_shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-6));
    }
}

TEST_CASE("cpu::math::statistics() - complex", "[assets][noa][cpu][math]") {
    const path_t path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_complex"];
    const auto shape = tests["shape"].as<size4_t>();
    const auto input_filename = path / tests["input_path"].as<path_t>();
    const auto output_filename = path / tests["output_path"].as<path_t>();

    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    const YAML::Node expected = YAML::LoadFile(output_filename);
    const cfloat_t expected_sum{expected["sum_real"].as<float>(),
                                 expected["sum_imag"].as<float>()};
    const cfloat_t expected_mean{expected["mean_real"].as<float>(),
                                 expected["mean_imag"].as<float>()};
    const auto expected_std = expected["std"].as<float>();
    const auto expected_var = expected["var"].as<float>();

    io::ImageFile file(input_filename, io::READ);
    cpu::memory::PtrHost<cfloat_t> data(elements);
    file.readAll(data.get());

    cpu::Stream stream(cpu::Stream::DEFAULT);

    THEN("sum, mean") {
        const cfloat_t complex_sum = cpu::math::sum<cfloat_t>(data.share(), stride, shape, stream);
        const cfloat_t complex_mean = cpu::math::mean<cfloat_t>(data.share(), stride, shape, stream);

        REQUIRE_THAT(complex_sum.real, Catch::WithinRel(expected_sum.real));
        REQUIRE_THAT(complex_sum.imag, Catch::WithinRel(expected_sum.imag));
        REQUIRE_THAT(complex_mean.real, Catch::WithinRel(expected_mean.real));
        REQUIRE_THAT(complex_mean.imag, Catch::WithinRel(expected_mean.imag));
    }

    THEN("var, std") {
        const float var = cpu::math::var<1, cfloat_t>(data.share(), stride, shape, stream);
        const float std = cpu::math::std<1, cfloat_t>(data.share(), stride, shape, stream);
        REQUIRE_THAT(var, Catch::WithinRel(expected_var));
        REQUIRE_THAT(std, Catch::WithinRel(expected_std));
    }
}
