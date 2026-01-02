#include <noa/runtime/Array.hpp>
#include <noa/runtime/Reduce.hpp>
#include <noa/runtime/Factory.hpp>

#include <noa/io/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("runtime::reduce - batched reductions vs numpy", "[asset]") {
    const auto path = test::NOA_DATA_PATH / "runtime";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto shape = input["shape"].as<Shape4>();
    const auto output_shape = tests["batch"]["output_shape"].as<Shape4>();
    const auto input_filename = path / input["path"].as<Path>();
    const auto output_filename = path / tests["batch"]["output_path"].as<Path>();

    auto data = noa::read_image<f64>(input_filename).data;
    REQUIRE(data.shape() == shape);

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
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const Array<f32> results({7, 1, 1, output_shape.n_elements()}, options);
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

        for (u32 batch = 0; batch < shape[0]; ++batch) {
            REQUIRE_THAT(mins.span_1d()[batch], Catch::Matchers::WithinAbs(static_cast<double>(expected_min[batch]), 1e-6));
            REQUIRE_THAT(maxs.span_1d()[batch], Catch::Matchers::WithinAbs(static_cast<double>(expected_max[batch]), 1e-6));
            REQUIRE_THAT(sums.span_1d()[batch], Catch::Matchers::WithinRel(expected_sum[batch]));
            REQUIRE_THAT(means.span_1d()[batch], Catch::Matchers::WithinRel(expected_mean[batch]));
            REQUIRE_THAT(norms.span_1d()[batch], Catch::Matchers::WithinRel(expected_norm[batch]));
            REQUIRE_THAT(vars.span_1d()[batch], Catch::Matchers::WithinRel(expected_var[batch]));
            REQUIRE_THAT(stds.span_1d()[batch], Catch::Matchers::WithinRel(expected_std[batch]));
        }
    }
}

TEMPLATE_TEST_CASE("runtime::reduce - batched reductions, cpu vs gpu", "", i64, f32, f64, c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const auto pad = GENERATE(true, false);
    const auto large = GENERATE(true, false);
    const auto subregion_shape =
        test::random_shape_batched(3) +
        (large ? Shape4{1, 164, 164, 164} : Shape4{});
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }
    INFO("padded=" << pad);
    INFO("large=" << large);
    INFO("subregion_shape=" << subregion_shape);

    Array<TestType> cpu_data(shape);
    test::Randomizer<TestType> randomizer(-50, 50);
    test::randomize(cpu_data.get(), cpu_data.n_elements(), randomizer);
    auto gpu_data = cpu_data.to(ArrayOption("gpu", "managed"));

    cpu_data = cpu_data.subregion(
        Full{},
        Slice{0, subregion_shape[1]},
        Slice{0, subregion_shape[2]},
        Slice{0, subregion_shape[3]});
    gpu_data = gpu_data.subregion(
        Full{},
        Slice{0, subregion_shape[1]},
        Slice{0, subregion_shape[2]},
        Slice{0, subregion_shape[3]});

    using real_t = noa::traits::value_type_t<TestType>;
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(1e-4) : static_cast<real_t>(1e-10);
    const auto output_shape = Shape4{subregion_shape.batch(), 1, 1, 1};

    if constexpr (not noa::traits::complex<TestType>) {
        const Array<TestType> cpu_results({2, 1, 1, output_shape.n_elements()});
        const auto cpu_min = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_max = cpu_results.view().subregion(1).reshape(output_shape);
        noa::min(cpu_data, cpu_min);
        noa::max(cpu_data, cpu_max);

        const Array<TestType> gpu_results({2, 1, 1, output_shape.n_elements()}, gpu_data.options());
        const auto gpu_min = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_max = gpu_results.view().subregion(1).reshape(output_shape);
        noa::min(gpu_data, gpu_min);
        noa::max(gpu_data, gpu_max);

        REQUIRE(test::allclose_abs(cpu_results, gpu_results, 1e-8));
    }

    if constexpr (not noa::traits::integer<TestType>) {
        const Array<real_t> cpu_results({3, 1, 1, output_shape.n_elements()});
        const auto cpu_norm = cpu_results.view().subregion(0).reshape(output_shape);
        const auto cpu_var = cpu_results.view().subregion(1).reshape(output_shape);
        const auto cpu_std = cpu_results.view().subregion(2).reshape(output_shape);

        const Array<real_t> gpu_results({3, 1, 1, output_shape.n_elements()}, gpu_data.options());
        const auto gpu_norm = gpu_results.view().subregion(0).reshape(output_shape);
        const auto gpu_var = gpu_results.view().subregion(1).reshape(output_shape);
        const auto gpu_std = gpu_results.view().subregion(2).reshape(output_shape);

        noa::l2_norm(cpu_data, cpu_norm);
        noa::l2_norm(gpu_data, gpu_norm);
        REQUIRE(test::allclose_abs_safe(cpu_norm.to_cpu(), gpu_norm.to_cpu(), eps));

        for (i32 ddof = 0; ddof < 2; ++ddof) {
            INFO("ddof: " << ddof);
            noa::variance(cpu_data, cpu_var, {.ddof = ddof});
            noa::stddev(cpu_data, cpu_std, {.ddof = ddof});

            noa::variance(gpu_data, gpu_var, {.ddof = ddof});
            noa::stddev(gpu_data, gpu_std, {.ddof = ddof});

            REQUIRE(test::allclose_abs_safe(cpu_results, gpu_results, eps));
        }
    }

    const Array<TestType> cpu_results({2, 1, 1, output_shape.n_elements()});
    const auto cpu_sum = cpu_results.view().subregion(0).reshape(output_shape);
    const auto cpu_mean = cpu_results.view().subregion(1).reshape(output_shape);
    noa::sum(cpu_data, cpu_sum);
    noa::mean(cpu_data, cpu_mean);

    const Array<TestType> gpu_results({2, 1, 1, output_shape.n_elements()}, gpu_data.options());
    const auto gpu_sum = gpu_results.view().subregion(0).reshape(output_shape);
    const auto gpu_mean = gpu_results.view().subregion(1).reshape(output_shape);
    noa::sum(gpu_data, gpu_sum);
    noa::mean(gpu_data, gpu_mean);

    REQUIRE(test::allclose_abs_safe(cpu_results, gpu_results, eps));
}
