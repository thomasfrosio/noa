#include <noa/unified/Array.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("unified::reduce - axis reductions vs numpy", "[assets]") {
    const auto path = test::NOA_DATA_PATH / "math";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["reduce_to_stats"];

    const YAML::Node& input = tests["input"];
    const auto input_filename = path / input["path"].as<Path>();
    const auto shape = input["shape"].as<Shape4<i64>>();

    auto data = noa::io::read_data<f32>(input_filename);
    REQUIRE(noa::all(data.shape() == shape));

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
        const auto output_shape = tests[key]["output_shape"].as<Shape4<i64>>();

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, "managed");
            INFO(device);
            data = device == data.device() ? data : data.to(options);

            const Array<f32> output(output_shape, options);

            auto expected = noa::io::read_data<f32>(output_path_min);
            noa::min(data, output);
            REQUIRE(test::allclose_abs_safe(expected, output, 1e-5));

            expected = noa::io::read_data<f32>(output_path_max);
            noa::max(data, output);
            REQUIRE(test::allclose_abs_safe(expected, output, 1e-5));

            expected = noa::io::read_data<f32>(output_path_sum);
            noa::sum(data, output);
            REQUIRE(test::allclose_abs_safe(expected.release(), output, 1e-5));

            const auto expected_std = noa::io::read_data<f32>(output_path_std);
            noa::stddev(data, output);
            REQUIRE(test::allclose_abs_safe(expected_std, output, 1e-5));

            const auto expected_mean = noa::io::read_data<f32>(output_path_mean);
            noa::mean(data, output);
            REQUIRE(test::allclose_abs_safe(expected_mean, output, 1e-5));

            const auto expected_norm = noa::io::read_data<f32>(output_path_norm);
            noa::l2_norm(data, output);
            REQUIRE(test::allclose_abs_safe(expected_norm, output, 1e-5));

            const auto expected_variance = noa::io::read_data<f32>(output_path_var);
            noa::variance(data, output);
            REQUIRE(test::allclose_abs_safe(expected_variance, output, 1e-5));

            const Array<f32> mean(output_shape, options);
            noa::mean_variance(data, mean, output);
            REQUIRE(test::allclose_abs_safe(expected_mean, mean, 1e-5));
            REQUIRE(test::allclose_abs_safe(expected_variance, output, 1e-5));

            noa::mean_stddev(data, mean, output);
            REQUIRE(test::allclose_abs_safe(expected_mean, mean, 1e-5));
            REQUIRE(test::allclose_abs_safe(expected_std, output, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::reduce - axis reductions, cpu vs gpu", "[noa]", i64, f32, f64, c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const auto pad = GENERATE(true, false);
    const auto large = GENERATE(true, false);
    const auto subregion_shape =
        test::random_shape(3, {.batch_range = {2, 10}}) +
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
    test::randomize(cpu_data.get(), cpu_data.n_elements(), randomizer);
    auto gpu_data = cpu_data.to(ArrayOption("gpu", "managed"));

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

    // The CPU backend uses double precision + the Kahan summation, as opposed to the GPU which is a simple
    // parallel sum in using the precision of the input, hence the 4e-4 epsilon for f32 and c32.
    using real_t = noa::traits::value_type_t<TestType>;
    const real_t eps = std::is_same_v<real_t, f32> ? static_cast<real_t>(4e-4) : static_cast<real_t>(1e-10);

    for (i64 axis = 0; axis < 4; ++axis) {
        INFO("axis: " << axis << ", shape=" << subregion_shape);
        auto output_shape = subregion_shape;
        output_shape[axis] = 1;

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
            auto cpu_norm = noa::l2_norm(cpu_data, noa::ReduceAxes::from_shape(output_shape));
            auto gpu_norm = noa::l2_norm(gpu_data, noa::ReduceAxes::from_shape(output_shape));
            REQUIRE(test::allclose_abs_safe(std::move(cpu_norm), std::move(gpu_norm), eps));

            for (i64 ddof = 0; ddof < 2; ++ddof) {
                INFO("ddof: " << ddof);
                const Array<real_t> cpu_results({2, 1, 1, output_shape.n_elements()});
                const auto cpu_var = cpu_results.view().subregion(0).reshape(output_shape);
                const auto cpu_std = cpu_results.view().subregion(1).reshape(output_shape);
                noa::variance(cpu_data, cpu_var, ddof);
                noa::stddev(cpu_data, cpu_std, ddof);

                const Array<real_t> gpu_results({2, 1, 1, output_shape.n_elements()}, gpu_data.options());
                const auto gpu_var = gpu_results.view().subregion(0).reshape(output_shape);
                const auto gpu_std = gpu_results.view().subregion(1).reshape(output_shape);
                noa::variance(gpu_data, gpu_var, ddof);
                noa::stddev(gpu_data, gpu_std, ddof);

                REQUIRE(test::allclose_abs_safe(cpu_results, gpu_results, eps));

                const Array<TestType> cpu_mean(output_shape);
                const Array<real_t> cpu_variance(output_shape);
                const Array<TestType> gpu_mean(output_shape, gpu_data.options());
                const Array<real_t> gpu_variance(output_shape, gpu_data.options());

                noa::mean_variance(cpu_data, cpu_mean, cpu_variance);
                noa::mean_variance(gpu_data, gpu_mean, gpu_variance);
                REQUIRE(test::allclose_abs_safe(cpu_mean, gpu_mean, eps));
                REQUIRE(test::allclose_abs_safe(cpu_variance, gpu_variance, eps));

                noa::mean_stddev(cpu_data, cpu_mean, cpu_variance);
                noa::mean_stddev(gpu_data, gpu_mean, gpu_variance);
                REQUIRE(test::allclose_abs_safe(cpu_mean, gpu_mean, eps));
                REQUIRE(test::allclose_abs_safe(cpu_variance, gpu_variance, eps));
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
}

TEST_CASE("unified::reduce - argmax/argmin()", "[noa]") {
    const bool small = GENERATE(true, false);
    const auto shape = small ? Shape4<i64>{3, 8, 41, 65} : Shape4<i64>{3, 256, 256, 300};
    const auto n_elements_per_batch = shape.pop_front().as<u32>().n_elements();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    SECTION("reduce entire array") {
        Array<f32> input(shape);
        test::randomize(input.get(), input.n_elements(), test::Randomizer<f32>(-100., 100.));

        auto expected_min_offset = test::Randomizer<u32>(0, input.n_elements() - 2).get();
        auto expected_max_offset = test::Randomizer<u32>(0, input.n_elements() - 2).get();
        if (expected_min_offset == expected_max_offset)
            expected_max_offset += 1;

        input.span_1d()[expected_min_offset] = -101;
        input.span_1d()[expected_max_offset] = 101;
        for (auto& device: devices) {
            INFO(device);
            const auto options = ArrayOption{device, "managed"};
            input = input.device().is_cpu() ? std::move(input) : std::move(input).to(options);

            const auto [min, min_offset] = noa::argmin(input);
            REQUIRE((min == -101 and min_offset == expected_min_offset));
            const auto [max, max_offset] = noa::argmax(input);
            REQUIRE((max == 101 and max_offset == expected_max_offset));
        }
    }

    SECTION("per batch") {
        Array<f32> input(shape);
        test::randomize(input.get(), input.n_elements(), test::Randomizer<f32>(-100., 100.));

        Array expected_min_values = noa::empty<f32>({shape.batch(), 1, 1, 1});
        Array expected_min_offsets = noa::like<i32>(expected_min_values);

        test::randomize(expected_min_values.get(), expected_min_values.n_elements(),
                        test::Randomizer<f32>(-200, -101));
        test::randomize(expected_min_offsets.get(), expected_min_offsets.n_elements(),
                        test::Randomizer<i32>(0u, n_elements_per_batch - 1));

        const auto input_2d = input.reshape({shape.batch(), 1, 1, -1});
        for (i64 batch = 0; batch < shape.batch(); ++batch) {
            auto& offset = expected_min_offsets(batch, 0, 0, 0);
            input_2d(batch, 0, 0, offset) = expected_min_values(batch, 0, 0, 0);
            offset += static_cast<i32>(batch * n_elements_per_batch); // back to global offset
        }

        for (auto& device: devices) {
            INFO(device);
            const auto options = ArrayOption{device, "managed"};
            Array min_values = noa::empty<f32>(expected_min_values.shape(), options);
            Array min_offsets = noa::empty<i32>(expected_min_values.shape(), options);

            if (input.device() != device)
                input = std::move(input).to(options);

            noa::argmin(input, min_values, min_offsets);
            REQUIRE(test::allclose_abs(min_values, expected_min_values, 1e-7));
            REQUIRE(test::allclose_abs(min_offsets, expected_min_offsets, 1e-7));

            noa::fill(min_values, {});
            noa::argmin(input, min_values, {});
            REQUIRE(test::allclose_abs(min_values, expected_min_values, 1e-7));

            noa::fill(min_offsets, {});
            noa::argmin(input, {}, min_offsets);
            REQUIRE(test::allclose_abs(min_offsets, expected_min_offsets, 1e-7));
        }
    }
}
