#include <noa/runtime/Array.hpp>
#include <noa/runtime/Random.hpp>
#include <noa/runtime/Resize.hpp>
#include <noa/runtime/Factory.hpp>

#include <noa/io/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEST_CASE("runtime::resize()", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "memory";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["resize"];
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

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
            const auto expected_filename = path_base / test["expected"].as<Path>();
            const auto is_centered = test["is_centered"].as<bool>();
            const auto input_shape = test["shape"].as<Shape4> ();
            const auto border_mode = test["border"].as<noa::Border>();
            const auto border_value = test["border_value"].as<f32>();

            Shape4 output_shape;
            Vec<isize, 4> left, right;
            if (is_centered) {
                output_shape = test["o_shape"].as<Shape4>();
            } else {
                left = test["left"].as<Vec<isize, 4>>();
                right = test["right"].as<Vec<isize, 4>>();
                output_shape = Shape{input_shape.vec + left + right};
            }

            auto padded_shape = input_shape;
            if (pad) {
                padded_shape[1] += 10;
                padded_shape[2] += 11;
                padded_shape[3] += 12;
            }

            // Initialize input and output:
            auto input = noa::zeros<f32>(padded_shape);
            input = input.subregion(
                Ellipsis{},
                Slice{0, input_shape[1]},
                Slice{0, input_shape[2]},
                Slice{0, input_shape[3]});
            noa::arange(input);
            input.eval();
            if (is_centered) {
                const auto center = input_shape / 2;
                for (i64 batch = 0; batch < input_shape[0]; ++batch)
                    input(batch, center[1], center[2], center[3]) = 0;
            }

            if (device != input.device())
                input = input.to(options);
            const auto output = noa::fill<f32>(output_shape, 2, options);

            // Test:
            if (is_centered)
                noa::resize(input, output, border_mode, border_value);
            else
                noa::resize(input, output, left, right, border_mode, border_value);

            if (COMPUTE_ASSETS) {
                noa::io::write_image(output, expected_filename);
            } else {
                const auto expected = noa::read_image<f32>(expected_filename).data.reshape(output_shape);
                REQUIRE(test::allclose_abs_safe(expected, output, 1e-6));
            }
        }
    }
}

TEMPLATE_TEST_CASE("runtime::resize() - edge cases", "", i32, u32, i64, u64, f32, f64) {
    const i64 ndim = GENERATE(2, 3);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        AND_THEN("copy") {
            const auto shape = test::random_shape_batched(ndim);
            const auto input = noa::random(noa::Uniform<TestType>{0, 50}, shape, options);
            const auto output = noa::empty<TestType>(shape, options);
            noa::resize(input, output, noa::Border::VALUE, 0);
            REQUIRE(test::allclose_abs(input, output, 1e-8));
        }
    }
}

TEMPLATE_TEST_CASE("runtime::resize(), borders", "", i32, f32, f64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::random_shape(ndim);

    const auto border_mode = GENERATE(noa::Border::VALUE, noa::Border::PERIODIC);
    const auto border_value = TestType{2};
    const auto border_left = Vec<isize, 4>{1, 10, -5, 20};
    const auto border_right = Vec<isize, 4>{1, -3, 25, 41};
    const auto output_shape = Shape{shape.vec + border_left + border_right};

    auto data = noa::random(noa::Uniform<TestType>{-5, 5}, shape);
    const auto expected = Array<TestType>(output_shape);
    noa::resize(data, expected, border_left, border_right, border_mode, border_value);
    expected.eval();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (device != data.device())
            data = data.to(options);

        const auto result = noa::resize(data, border_left, border_right, border_mode, border_value);
        REQUIRE(result.shape() == output_shape);
        REQUIRE(test::allclose_abs(expected, result, 1e-8));
    }
}
