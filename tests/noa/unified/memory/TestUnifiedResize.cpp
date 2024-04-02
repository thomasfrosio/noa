#include <noa/unified/Array.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/Resize.hpp>
#include <noa/unified/Factory.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::memory::resize()", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "memory";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["resize"];
    const bool pad = GENERATE(false, true);
    INFO("pad=" << pad);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
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
            const auto input_shape = test["shape"].as<Shape4<i64>> ();
            const auto border_mode = test["border"].as<Border>();
            const auto border_value = test["border_value"].as<f32>();

            Shape4<i64> output_shape;
            Vec4<i64> left, right;
            if (is_centered) {
                output_shape = test["o_shape"].as<Shape4<i64>>();
            } else {
                left = test["left"].as<Vec4 <i64>>();
                right = test["right"].as<Vec4<i64>>();
                output_shape = Shape4<i64>(input_shape.vec + left + right);
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
                    noa::indexing::Ellipsis{},
                    noa::indexing::Slice{0, input_shape[1]},
                    noa::indexing::Slice{0, input_shape[2]},
                    noa::indexing::Slice{0, input_shape[3]});
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
                noa::io::save(output, expected_filename);
            } else {
                const auto expected = noa::io::load_data<f32>(expected_filename).reshape(output_shape);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, output, 1e-6));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::memory::resize() - edge cases", "[noa][unified]", i32, u32, i64, u64, f32, f64) {
    const i64 ndim = GENERATE(2, 3);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        AND_THEN("copy") {
            const auto shape = test::get_random_shape4_batched(ndim);
            const auto input = noa::random<TestType>(noa::math::uniform_t{}, shape, 0, 50, options);
            const auto output = noa::empty<TestType>(shape, options);
            noa::resize(input, output, Border::VALUE, TestType{0});
            REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-8));
        }
    }
}


TEMPLATE_TEST_CASE("unified::memory::resize(), borders", "[noa][unified]", i32, f32, f64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4(ndim);

    const auto border_mode = GENERATE(Border::VALUE, Border::PERIODIC);
    const auto border_value = TestType{2};
    const auto border_left = Vec4<i64>{1, 10, -5, 20};
    const auto border_right = Vec4<i64>{1, -3, 25, 41};
    const auto output_shape = Shape4<i64>(shape.vec + border_left + border_right);

    auto data = noa::random<TestType>(noa::Uniform{-5, 5}, shape);
    const auto expected = Array<TestType>(output_shape);
    noa::resize(data, expected, border_left, border_right, border_mode, border_value);
    expected.eval();

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, MemoryResource::MANAGED);
        INFO(device);

        if (device != data.device())
            data = data.to(options);

        const auto result = noa::resize(data, border_left, border_right, border_mode, border_value);
        REQUIRE(noa::all(result.shape() == output_shape));
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, result, 1e-8));
    }
}
