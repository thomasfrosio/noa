#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/signal/Median.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace noa;

TEST_CASE("unified::signal::median_filter()", "[asset][noa][unified]") {
    const Path path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["median"]["tests"];

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        for (size_t nb = 0; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto window = test["window"].as<i64>();
            const auto dim = test["dim"].as<i32>();
            const auto border = test["border"].as<BorderMode>();
            const auto filename_expected = path_base / test["expected"].as<Path>();

            const auto input = noa::io::load_data<f32>(filename_input, false, options);
            const auto expected = noa::io::load_data<f32>(filename_expected, false, options);

            const auto result = noa::memory::like(input);
            if (dim == 1)
                noa::signal::median_filter_1d(input, result, window, border);
            else if (dim == 2)
                noa::signal::median_filter_2d(input, result, window, border);
            else if (dim == 3)
                noa::signal::median_filter_3d(input, result, window, border);
            else
                FAIL("dim is not correct");

            REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::median_filter(), cpu vs gpu", "[noa][unified]", i32, f16, f32, f64) {
    if (!Device::is_any(DeviceType::GPU))
        return;

    const i64 ndim = GENERATE(1, 2, 3);
    const BorderMode mode = GENERATE(BorderMode::ZERO, BorderMode::REFLECT);
    i64 window = test::Randomizer<i64>(2, 11).get();
    if (!(window % 2))
        window -= 1;
    if (ndim == 3 && window > 5)
        window = 3;

    auto shape = test::get_random_shape4_batched(3);
    if (ndim != 3 && test::Randomizer<i64>(16, 100).get() % 2)
        shape[1] = 1; // randomly switch to 2d
    INFO(string::format("ndim:{}, mode:{}, window:{}, shape:{}", ndim, mode, window, shape));

    const auto options = ArrayOption(Device{"gpu"}, Allocator::PITCHED);
    const auto cpu_data = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -128, 128);
    const auto gpu_data = cpu_data.to(options);

    const auto cpu_result = noa::memory::like(cpu_data);
    const auto gpu_result = noa::memory::like(gpu_data);

    if (ndim == 1) {
        noa::signal::median_filter_1d(cpu_data, cpu_result, window, mode);
        noa::signal::median_filter_1d(gpu_data, gpu_result, window, mode);
    } else if (ndim == 2) {
        noa::signal::median_filter_2d(cpu_data, cpu_result, window, mode);
        noa::signal::median_filter_2d(gpu_data, gpu_result, window, mode);
    } else {
        noa::signal::median_filter_3d(cpu_data, cpu_result, window, mode);
        noa::signal::median_filter_3d(gpu_data, gpu_result, window, mode);
    }

    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_result, gpu_result.to_cpu(), 1e-5));
}
