#include <noa/runtime/Factory.hpp>
#include <noa/runtime/Random.hpp>

#include <noa/io/IO.hpp>
#include <noa/signal/Convolve.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("signal::convolve()", "[asset]") {
    using namespace noa::types;

    const Path path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve"]["tests"];

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
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto filename_filter = path_base / test["filter"].as<Path>();
            const auto filename_expected = path_base / test["expected"].as<Path>();

            // Inputs:
            const auto data = noa::read_image<f32>(filename_input, {}, options).data;
            const auto expected = noa::read_image<f32>(filename_expected, {}, options).data;
            auto filter = noa::read_image<f32>(filename_filter, {}, options).data;
            if (filter.shape()[1] == 1 and filter.shape()[2] == 2) // for 1d case, the MRC file as an extra row to make it 2D.
                filter = filter.subregion(0, 0, 0, Full{});

            const auto result = noa::like(data);
            noa::signal::convolve(data, result, filter);
            REQUIRE(test::allclose_abs_safe(expected, result, 1e-5));
        }
    }
}

TEST_CASE("signal::convolve_separable()", "[asset]") {
    using namespace noa::types;

    const Path path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["convolve_separable"]["tests"];

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
            const auto filename_input = path_base / test["input"].as<Path>();
            const auto filename_filter = path_base / test["filter"].as<Path>();
            const auto filename_expected = path_base / test["expected"].as<Path>();
            const auto dim = test["dim"].as<std::vector<i32>>();

            // Input
            const auto data = noa::read_image<f32>(filename_input, {}, options).data;
            const auto filter = noa::read_image<f32>(filename_filter, {}, options).data.subregion(0, 0, 0, Full{});
            const auto expected = noa::read_image<f32>(filename_expected, {}, options).data;

            View<const f32> filter0;
            View<const f32> filter1;
            View<const f32> filter2;
            for (auto i: dim) {
                if (i == 0)
                    filter0 = filter.view();
                if (i == 1)
                    filter1 = filter.view();
                if (i == 2)
                    filter2 = filter.view();
            }

            const auto result = noa::like(data);
            noa::signal::convolve_separable(data, result, filter0, filter1, filter2);
            REQUIRE(test::allclose_abs_safe(expected, result, 1e-5));
        }
    }
}
