#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/signal/Convolve.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("unified::signal::convolve()", "[asset]") {
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
            const auto data = noa::io::read_data<f32>(filename_input, {}, options);
            const auto expected = noa::io::read_data<f32>(filename_expected, {}, options);
            auto filter = noa::io::read_data<f32>(filename_filter, {}, options);
            if (filter.shape()[1] == 1 and filter.shape()[2] == 2) // for 1d case, the MRC file as an extra row to make it 2D.
                filter = filter.subregion(0, 0, 0, noa::indexing::FullExtent{});

            const auto result = noa::like(data);
            noa::signal::convolve(data, result, filter);
            REQUIRE(test::allclose_abs_safe(expected, result, 1e-5));
        }
    }
}

TEST_CASE("unified::signal::convolve_separable()", "[asset]") {
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
            const auto data = noa::io::read_data<f32>(filename_input, {}, options);
            const auto filter = noa::io::read_data<f32>(filename_filter, {}, options)
                    .subregion(0, 0, 0, noa::indexing::FullExtent{});
            const auto expected = noa::io::read_data<f32>(filename_expected, {}, options);

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
