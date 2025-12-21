#include <noa/runtime/Random.hpp>
#include <noa/runtime/Factory.hpp>
#include <noa/io/IO.hpp>

#include <noa/fft/core/Layout.hpp>
#include <noa/fft/Resize.hpp>
#include <noa/fft/Remap.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nf = noa::fft;
namespace fs = std::filesystem;

TEST_CASE("fft::resize()", "[asset]") {
    const fs::path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["resize"];

    constexpr bool GENERATE_ASSETS = false;
    if constexpr (GENERATE_ASSETS) {
        for (const YAML::Node& node : tests["input"]) {
            const auto shape = node["shape"].as<Shape4>();
            const auto path_input = path / node["path"].as<fs::path>();
            const auto input = noa::random(noa::Uniform{-128.f, 128.f}, shape.rfft());
            noa::write_image(input, path_input);
        }
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (size_t i = 0; i < tests["tests"].size(); ++i) {
            const YAML::Node& test = tests["tests"][i];
            const auto filename_input = path / test["input"].as<fs::path>();
            const auto filename_expected = path / test["expected"].as<fs::path>();
            const auto shape_input = test["shape_input"].as<Shape4>();
            const auto shape_expected = test["shape_expected"].as<Shape4>();

            const auto input = noa::read_image<f32>(filename_input, {}, options).data;
            const auto output = noa::empty<f32>(shape_expected.rfft(), options);
            nf::resize<"h2h">(input, shape_input, output, shape_expected);

            if constexpr (GENERATE_ASSETS) {
                noa::io::write_image(output, filename_expected);
            } else {
                const auto expected = noa::read_image<f32>(filename_expected, {}, options).data;
                REQUIRE(test::allclose_abs(expected, output, 1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE("fft::resize and remap", "", f32, f64, c32, c64) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto input_shape = test::random_shape_batched(4);
    auto output_shape = input_shape;

    test::Randomizer<i64> randomizer(0, 20);
    output_shape[1] += randomizer.get();
    output_shape[2] += randomizer.get();
    output_shape[3] += randomizer.get();

    INFO(input_shape);
    INFO(output_shape);

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        Array a0 = noa::random(noa::Uniform<TestType>{-50, 50}, input_shape.rfft(), options);
        Array a1 = nf::resize<"h2h">(a0, input_shape, output_shape);
        Array a2 = nf::remap(nf::Layout::H2HC, a1, output_shape);
        Array a3 = nf::resize<"hc2hc">(a2, output_shape, input_shape);
        Array a4 = nf::remap(nf::Layout::HC2H, a3, input_shape);
        REQUIRE(test::allclose_abs_safe(a0, a4, 5e-6));

        a0 = noa::random(noa::Uniform<TestType>{-50, 50}, input_shape, options);
        a1 = nf::resize<"f2f">(a0, input_shape, output_shape);
        a2 = nf::remap(nf::Layout::F2FC, a1, output_shape);
        a3 = nf::resize<"fc2fc">(a2, output_shape, input_shape);
        a4 = nf::remap(nf::Layout::FC2F, a3, input_shape);
        REQUIRE(test::allclose_abs_safe(a0, a4, 5e-6));
    }
}
