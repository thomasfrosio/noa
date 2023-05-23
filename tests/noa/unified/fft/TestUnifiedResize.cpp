#include <noa/unified/fft/Resize.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"
using namespace ::noa;

TEST_CASE("unified::fft::resize()", "[asset][noa][unified]") {
    const fs::path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["resize"];

    constexpr bool GENERATE_ASSETS = false;
    if constexpr (GENERATE_ASSETS) {
        for (const YAML::Node& node : tests["input"]) {
            const auto shape = node["shape"].as<Shape4<i64>>();
            const auto path_input = path / node["path"].as<Path>();
            const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape.rfft(), -128., 128.);
            noa::io::save(input, path_input);
        }
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (size_t i = 0; i < tests["tests"].size(); ++i) {
            const YAML::Node& test = tests["tests"][i];
            const auto filename_input = path / test["input"].as<Path>();
            const auto filename_expected = path / test["expected"].as<Path>();
            const auto shape_input = test["shape_input"].as<Shape4<i64>>();
            const auto shape_expected = test["shape_expected"].as<Shape4<i64>>();

            const auto input = noa::io::load_data<f32>(filename_input, false, options);
            const auto output = noa::memory::empty<f32>(shape_expected.rfft(), options);
            noa::fft::resize<fft::H2H>(input, shape_input, output, shape_expected);

            if constexpr (GENERATE_ASSETS) {
                noa::io::save(output, filename_expected);
            } else {
                const auto expected = noa::io::load_data<f32>(filename_expected, false, options);
                REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::resize and remap", "[noa][unified]", f32, f64, c32, c64) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto input_shape = test::get_random_shape4_batched(4);
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

        Array a0 = math::random<TestType>(math::uniform_t{}, input_shape.rfft(), -50, 50, options);
        Array a1 = fft::resize<fft::H2H>(a0, input_shape, output_shape);
        Array a2 = fft::remap(fft::H2HC, a1, output_shape);
        Array a3 = fft::resize<fft::HC2HC>(a2, output_shape, input_shape);
        Array a4 = fft::remap(fft::HC2H, a3, input_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, a0, a4, 5e-6));

        a0 = math::random<TestType>(math::uniform_t{}, input_shape, -50, 50, options);
        a1 = fft::resize<fft::F2F>(a0, input_shape, output_shape);
        a2 = fft::remap(fft::F2FC, a1, output_shape);
        a3 = fft::resize<fft::FC2FC>(a2, output_shape, input_shape);
        a4 = fft::remap(fft::FC2F, a3, input_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, a0, a4, 5e-6));
    }
}

#include "noa/unified/geometry/Transform.hpp"
#include "noa/unified/fft/Resize.hpp"
#include "noa/unified/fft/Transform.hpp"
#include "noa/unified/memory/Resize.hpp"

TEST_CASE("test fft rescale", "[.]") {
    const Path directory = test::NOA_DATA_PATH / "fft";
    const Array input = noa::io::load_data<f32>(directory / "input_rescale_shift.mrc");

    const auto height_width = input.shape().vec().filter(2, 3);

    const Vec2<f32> scaling_factor{0.5};
    const Vec2<f32> center = (height_width / 2).as<f32>();
    const Float33 inv_affine = noa::math::inverse(
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::scale(scaling_factor)) *
            noa::geometry::translate(-center));
    const Shape4<i64> rescaled_shape = Shape2<i64>(noa::math::ceil(height_width.as<f32>() * scaling_factor).as<i64>())
            .push_front(Vec2<i64>{1});

    // Rescale using linear transformation in real-space.
    const Array output = noa::memory::like(input);
    noa::geometry::transform_2d(input, output, inv_affine);
    const Array output_cropped = noa::memory::resize(output, rescaled_shape);
    noa::io::save(output_cropped, directory / "output_rescale_linear.mrc");

    // Rescale using Fourier cropping.
    const Array input_rfft = noa::fft::r2c(input);
    const Array cropped_rfft = noa::fft::resize<noa::fft::H2H>(input_rfft, input.shape(), rescaled_shape);
    const Array cropped = noa::fft::c2r(cropped_rfft, rescaled_shape);
    noa::io::save(cropped, directory / "output_rescale_cropped.mrc");
}
