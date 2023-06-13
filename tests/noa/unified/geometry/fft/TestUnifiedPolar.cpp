#include <noa/unified/geometry/fft/Polar.hpp>
#include <noa/unified/geometry/fft/Shape.hpp>
#include <noa/unified/geometry/Polar.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace noa;

TEST_CASE("unified::geometry::rotational_average, assets", "[noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["polar"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const auto shape = Shape4<i64>{1};
        const auto input = noa::memory::zeros<f32>(shape);
        noa::geometry::fft::rotational_average<fft::H2H>(input, input, shape);

    }
}

TEST_CASE("unified::geometry::fft::rotational_average", "[noa][unified]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const i64 size = GENERATE(64, 65);

    const i64 batches = 3;
    const auto shape = Shape4<i64>{batches, 1, size, size};
    const auto rotational_average_size = noa::math::min(shape.filter(2, 3)) / 2 + 1;
    const auto polar_shape = Shape4<i64>{batches, 1, 256, rotational_average_size};
    const auto rotational_average_shape = Shape4<i64>{batches, 1, 1, rotational_average_size};
    const auto vec = shape.vec().filter(2, 3).as<f32>();
    const auto center = vec / 2;

    for (auto device: devices) {
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const auto input = noa::memory::zeros<f32>(shape, options);
        noa::geometry::fft::sphere<fft::FC2FC>({}, input, center, 0.f, noa::math::min(center));

        // Rotational average using polar transformation.
        const auto input_rfft = noa::fft::remap(fft::FC2HC, input, shape);
        const auto polar = noa::memory::zeros<f32>(polar_shape, options);
        noa::geometry::fft::cartesian2polar<fft::HC2FC>(input_rfft, shape, polar);
        const auto polar_reduced = noa::memory::zeros<f32>(rotational_average_shape, options);
        noa::math::mean(polar, polar_reduced);

        // Rotational average.
        const auto output = noa::memory::zeros<f32>(rotational_average_shape, options);
        noa::geometry::fft::rotational_average<fft::HC2H>(input_rfft, output, shape);

        REQUIRE(test::Matcher(test::MATCH_ABS, polar_reduced, output, 1e-3));
    }
}
