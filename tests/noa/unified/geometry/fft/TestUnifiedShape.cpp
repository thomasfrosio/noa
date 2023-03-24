#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/geometry/fft/Shape.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::signal::fft::ellipse", "[noa][unified]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["tests"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const auto test = tests[nb];
        const auto filename = path_base / test["filename"].as<Path>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f32>>();
        const auto radius = test["radius"].as<Vec3<f32>>();
        const auto edge_size = test["edge_size"].as<f32>();
        const auto fwd_transform = noa::geometry::euler2matrix(
                noa::math::deg2rad(test["angles"].as<Vec3<f32>>()), "ZYX").inverse();

        constexpr bool COMPUTE_ASSETS = false;
        if constexpr (COMPUTE_ASSETS) {
            const Array<float> output(shape);
            noa::geometry::fft::ellipse<fft::FC2FC>({}, output, center, radius, edge_size, fwd_transform);
            io::save(output, filename);
            continue;
        }

        const Array expected = noa::io::load_data<f32>(filename);
        const Array expected_remap = fft::remap(fft::FC2F, expected, shape);
        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            const Array<float> output(shape, options);
            noa::geometry::fft::ellipse<fft::FC2FC>({}, output, center, radius, edge_size, fwd_transform);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 5e-5));

            noa::geometry::fft::ellipse<fft::F2F>({}, output, center, radius, edge_size, fwd_transform);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_remap, output, 5e-5));
        }
    }
}
