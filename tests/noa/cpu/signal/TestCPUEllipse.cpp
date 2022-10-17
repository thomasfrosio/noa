#include <noa/common/io/MRCFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Shape.h>


#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

static constexpr bool GENERATE_ASSETS = false;

// Just compare against manually checked data.
TEST_CASE("cpu::signal::ellipse(), 2D", "[assets][noa][cpu]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["tests2D"];
    io::MRCFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<size4_t>();
        const auto center = test["center"].as<float2_t>();
        const auto radius = test["radius"].as<float2_t>();
        const auto taper = test["taper"].as<float>();
        const auto fwd_transform = geometry::rotate(math::deg2rad(test["angle"].as<float>()));
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const size4_t strides = shape.strides();
        const size_t elements = shape.elements();

        if constexpr (GENERATE_ASSETS) {
            cpu::memory::PtrHost<float> mask(elements);
            cpu::signal::ellipse<float>(nullptr, {}, mask.share(), strides, shape,
                                        center, radius, taper, math::inverse(fwd_transform), invert, stream);
            stream.synchronize();
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(mask.get());
            file.close();
            continue;
        }

        file.open(filename_expected, io::READ);
        if (all(file.shape() != shape))
            FAIL("asset shape is not correct");
        cpu::memory::PtrHost<float> mask_expected(elements);
        file.readAll(mask_expected.get());

        cpu::memory::PtrHost<float> input_expected(elements);
        cpu::memory::PtrHost<float> input_result(elements);
        cpu::memory::PtrHost<float> mask_result(elements);

        test::randomize(input_expected.get(), elements, randomizer);
        test::copy(input_expected.get(), input_result.get(), elements);

        // Test saving the mask.
        cpu::signal::ellipse<float>(nullptr, {}, mask_result.share(), strides, shape,
                                    center, radius, taper, math::inverse(fwd_transform), invert, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, mask_expected.get(), mask_result.get(), elements, 5e-6));

        // Test on-the-fly, in-place.
        cpu::signal::ellipse(input_result.share(), strides, input_result.share(), strides, shape,
                             center, radius, taper, math::inverse(fwd_transform), invert, stream);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 5e-6));
    }
}

TEST_CASE("cpu::signal::ellipse(), 3D", "[assets][noa][cpu]") {
    test::Randomizer<float> randomizer(-5, 5);

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["tests3D"];
    io::MRCFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<size4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float3_t>();
        const auto taper = test["taper"].as<float>();
        const auto fwd_transform = geometry::euler2matrix(math::deg2rad(test["angles"].as<float3_t>()));
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const size4_t strides = shape.strides();
        const size_t elements = shape.elements();

        if constexpr (GENERATE_ASSETS) {
            cpu::memory::PtrHost<float> mask(elements);
            cpu::signal::ellipse<float>(nullptr, {}, mask.share(), strides, shape,
                                        center, radius, taper, math::inverse(fwd_transform), invert, stream);
            stream.synchronize();
            file.open(filename_expected, io::WRITE);
            file.shape(shape);
            file.writeAll(mask.get());
            file.close();
            continue;
        }

        file.open(filename_expected, io::READ);
        if (all(file.shape() != shape))
            FAIL("asset shape is not correct");
        cpu::memory::PtrHost<float> mask_expected(elements);
        file.readAll(mask_expected.get());

        cpu::memory::PtrHost<float> input_expected(elements);
        cpu::memory::PtrHost<float> input_result(elements);
        cpu::memory::PtrHost<float> mask_result(elements);

        test::randomize(input_expected.get(), elements, randomizer);
        test::copy(input_expected.get(), input_result.get(), elements);

        // Test saving the mask.
        cpu::signal::ellipse<float>(nullptr, {}, mask_result.share(), strides, shape,
                                    center, radius, taper, math::inverse(fwd_transform), invert, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, mask_expected.get(), mask_result.get(), elements, 5e-6));

        // Test on-the-fly, in-place.
        cpu::signal::ellipse(input_result.share(), strides, input_result.share(), strides, shape,
                             center, radius, taper, math::inverse(fwd_transform), invert, stream);
        for (size_t idx = 0; idx < elements; ++idx)
            input_expected[idx] *= mask_expected[idx];
        REQUIRE(test::Matcher(test::MATCH_ABS, input_result.get(), input_expected.get(), elements, 5e-6));
    }
}

TEMPLATE_TEST_CASE("cpu::signal::ellipse(), 2D matches 3D", "[assets][noa][cpu]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const dim4_t strides = shape.strides();
    const dim_t elements = shape.elements();

    cpu::memory::PtrHost<TestType> input(elements);
    cpu::memory::PtrHost<TestType> output_2d(elements);
    cpu::memory::PtrHost<TestType> output_3d(elements);

    test::Randomizer<float> randomizer(-5, 5);
    test::randomize(input.get(), elements, randomizer);

    const auto center_3d = float3_t(dim3_t(shape.get(1)) / 2);
    const auto center_2d = float2_t(dim2_t(shape.get(2)) / 2);

    const float3_t radius_3d{1, 40, 25};
    const float2_t radius_2d{40, 25};

    const float edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();

    const float angle = math::deg2rad(25.f);
    const float22_t fwd_transform_2d = geometry::rotate(angle);
    const float33_t fwd_transform_3d = geometry::euler2matrix(float3_t{angle, 0, 0});

    cpu::Stream stream;
    cpu::signal::ellipse(input.share(), strides, output_2d.share(), strides, shape,
                         center_2d, radius_2d, edge_size, math::inverse(fwd_transform_2d), invert, stream);
    cpu::signal::ellipse(input.share(), strides, output_3d.share(), strides, shape,
                         center_3d, radius_3d, edge_size, math::inverse(fwd_transform_3d), invert, stream);

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_2d.get(), output_3d.get(), elements, 5e-6));
}
