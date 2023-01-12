#include <noa/Geometry.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/IO.h>
#include <noa/Signal.h>

#include "catch2/catch.hpp"
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::signal::sphere, 2D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float2_t>();
        const auto radius = test["radius"].as<float>();
        const auto taper = test["taper"].as<float>();
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::sphere({}, asset, center, radius, taper, {}, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            INFO(device);
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::sphere({}, result, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::sphere(data, data, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::sphere(data, data, center, radius, taper, {}, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::signal::rectangle, 2D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float2_t>();
        const auto radius = test["radius"].as<float2_t>();
        const auto taper = test["taper"].as<float>();
        const auto inv_matrix = geometry::rotate(math::deg2rad(-test["angle"].as<float>()));
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::rectangle({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::rectangle({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::signal::ellipse, 2D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float2_t>();
        const auto radius = test["radius"].as<float2_t>();
        const auto taper = test["taper"].as<float>();
        const auto inv_matrix = geometry::rotate(math::deg2rad(-test["angle"].as<float>()));
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::ellipse({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::ellipse({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-4));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-4));
        }
    }
}

TEST_CASE("unified::signal::sphere, 3D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float>();
        const auto taper = test["taper"].as<float>();
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::sphere({}, asset, center, radius, taper, {}, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::sphere({}, result, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::sphere(data, data, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::sphere(data, data, center, radius, taper, {}, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::signal::rectangle, 3D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float3_t>();
        const auto taper = test["taper"].as<float>();
        const auto inv_matrix = geometry::rotateY(-math::deg2rad(test["tilt"].as<float>()));
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::rectangle({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::rectangle({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::signal::ellipse, 3D", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float3_t>();
        const auto taper = test["taper"].as<float>();
        const auto inv_matrix = geometry::rotateY(-math::deg2rad(test["tilt"].as<float>()));
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::ellipse({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::ellipse({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 1e-4));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 1e-4));
        }
    }
}

TEST_CASE("unified::signal::cylinder", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["test"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float>();
        const auto length = test["length"].as<float>();
        const auto taper = test["taper"].as<float>();
        const auto inv_matrix = geometry::rotateY(math::deg2rad(-test["tilt"].as<float>()));
        const auto cvalue = test["cvalue"].as<float>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            signal::cylinder({}, asset, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            INFO(device);
            ArrayOption option(device, Allocator::MANAGED);
            Array asset = io::load<float>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            Array result = memory::empty<float>(shape, option);
            signal::cylinder({}, result, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            Array data = math::random<float>(math::uniform_t{}, shape, -10, 10, option);
            math::ewise(result, data, result, math::multiply_t{});
            signal::cylinder(data, data, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            math::randomize(math::uniform_t{}, data, -10, 10);
            math::ewise(cvalue, asset, asset, math::minus_t{});
            math::ewise(data, asset, asset, math::multiply_t{});
            signal::cylinder(data, data, center, radius, length, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::sphere, 2D matches 3D", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const auto center_3d = float3_t(dim3_t(shape.get(1)) / 2);
    const auto center_2d = float2_t(dim2_t(shape.get(2)) / 2);

    const float radius = 20;
    const float edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const float angle = math::deg2rad(-67.f);
    const float22_t inv_transform_2d = geometry::rotate(-angle);
    const float33_t inv_transform_3d = geometry::euler2matrix(float3_t{angle, 0, 0}).transpose();

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        ArrayOption option(device, Allocator::MANAGED);
        Array input = math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        Array output_2d = memory::empty<TestType>(shape, option);
        Array output_3d = memory::empty<TestType>(shape, option);

        const auto cvalue = traits::value_type_t<TestType>(1);
        signal::sphere(input, output_2d, center_2d, radius, edge_size, inv_transform_2d, {}, cvalue, invert);
        signal::sphere(input, output_3d, center_3d, radius, edge_size, inv_transform_3d, {}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::signal::rectangle, 2D matches 3D", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const auto center_3d = float3_t(dim3_t(shape.get(1)) / 2);
    const auto center_2d = float2_t(dim2_t(shape.get(2)) / 2);

    const float3_t radius_3d{1, 20, 20};
    const float2_t radius_2d{20, 20};

    const float edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const float angle = math::deg2rad(-67.f);
    const float22_t inv_transform_2d = geometry::rotate(-angle);
    const float33_t inv_transform_3d = geometry::euler2matrix(float3_t{angle, 0, 0}).transpose();

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        ArrayOption option(device, Allocator::MANAGED);
        Array input = math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        Array output_2d = memory::empty<TestType>(shape, option);
        Array output_3d = memory::empty<TestType>(shape, option);

        const auto cvalue = traits::value_type_t<TestType>(1);
        signal::rectangle(input, output_2d, center_2d, radius_2d, edge_size, inv_transform_2d, {}, cvalue, invert);
        signal::rectangle(input, output_3d, center_3d, radius_3d, edge_size, inv_transform_3d, {}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::signal::ellipse, 2D matches 3D", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const auto center_3d = float3_t(dim3_t(shape.get(1)) / 2);
    const auto center_2d = float2_t(dim2_t(shape.get(2)) / 2);

    const float3_t radius_3d{1, 40, 25};
    const float2_t radius_2d{40, 25};

    const float edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const float angle = math::deg2rad(25.f);
    const float22_t inv_transform_2d = geometry::rotate(-angle);
    const float33_t inv_transform_3d = geometry::euler2matrix(float3_t{angle, 0, 0}).transpose();

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        ArrayOption option(device, Allocator::MANAGED);
        Array input = math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        Array output_2d = memory::empty<TestType>(shape, option);
        Array output_3d = memory::empty<TestType>(shape, option);
        const auto cvalue = traits::value_type_t<TestType>(1);

        signal::ellipse(input, output_2d, center_2d, radius_2d, edge_size, inv_transform_2d, {}, cvalue, invert);
        signal::ellipse(input, output_3d, center_3d, radius_3d, edge_size, inv_transform_3d, {}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::signal::sphere, cpu vs gpu", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(2u, 3u);
    const dim4_t shape = test::getRandomShapeBatched(ndim);

    if (!Device::any(Device::GPU))
        return;

    Array input_cpu = math::random<TestType>(math::uniform_t{}, shape, -5, 5);
    Array input_gpu = input_cpu.to(Device("gpu"));
    Array output_cpu = memory::like(input_cpu);
    Array output_gpu = memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = float3_t(dim3_t(shape.get(1)) / 2) + randomizer.get();
    const auto radius = math::abs(randomizer.get() * 10);
    const auto edge_size = math::abs(randomizer.get()) + 1;
    const auto invert = test::Randomizer<int>(0, 1).get();
    const auto angles = float3_t{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = geometry::euler2matrix(angles).transpose();
    const auto cvalue = traits::value_type_t<TestType>(1);

    signal::sphere(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    signal::sphere(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to(Device("cpu")), 1e-4));
}

TEMPLATE_TEST_CASE("unified::signal::rectangle, cpu vs gpu", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(2u, 3u);
    const dim4_t shape = test::getRandomShapeBatched(ndim);

    if (!Device::any(Device::GPU))
        return;

    Array input_cpu = math::random<TestType>(math::uniform_t{}, shape, -5, 5);
    Array input_gpu = input_cpu.to(Device("gpu"));
    Array output_cpu = memory::like(input_cpu);
    Array output_gpu = memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = float3_t(dim3_t(shape.get(1)) / 2) + randomizer.get();
    const auto radius = float3_t{math::abs(randomizer.get() * 10),
                                 math::abs(randomizer.get() * 10),
                                 math::abs(randomizer.get() * 10)};
    const float edge_size = math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = float3_t{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = geometry::euler2matrix(angles).transpose();
    const auto cvalue = traits::value_type_t<TestType>(1);

    signal::rectangle(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    signal::rectangle(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to(Device("cpu")), 1e-4));
}

TEMPLATE_TEST_CASE("unified::signal::ellipse, cpu vs gpu", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(2u, 3u);
    const dim4_t shape = test::getRandomShapeBatched(ndim);

    if (!Device::any(Device::GPU))
        return;

    Array input_cpu = math::random<TestType>(math::uniform_t{}, shape, -5, 5);
    Array input_gpu = input_cpu.to(Device("gpu"));
    Array output_cpu = memory::like(input_cpu);
    Array output_gpu = memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = float3_t(dim3_t(shape.get(1)) / 2) + randomizer.get();
    const auto radius = float3_t{math::abs(randomizer.get() * 10),
                                 math::abs(randomizer.get() * 10),
                                 math::abs(randomizer.get() * 10)};
    const float edge_size = math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = float3_t{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = geometry::euler2matrix(angles).transpose();
    const auto cvalue = traits::value_type_t<TestType>(1);

    signal::ellipse(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    signal::ellipse(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to(Device("cpu")), 5e-4));
}

TEMPLATE_TEST_CASE("unified::signal::cylinder, cpu vs gpu", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(2u, 3u);
    const dim4_t shape = test::getRandomShapeBatched(ndim);

    if (!Device::any(Device::GPU))
        return;

    Array input_cpu = math::random<TestType>(math::uniform_t{}, shape, -5, 5);
    Array input_gpu = input_cpu.to(Device("gpu"));
    Array output_cpu = memory::like(input_cpu);
    Array output_gpu = memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = float3_t(dim3_t(shape.get(1)) / 2) + randomizer.get();
    const auto radius = math::abs(randomizer.get() * 10);
    const auto length = math::abs(randomizer.get() * 10);
    const auto edge_size = math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = float3_t{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = geometry::euler2matrix(angles).transpose();
    const auto cvalue = traits::value_type_t<TestType>(1);

    signal::cylinder(input_cpu, output_cpu, center, radius, length, edge_size, inv_transform, {}, cvalue, invert);
    signal::cylinder(input_gpu, output_gpu, center, radius, length, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to(Device("cpu")), 1e-4));
}

TEMPLATE_TEST_CASE("unified::signal::ellipse, 2D affine", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(2);
    const auto center = float2_t(dim2_t(shape.get(2)) / 2);
    const float2_t radius{40, 25};
    const float edge_size = 5;

    const float angle = math::deg2rad(25.f);
    const float22_t inv_matrix_linear = geometry::rotate(-angle);
    const float33_t inv_matrix_affine =
            geometry::translate(center) *
            float33_t(geometry::rotate(-angle)) *
            geometry::translate(-center);

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        ArrayOption option(device, Allocator::MANAGED);
        Array input = math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        Array output_linear = memory::empty<TestType>(shape, option);
        Array output_affine = memory::empty<TestType>(shape, option);

        signal::ellipse(input, output_linear, center, radius, edge_size, inv_matrix_linear);
        signal::ellipse(input, output_affine, center, radius, edge_size, inv_matrix_affine);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_linear, output_affine, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::signal::ellipse, 3D affine", "[assets][noa][unified]",
                   float, double, cfloat_t, cdouble_t) {
    const dim4_t shape = test::getRandomShapeBatched(3);
    const auto center = float3_t(dim3_t(shape.get(1)) / 2);
    const float3_t radius{10, 15, 25};
    const float edge_size = 5;

    const auto angles = math::deg2rad(float3_t{25, 15, 0});
    const float33_t inv_matrix_linear = geometry::euler2matrix(angles).transpose();
    const float44_t inv_matrix_affine =
            geometry::translate(center) *
            float44_t(inv_matrix_linear) *
            geometry::translate(-center);

    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        ArrayOption option(device, Allocator::MANAGED);
        Array input = math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        Array output_linear = memory::empty<TestType>(shape, option);
        Array output_affine = memory::empty<TestType>(shape, option);

        signal::ellipse(input, output_linear, center, radius, edge_size, inv_matrix_linear);
        signal::ellipse(input, output_affine, center, radius, edge_size, inv_matrix_affine);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_linear, output_affine, 1e-4));
    }
}
