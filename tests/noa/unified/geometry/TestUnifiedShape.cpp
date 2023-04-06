#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/Ewise.hpp>

#include "catch2/catch.hpp"
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::geometry::sphere(), 2d", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f32>>();
        const auto radius = test["radius"].as<f32>();
        const auto taper = test["taper"].as<f32>();
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::sphere({}, asset, center, radius, taper, {}, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::sphere({}, result, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::sphere(data, data, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::sphere(data, data, center, radius, taper, {}, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::rectangle(), 2d", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f32>>();
        const auto radius = test["radius"].as<Vec2<f32>>();
        const auto taper = test["taper"].as<f32>();
        const auto inv_matrix = noa::geometry::rotate(noa::math::deg2rad(-test["angle"].as<f32>()));
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::rectangle({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::rectangle({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::ellipse(), 2d", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f32>>();
        const auto radius = test["radius"].as<Vec2<f32>>();
        const auto taper = test["taper"].as<f32>();
        const auto inv_matrix = noa::geometry::rotate(noa::math::deg2rad(-test["angle"].as<f32>()));
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::ellipse({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::ellipse({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-4));
        }
    }
}

TEST_CASE("unified::geometry::sphere, 3d", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f32>>();
        const auto radius = test["radius"].as<f32>();
        const auto taper = test["taper"].as<f32>();
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::sphere({}, asset, center, radius, taper, {}, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::sphere({}, result, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::sphere(data, data, center, radius, taper, {}, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::sphere(data, data, center, radius, taper, {}, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::rectangle, 3d", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f32>>();
        const auto radius = test["radius"].as<Vec3<f32>>();
        const auto taper = test["taper"].as<f32>();
        const auto inv_matrix = noa::geometry::rotate_y(-noa::math::deg2rad(test["tilt"].as<f32>()));
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            Array asset = memory::empty<float>(shape);
            geometry::rectangle({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::rectangle({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::rectangle(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::ellipse, 3d", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f32>>();
        const auto radius = test["radius"].as<Vec3<f32>>();
        const auto taper = test["taper"].as<f32>();
        const auto inv_matrix = noa::geometry::rotate_y(-noa::math::deg2rad(test["tilt"].as<f32>()));
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::ellipse({}, asset, center, radius, taper, inv_matrix, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::ellipse({}, result, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 1e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::ellipse(data, data, center, radius, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 1e-4));
        }
    }
}

TEST_CASE("unified::geometry::cylinder", "[assets][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["test"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f32>>();
        const auto radius = test["radius"].as<f32>();
        const auto length = test["length"].as<f32>();
        const auto taper = test["taper"].as<f32>();
        const auto inv_matrix = geometry::rotate_y(math::deg2rad(-test["tilt"].as<f32>()));
        const auto cvalue = test["cvalue"].as<f32>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::memory::empty<f32>(shape);
            noa::geometry::cylinder({}, asset, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            noa::io::save(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            INFO(device);
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::load_data<f32>(filename_expected, false, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::memory::empty<f32>(shape, option);
            noa::geometry::cylinder({}, result, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::math::random<f32>(noa::math::uniform_t{}, shape, -10, 10, option);
            noa::ewise_binary(result, data, result, noa::multiply_t{});
            noa::geometry::cylinder(data, data, center, radius, length, taper, inv_matrix, {}, cvalue, invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::math::randomize(noa::math::uniform_t{}, data, -10, 10);
            noa::ewise_trinary(cvalue, asset, data, asset, noa::minus_multiply_t{});
            noa::geometry::cylinder(data, data, center, radius, length, taper, inv_matrix, {}, cvalue, !invert);
            REQUIRE(test::Matcher(test::MATCH_ABS, asset, data, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::sphere, 2d matches 3d", "[noa][unified]", f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(2);
    const auto center_3d = (shape.pop_front().vec() / 2).as<f32>();
    const auto center_2d = (shape.filter(2, 3).vec() / 2).as<f32>();

    const f32 radius = 20;
    const f32 edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const f32 angle = noa::math::deg2rad(-67.f);
    const Float22 inv_transform_2d = noa::geometry::rotate(-angle);
    const Float33 inv_transform_3d = noa::geometry::euler2matrix(Vec3<f32>{angle, 0, 0}).transpose();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, option);
        const auto output_2d = noa::memory::empty<TestType>(shape, option);
        const auto output_3d = noa::memory::empty<TestType>(shape, option);

        const auto cvalue = noa::traits::value_type_t<TestType>(1);
        noa::geometry::sphere(input, output_2d, center_2d, radius, edge_size, inv_transform_2d, {}, cvalue, invert);
        noa::geometry::sphere(input, output_3d, center_3d, radius, edge_size, inv_transform_3d, noa::multiply_t{}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::rectangle, 2d matches 3d", "[noa][unified]", f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(2);
    const auto center_3d = (shape.pop_front().vec() / 2).as<f32>();
    const auto center_2d = (shape.filter(2, 3).vec() / 2).as<f32>();

    const Vec3<f32> radius_3d{1, 20, 20};
    const Vec2<f32> radius_2d{20, 20};

    const f32 edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const f32 angle = noa::math::deg2rad(-67.f);
    const Float22 inv_transform_2d = noa::geometry::rotate(-angle);
    const Float33 inv_transform_3d = noa::geometry::euler2matrix(Vec3<f32>{angle, 0, 0}).transpose();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, option);
        const auto output_2d = noa::memory::empty<TestType>(shape, option);
        const auto output_3d = noa::memory::empty<TestType>(shape, option);

        const auto cvalue = noa::traits::value_type_t<TestType>(1);
        noa::geometry::rectangle(input, output_2d, center_2d, radius_2d, edge_size, inv_transform_2d, {}, cvalue, invert);
        noa::geometry::rectangle(input, output_3d, center_3d, radius_3d, edge_size, inv_transform_3d, {}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 2d matches 3d", "[noa][unified]", f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(2);
    const auto center_3d = (shape.pop_front().vec() / 2).as<f32>();
    const auto center_2d = (shape.filter(2, 3).vec() / 2).as<f32>();

    const Vec3<f32> radius_3d{1, 40, 25};
    const Vec2<f32> radius_2d{40, 25};

    const f32 edge_size = 5;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const f32 angle = noa::math::deg2rad(25.f);
    const Float22 inv_transform_2d = noa::geometry::rotate(-angle);
    const Float33 inv_transform_3d = noa::geometry::euler2matrix(Vec3<f32>{angle, 0, 0}).transpose();

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, option);
        const auto output_2d = noa::memory::empty<TestType>(shape, option);
        const auto output_3d = noa::memory::empty<TestType>(shape, option);
        const auto cvalue = noa::traits::value_type_t<TestType>(1);

        noa::geometry::ellipse(input, output_2d, center_2d, radius_2d, edge_size, inv_transform_2d, {}, cvalue, invert);
        noa::geometry::ellipse(input, output_3d, center_3d, radius_3d, edge_size, inv_transform_3d, {}, cvalue, invert);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::sphere, cpu vs gpu", "[noa][unified]", f32, f64, c32, c64) {
    const i64 ndim = GENERATE(2, 3);
    const Shape4<i64> shape = test::get_random_shape4_batched(ndim);

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5);
    const auto input_gpu = input_cpu.to(Device("gpu"));
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = (shape.pop_front() / 2).vec().as<f32>() + randomizer.get();
    const auto radius = noa::math::abs(randomizer.get() * 10);
    const auto edge_size = noa::math::abs(randomizer.get()) + 1;
    const auto invert = test::Randomizer<int>(0, 1).get();
    const auto angles = Vec3<f32>{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = noa::geometry::euler2matrix(angles).transpose();
    const auto cvalue = noa::traits::value_type_t<TestType>(1);

    noa::geometry::sphere(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    noa::geometry::sphere(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to_cpu(), 1e-4));
}

TEMPLATE_TEST_CASE("unified::geometry::rectangle, cpu vs gpu", "[noa][unified]", f32, f64, c32, c64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4_batched(ndim);

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5);
    const auto input_gpu = input_cpu.to(Device("gpu"));
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = (shape.pop_front() / 2).vec().as<f32>() + randomizer.get();
    const auto radius = Vec3<f32>{noa::math::abs(randomizer.get() * 10),
                                  noa::math::abs(randomizer.get() * 10),
                                  noa::math::abs(randomizer.get() * 10)};
    const f32 edge_size = noa::math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = Vec3<f32>{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = noa::geometry::euler2matrix(angles).transpose();
    const auto cvalue = noa::traits::value_type_t<TestType>(1);

    noa::geometry::rectangle(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    noa::geometry::rectangle(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to_cpu(), 1e-4));
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, cpu vs gpu", "[noa][unified]", f32, f64, c32, c64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4_batched(ndim);

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5);
    const auto input_gpu = input_cpu.to(Device("gpu"));
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = (shape.pop_front() / 2).vec().as<f32>() + randomizer.get();
    const auto radius = Vec3<f32>{noa::math::abs(randomizer.get() * 10),
                                  noa::math::abs(randomizer.get() * 10),
                                  noa::math::abs(randomizer.get() * 10)};
    const float edge_size = noa::math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = Vec3<f32>{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = noa::geometry::euler2matrix(angles).transpose();
    const auto cvalue = noa::traits::value_type_t<TestType>(1);

    noa::geometry::ellipse(input_cpu, output_cpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    noa::geometry::ellipse(input_gpu, output_gpu, center, radius, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to_cpu(), 5e-4));
}

TEMPLATE_TEST_CASE("unified::geometry::cylinder, cpu vs gpu", "[noa][unified]", f32, f64, c32, c64) {
    const i64 ndim = GENERATE(2, 3);
    const auto shape = test::get_random_shape4_batched(ndim);

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5);
    const auto input_gpu = input_cpu.to(Device("gpu"));
    const auto output_cpu = noa::memory::like(input_cpu);
    const auto output_gpu = noa::memory::like(input_gpu);

    test::Randomizer<float> randomizer(-5, 5);
    const auto center = (shape.pop_front() / 2).vec().as<f32>() + randomizer.get();
    const auto radius = noa::math::abs(randomizer.get() * 10);
    const auto length = noa::math::abs(randomizer.get() * 10);
    const auto edge_size = noa::math::abs(randomizer.get()) + 1;
    const bool invert = test::Randomizer<int>(0, 1).get();
    const auto angles = Vec3<f32>{randomizer.get(), randomizer.get(), randomizer.get()};
    const auto inv_transform = noa::geometry::euler2matrix(angles).transpose();
    const auto cvalue = noa::traits::value_type_t<TestType>(1);

    noa::geometry::cylinder(input_cpu, output_cpu, center, radius, length, edge_size, inv_transform, {}, cvalue, invert);
    noa::geometry::cylinder(input_gpu, output_gpu, center, radius, length, edge_size, inv_transform, {}, cvalue, invert);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to_cpu(), 1e-4));
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 2d affine", "[noa][unified]", f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(2);
    const auto center = (shape.filter(2, 3) / 2).vec().as<f32>();
    const Vec2<f32> radius{40, 25};
    const f32 edge_size = 5;

    const f32 angle = noa::math::deg2rad(25.f);
    const Float22 inv_matrix_linear = noa::geometry::rotate(-angle);
    const Float33 inv_matrix_affine =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-angle)) *
            noa::geometry::translate(-center);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, option);
        const auto output_linear = noa::memory::empty<TestType>(shape, option);
        const auto output_affine = noa::memory::empty<TestType>(shape, option);

        noa::geometry::ellipse(input, output_linear, center, radius, edge_size, inv_matrix_linear);
        noa::geometry::ellipse(input, output_affine, center, radius, edge_size, inv_matrix_affine);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_linear, output_affine, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 3d affine", "[noa][unified]", f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(3);
    const auto center =(shape.filter(1, 2, 3) / 2).vec().as<f32>();
    const Vec3<f32> radius{10, 15, 25};
    const f32 edge_size = 5;

    const auto angles = noa::math::deg2rad(Vec3<f32>{25, 15, 0});
    const Float33 inv_matrix_linear = noa::geometry::euler2matrix(angles).transpose();
    const Float44 inv_matrix_affine =
            noa::geometry::translate(center) *
            noa::geometry::linear2affine(inv_matrix_linear) *
            noa::geometry::translate(-center);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::math::random<TestType>(math::uniform_t{}, shape, -5, 5, option);
        const auto output_linear = noa::memory::empty<TestType>(shape, option);
        const auto output_affine = noa::memory::empty<TestType>(shape, option);

        noa::geometry::ellipse(input, output_linear, center, radius, edge_size, inv_matrix_linear);
        noa::geometry::ellipse(input, output_affine, center, radius, edge_size, inv_matrix_affine);
        REQUIRE(test::Matcher(test::MATCH_ABS, output_linear, output_affine, 1e-4));
    }
}

TEST_CASE("unified::geometry::shapes, 2d batched", "[noa][unified]") {
    const auto shape = Shape4<i64>{10, 1, 512, 512};
    const auto center = shape.filter(2, 3).vec().as<f32>() / 2;
    const auto radius = Vec2<f32>{128, 198};

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    Array matrices = noa::memory::empty<Float22>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices(0, 0, 0, i) = noa::geometry::rotate(static_cast<f32>(i * 4));

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::memory::empty<f32>(shape, option);
        const Array serial = noa::memory::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::ellipse({}, serial.subregion(i), center, radius, 10.f, matrices(0, 0, 0, i));
            noa::geometry::ellipse({}, batched, center, radius, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::sphere({}, serial.subregion(i), center, 126.f, 10.f, matrices(0, 0, 0, i));
            noa::geometry::sphere({}, batched, center, 126.f, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::rectangle({}, serial.subregion(i), center, radius, 10.f, matrices(0, 0, 0, i));
            noa::geometry::rectangle({}, batched, center, radius, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }
    }
}

TEST_CASE("unified::geometry::shapes, 3d batched", "[noa][unified]") {
    const auto shape = Shape4<i64>{4, 175, 165, 198};
    const auto center = shape.filter(1, 2, 3).vec().as<f32>() / 2;
    const auto radius = Vec3<f32>{64, 54, 56};

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    Array matrices = noa::memory::empty<Float33>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices(0, 0, 0, i) = noa::geometry::euler2matrix(Vec3<f32>{static_cast<f32>(i * 4), 0, 0}, "zyx");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::memory::empty<f32>(shape, option);
        const Array serial = noa::memory::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::ellipse({}, serial.subregion(i), center, radius, 10.f, matrices(0, 0, 0, i));
            noa::geometry::ellipse({}, batched, center, radius, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::sphere({}, serial.subregion(i), center, 91.f, 10.f, matrices(0, 0, 0, i));
            noa::geometry::sphere({}, batched, center, 91.f, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::rectangle({}, serial.subregion(i), center, radius, 10.f, matrices(0, 0, 0, i));
            noa::geometry::rectangle({}, batched, center, radius, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }

        AND_THEN("cylinder") {
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::cylinder({}, serial.subregion(i), center, 45.f, 71.f, 10.f, matrices(0, 0, 0, i));
            noa::geometry::cylinder({}, batched, center, 45.f, 71.f, 10.f, matrices_device);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, batched, serial, 1e-6));
        }
    }
}
