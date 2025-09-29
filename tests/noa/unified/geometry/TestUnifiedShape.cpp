#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/Draw.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/Reduce.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

namespace {
    struct ApplyInvertedMask {
        f32 cvalue;
        constexpr void operator()(f32 input, f32 mask, f32& output) const {
            output = input * (cvalue - mask);
        }
    };
}

TEST_CASE("unified::geometry::sphere(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f64>>();
        const auto radius = test["radius"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto sphere = noa::geometry::Sphere{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_sphere = sphere;
        inverted_sphere.invert = not sphere.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, sphere.draw());
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, sphere.draw<f32>());
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, -10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, sphere.draw<f32>());
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, -10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_sphere.draw<f32>());
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::rectangle(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f64>>();
        const auto radius = test["radius"].as<Vec2<f64>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = noa::geometry::rotate(noa::deg2rad(-test["angle"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto rectangle = noa::geometry::Rectangle{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_rectangle = rectangle;
        inverted_rectangle.invert = not rectangle.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, rectangle.draw(), inv_matrix);
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::ellipse(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec2<f64>>();
        const auto radius = test["radius"].as<Vec2<f64>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = noa::geometry::rotate(noa::deg2rad(-test["angle"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto ellipse = noa::geometry::Ellipse{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_ellipse = ellipse;
        inverted_ellipse.invert = not ellipse.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, ellipse.draw(), inv_matrix);
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-4));
        }
    }
}

TEST_CASE("unified::geometry::sphere, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f64>>();
        const auto radius = test["radius"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto sphere = noa::geometry::Sphere{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_sphere = sphere;
        inverted_sphere.invert = not sphere.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, sphere.draw());
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, sphere.draw());
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, sphere.draw());
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_sphere.draw());
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::rectangle, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f64>>();
        const auto radius = test["radius"].as<Vec3<f64>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = noa::geometry::rotate_y(-noa::deg2rad(test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto rectangle = noa::geometry::Rectangle{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_rectangle = rectangle;
        inverted_rectangle.invert = not rectangle.invert;

        if constexpr (COMPUTE_ASSETS) {
            Array asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, rectangle.draw(), inv_matrix);
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_rectangle.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::ellipse, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f64>>();
        const auto radius = test["radius"].as<Vec3<f64>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = noa::geometry::rotate_y(-noa::deg2rad(test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto ellipse = noa::geometry::Ellipse{
            .center=center,
            .radius=radius,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_ellipse = ellipse;
        inverted_ellipse.invert = not ellipse.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, ellipse.draw(), inv_matrix);
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 1e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_ellipse.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 1e-4));
        }
    }
}

TEST_CASE("unified::geometry::cylinder", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["test"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4<i64>>();
        const auto center = test["center"].as<Vec3<f64>>();
        const auto radius = test["radius"].as<f64>();
        const auto length = test["length"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = noa::geometry::rotate_y(noa::deg2rad(-test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto cylinder = noa::geometry::Cylinder{
            .center=center,
            .radius=radius,
            .length=length,
            .smoothness=taper,
            .cvalue=cvalue,
            .invert=invert,
        };
        auto inverted_cylinder = cylinder;
        inverted_cylinder.invert = not cylinder.invert;

        if constexpr (COMPUTE_ASSETS) {
            const auto asset = noa::empty<f32>(shape);
            noa::geometry::draw({}, asset, cylinder.draw(), inv_matrix);
            noa::io::write(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            INFO(device);
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::io::read_data<f32>(filename_expected, {}, option);
            if (noa::any(asset.shape() != shape))
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            noa::geometry::draw({}, result, cylinder.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            noa::geometry::draw(data, data, cylinder.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            noa::geometry::draw(data, data, inverted_cylinder.draw(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::sphere, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(2);
    const auto center_3d = (shape.pop_front().vec / 2).as<f64>();
    const auto center_2d = (shape.filter(2, 3).vec / 2).as<f64>();

    constexpr f64 radius = 20.;
    constexpr f64 edge_size = 5.;
    constexpr f64 angle = noa::deg2rad(-67.);
    const auto inv_transform_2d = noa::geometry::rotate(-angle);
    const auto inv_transform_3d = noa::geometry::euler2matrix(Vec{angle, 0., 0.}).transpose();
    const bool invert = test::Randomizer<i32>(0, 1).get();

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random<TestType>(noa::Uniform{-5.f, 5.f}, shape, option);
        const auto output_2d = noa::empty<TestType>(shape, option);
        const auto output_3d = noa::empty<TestType>(shape, option);

        const auto sphere_2d = noa::geometry::Sphere{
            .center=center_2d,
            .radius=radius,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto sphere_3d = noa::geometry::Sphere{
            .center=center_3d,
            .radius=radius,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        noa::geometry::draw(input, output_2d, sphere_2d.draw(), inv_transform_2d);
        noa::geometry::draw(input, output_3d, sphere_3d.draw(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::rectangle, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(2);
    const auto center_3d = (shape.pop_front().vec / 2).as<f64>();
    const auto center_2d = (shape.filter(2, 3).vec / 2).as<f64>();

    constexpr auto radius_3d = Vec{1., 20., 20.};
    constexpr auto radius_2d = Vec{20., 20.};
    constexpr f64 edge_size = 5.;
    constexpr f64 angle = noa::deg2rad(-67.);
    const auto inv_transform_2d = noa::geometry::rotate(-angle);
    const auto inv_transform_3d = noa::geometry::euler2matrix(Vec{angle, 0., 0.}).transpose();

    const bool invert = test::Randomizer<i32>(0, 1).get();

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random<TestType>(noa::Uniform{-5.f, 5.f}, shape, option);
        const auto output_2d = noa::empty<TestType>(shape, option);
        const auto output_3d = noa::empty<TestType>(shape, option);

        const auto rectangle_2d = noa::geometry::Rectangle{
            .center=center_2d,
            .radius=radius_2d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto rectangle_3d = noa::geometry::Rectangle{
            .center=center_3d,
            .radius=radius_3d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        noa::geometry::draw(input, output_2d, rectangle_2d.draw(), inv_transform_2d);
        noa::geometry::draw(input, output_3d, rectangle_3d.draw(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(2);
    const auto center_3d = (shape.pop_front().vec / 2).as<f64>();
    const auto center_2d = (shape.filter(2, 3).vec / 2).as<f64>();

    constexpr auto radius_3d = Vec{1., 40., 25.};
    constexpr auto radius_2d = Vec{40., 25.};
    constexpr f64 edge_size = 5;
    constexpr f64 angle = noa::deg2rad(25.);
    const auto inv_transform_2d = noa::geometry::rotate(-angle);
    const auto inv_transform_3d = noa::geometry::euler2matrix(Vec{angle, 0., 0.}).transpose();
    const bool invert = test::Randomizer<i32>(0, 1).get();

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random<TestType>(noa::Uniform{-5.f, 5.f}, shape, option);
        const auto output_2d = noa::empty<TestType>(shape, option);
        const auto output_3d = noa::empty<TestType>(shape, option);

        const auto ellipse_2d = noa::geometry::Ellipse{
            .center=center_2d,
            .radius=radius_2d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto ellipse_3d = noa::geometry::Ellipse{
            .center=center_3d,
            .radius=radius_3d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        noa::geometry::draw(input, output_2d, ellipse_2d.draw(), inv_transform_2d);
        noa::geometry::draw(input, output_3d, ellipse_3d.draw(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 2d affine", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(2);
    const auto ellipse = noa::geometry::Ellipse{
        .center = (shape.filter(2, 3) / 2).vec.as<f64>(),
        .radius = Vec{40., 25.},
        .smoothness = 5.,
    };

    constexpr f64 angle = noa::deg2rad(25.);
    const auto inv_matrix_linear = noa::geometry::rotate(-angle);
    const auto inv_matrix_affine =
            noa::geometry::translate(ellipse.center) *
            noa::geometry::linear2affine(noa::geometry::rotate(-angle)) *
            noa::geometry::translate(-ellipse.center);

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random<TestType>(noa::Uniform{-5.f, 5.f}, shape, option);
        const auto output_linear = noa::empty<TestType>(shape, option);
        const auto output_affine = noa::empty<TestType>(shape, option);

        noa::geometry::draw(input, output_linear, ellipse.draw(), inv_matrix_linear);
        noa::geometry::draw(input, output_affine, ellipse.draw(), inv_matrix_affine);
        REQUIRE(test::allclose_abs(output_linear, output_affine, 1e-4));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::ellipse, 3d affine", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(3);
    const auto ellipse = noa::geometry::Ellipse{
        .center = (shape.filter(1, 2, 3) / 2).vec.as<f64>(),
        .radius = Vec{10., 15., 25.},
        .smoothness = 5.,
    };

    constexpr auto angles = noa::deg2rad(Vec{25., 15., 0.});
    const auto inv_matrix_linear = noa::geometry::euler2matrix(angles).transpose();
    const auto inv_matrix_affine =
            noa::geometry::translate(ellipse.center) *
            noa::geometry::linear2affine(inv_matrix_linear) *
            noa::geometry::translate(-ellipse.center);

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random(noa::Uniform<TestType>{-5., 5.}, shape, option);
        const auto output_linear = noa::empty<TestType>(shape, option);
        const auto output_affine = noa::empty<TestType>(shape, option);

        noa::geometry::draw(input, output_linear, ellipse.draw(), inv_matrix_linear);
        noa::geometry::draw(input, output_affine, ellipse.draw(), inv_matrix_affine);
        REQUIRE(test::allclose_abs(output_linear, output_affine, 1e-4));
    }
}

TEST_CASE("unified::geometry::shapes, 2d batched") {
    constexpr auto shape = Shape4<i64>{10, 1, 124, 115};
    constexpr auto center = shape.filter(2, 3).vec.as<f64>() / 2;
    constexpr auto radius = Vec{42., 12.};
    constexpr auto smoothness = 10.;

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    Array matrices = noa::empty<Mat<f32, 2, 2>>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices.span_1d()[i] = noa::geometry::rotate(static_cast<f32>(i * 4));

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::empty<f32>(shape, option);
        const Array serial = noa::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            const auto ellipse = noa::geometry::Ellipse{center, radius, smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), ellipse.draw<f32>(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, ellipse.draw<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            const auto sphere = noa::geometry::Sphere{center, 126., smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), sphere.draw<f32>(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, sphere.draw<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            const auto rectangle = noa::geometry::Rectangle{center, radius, smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), rectangle.draw<f32>(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, rectangle.draw<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }
    }
}

TEST_CASE("unified::geometry::shapes, 3d batched") {
    constexpr auto shape = Shape4<i64>{4, 55, 65, 58};
    constexpr auto center = shape.filter(1, 2, 3).vec.as<f64>() / 2;
    constexpr auto radius = Vec{9., 21., 8.};
    constexpr auto smoothness = 10.;

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    Array matrices = noa::empty<Mat<f64, 3, 3>>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices.span_1d()[i] = noa::geometry::euler2matrix(Vec{static_cast<f64>(i * 4), 0., 0.}, {.axes="zyx"});

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::empty<f32>(shape, option);
        const Array serial = noa::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            const auto ellipse = noa::geometry::Ellipse{center, radius, smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), ellipse.draw(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, ellipse.draw(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            const auto sphere = noa::geometry::Sphere{center, 91., smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), sphere.draw(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, sphere.draw(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            const auto rectangle = noa::geometry::Rectangle{center, radius, smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), rectangle.draw(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, rectangle.draw(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("cylinder") {
            const auto cylinder = noa::geometry::Cylinder{center, 45., 71., smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                noa::geometry::draw({}, serial.subregion(i), cylinder.draw(), matrices(0, 0, 0, i));
            noa::geometry::draw({}, batched, cylinder.draw(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }
    }
}

TEST_CASE("unified::draw, radius 0") {
    const auto array = Array<f64>({1, 1, 64, 64});

    noa::geometry::draw({}, array, noa::geometry::Ellipse{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.draw_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    noa::geometry::draw({}, array, noa::geometry::Ellipse{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.draw());
    REQUIRE(noa::allclose(noa::sum(array), 1.));

    noa::geometry::draw({}, array, noa::geometry::Sphere{.center=Vec{32., 32.}, .radius=0.}.draw_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    noa::geometry::draw({}, array, noa::geometry::Sphere{.center=Vec{32., 32.}, .radius=0.}.draw());
    REQUIRE(noa::allclose(noa::sum(array), 1.));

    noa::geometry::draw({}, array, noa::geometry::Rectangle{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.draw_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    noa::geometry::draw({}, array, noa::geometry::Rectangle{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.draw());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
}
