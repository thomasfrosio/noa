#include <noa/xform/core/Euler.hpp>
#include <noa/xform/Draw.hpp>
#include <noa/Runtime.hpp>
#include <noa/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nx = ::noa::xform;

namespace {
    struct ApplyInvertedMask {
        f32 cvalue;
        constexpr void operator()(f32 input, f32 mask, f32& output) const {
            output = input * (cvalue - mask);
        }
    };
}

TEST_CASE("xform::sphere(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 2>>();
        const auto radius = test["radius"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto sphere = nx::Sphere{
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
            nx::draw_2d({}, asset, sphere.get());
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_2d({}, result, sphere.get<f32>());
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, -10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_2d(data, data, sphere.get<f32>());
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, -10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_2d(data, data, inverted_sphere.get<f32>());
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("xform::rectangle(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 2>>();
        const auto radius = test["radius"].as<Vec<f64, 2>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = nx::rotate(noa::deg2rad(-test["angle"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto rectangle = nx::Rectangle{
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
            nx::draw_2d({}, asset, rectangle.get(), inv_matrix);
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_2d({}, result, rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_2d(data, data, rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_2d(data, data, inverted_rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("xform::ellipse(), 2d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back( "gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test2D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 2>>();
        const auto radius = test["radius"].as<Vec<f64, 2>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = nx::rotate(noa::deg2rad(-test["angle"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto ellipse = nx::Ellipse{
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
            nx::draw_2d({}, asset, ellipse.get(), inv_matrix);
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_2d({}, result, ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_2d(data, data, ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_2d(data, data, inverted_ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-4));
        }
    }
}

TEST_CASE("xform::sphere, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["sphere"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 3>>();
        const auto radius = test["radius"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto sphere = nx::Sphere{
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
            nx::draw_3d({}, asset, sphere.get());
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_3d({}, result, sphere.get());
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_3d(data, data, sphere.get());
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_3d(data, data, inverted_sphere.get());
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("xform::rectangle, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["rectangle"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 3>>();
        const auto radius = test["radius"].as<Vec<f64, 3>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = nx::rotate_y(-noa::deg2rad(test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto rectangle = nx::Rectangle{
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
            nx::draw_3d({}, asset, rectangle.get(), inv_matrix);
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_3d({}, result, rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_3d(data, data, rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_3d(data, data, inverted_rectangle.get(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEST_CASE("xform::ellipse, 3d", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["test3D"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 3>>();
        const auto radius = test["radius"].as<Vec<f64, 3>>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = nx::rotate_y(-noa::deg2rad(test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto ellipse = nx::Ellipse{
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
            nx::draw_3d({}, asset, ellipse.get(), inv_matrix);
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_3d({}, result, ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_3d(data, data, ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 1e-4));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_3d(data, data, inverted_ellipse.get(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 1e-4));
        }
    }
}

TEST_CASE("xform::cylinder", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector devices{Device("cpu")};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::noa_data_path() / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["cylinder"]["test"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto invert = test["invert"].as<bool>();
        const auto shape = test["shape"].as<Shape4>();
        const auto center = test["center"].as<Vec<f64, 3>>();
        const auto radius = test["radius"].as<f64>();
        const auto length = test["length"].as<f64>();
        const auto taper = test["taper"].as<f64>();
        const auto inv_matrix = nx::rotate_y(noa::deg2rad(-test["tilt"].as<f64>()));
        const auto cvalue = test["cvalue"].as<f64>();
        const auto filename_expected = path_base / test["expected"].as<Path>();

        const auto cylinder = nx::Cylinder{
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
            nx::draw_3d({}, asset, cylinder.get(), inv_matrix);
            noa::write_image(asset, filename_expected);
            continue;
        }

        for (auto device: devices) {
            INFO(device);
            const auto option = ArrayOption(device, Allocator::MANAGED);
            const auto asset = noa::read_image<f32, 4>(filename_expected, {}, option).data;
            if (asset.shape() != shape)
                FAIL("asset shape is not correct");

            // Save shape into the output.
            const auto result = noa::empty<f32>(shape, option);
            nx::draw_3d({}, result, cylinder.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, asset, 5e-5));

            // Apply the shape onto the input, in-place.
            const auto data = noa::random(noa::Uniform{-10.f, 10.f}, shape, option);
            noa::ewise(noa::wrap(result, data), result, noa::Multiply{});
            nx::draw_3d(data, data, cylinder.get(), inv_matrix);
            REQUIRE(test::allclose_abs(result, data, 5e-5));

            // Apply the shape onto the input, in-place, invert.
            noa::randomize(noa::Uniform{-10.f, 10.f}, data);
            noa::ewise(noa::wrap(data, asset), asset, ApplyInvertedMask{static_cast<f32>(cvalue)});
            nx::draw_3d(data, data, inverted_cylinder.get(), inv_matrix);
            REQUIRE(test::allclose_abs(asset, data, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("xform::sphere, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched<isize, 2>(2).push_front<2>(1);
    const auto center_2d = (shape.vec.filter(2, 3) / 2).as<f64>();
    const auto center_3d = center_2d.push_front(0);

    constexpr f64 radius = 20.;
    constexpr f64 edge_size = 5.;
    constexpr f64 angle = noa::deg2rad(-67.);
    const auto inv_transform_2d = nx::rotate(-angle);
    const auto inv_transform_3d = nx::euler2matrix(Vec{angle, 0., 0.}).transpose();
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

        const auto sphere_2d = nx::Sphere{
            .center=center_2d,
            .radius=radius,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto sphere_3d = nx::Sphere{
            .center=center_3d,
            .radius=radius,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        nx::draw_2d(input, output_2d, sphere_2d.get(), inv_transform_2d);
        nx::draw_3d(input, output_3d, sphere_3d.get(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("xform::rectangle, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched<isize, 2>(2).push_front<2>(1);
    const auto center_2d = (shape.vec.filter(2, 3) / 2).as<f64>();
    const auto center_3d = center_2d.push_front(0);

    constexpr auto radius_3d = Vec{1., 20., 20.};
    constexpr auto radius_2d = Vec{20., 20.};
    constexpr f64 edge_size = 5.;
    constexpr f64 angle = noa::deg2rad(-67.);
    const auto inv_transform_2d = nx::rotate(-angle);
    const auto inv_transform_3d = nx::euler2matrix(Vec{angle, 0., 0.}).transpose();

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

        const auto rectangle_2d = nx::Rectangle{
            .center=center_2d,
            .radius=radius_2d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto rectangle_3d = nx::Rectangle{
            .center=center_3d,
            .radius=radius_3d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        nx::draw_2d(input, output_2d, rectangle_2d.get(), inv_transform_2d);
        nx::draw_3d(input, output_3d, rectangle_3d.get(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("xform::ellipse, 2d matches 3d", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched<isize, 2>(2).push_front<2>(1);
    const auto center_2d = (shape.vec.filter(2, 3) / 2).as<f64>();
    const auto center_3d = center_2d.push_front(0);

    constexpr auto radius_3d = Vec{1., 40., 25.};
    constexpr auto radius_2d = Vec{40., 25.};
    constexpr f64 edge_size = 5;
    constexpr f64 angle = noa::deg2rad(25.);
    const auto inv_transform_2d = nx::rotate(-angle);
    const auto inv_transform_3d = nx::euler2matrix(Vec{angle, 0., 0.}).transpose();
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

        const auto ellipse_2d = nx::Ellipse{
            .center=center_2d,
            .radius=radius_2d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };
        const auto ellipse_3d = nx::Ellipse{
            .center=center_3d,
            .radius=radius_3d,
            .smoothness=edge_size,
            .cvalue=1.,
            .invert=invert,
        };

        nx::draw_2d(input, output_2d, ellipse_2d.get(), inv_transform_2d);
        nx::draw_3d(input, output_3d, ellipse_3d.get(), inv_transform_3d);
        REQUIRE(test::allclose_abs(output_2d, output_3d, 1e-4));
    }
}

TEMPLATE_TEST_CASE("xform::ellipse, 2d affine", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(2);
    const auto ellipse = nx::Ellipse{
        .center = (shape.filter(2, 3) / 2).vec.as<f64>(),
        .radius = Vec{40., 25.},
        .smoothness = 5.,
    };

    constexpr f64 angle = noa::deg2rad(25.);
    const auto inv_matrix_linear = nx::rotate(-angle);
    const auto inv_matrix_affine =
            nx::translate(ellipse.center) *
            nx::affine(nx::rotate(-angle)) *
            nx::translate(-ellipse.center);

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random<TestType>(noa::Uniform{-5.f, 5.f}, shape, option);
        const auto output_linear = noa::empty<TestType>(shape, option);
        const auto output_affine = noa::empty<TestType>(shape, option);

        nx::draw_2d(input, output_linear, ellipse.get(), inv_matrix_linear);
        nx::draw_2d(input, output_affine, ellipse.get(), inv_matrix_affine);
        REQUIRE(test::allclose_abs(output_linear, output_affine, 1e-4));
    }
}

TEMPLATE_TEST_CASE("xform::ellipse, 3d affine", "", f32, f64, c32, c64) {
    const auto shape = test::random_shape_batched(3);
    const auto ellipse = nx::Ellipse{
        .center = (shape.filter(1, 2, 3) / 2).vec.as<f64>(),
        .radius = Vec{10., 15., 25.},
        .smoothness = 5.,
    };

    constexpr auto angles = noa::deg2rad(Vec{25., 15., 0.});
    const auto inv_matrix_linear = nx::euler2matrix(angles).transpose();
    const auto inv_matrix_affine =
            nx::translate(ellipse.center) *
            nx::affine(inv_matrix_linear) *
            nx::translate(-ellipse.center);

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const auto input = noa::random(noa::Uniform<TestType>{-5., 5.}, shape, option);
        const auto output_linear = noa::empty<TestType>(shape, option);
        const auto output_affine = noa::empty<TestType>(shape, option);

        nx::draw_3d(input, output_linear, ellipse.get(), inv_matrix_linear);
        nx::draw_3d(input, output_affine, ellipse.get(), inv_matrix_affine);
        REQUIRE(test::allclose_abs(output_linear, output_affine, 1e-4));
    }
}

TEST_CASE("xform::draw, 2d batched") {
    constexpr auto shape = Shape3{10, 124, 115};
    constexpr auto center = shape.pop_front().vec.as<f64>() / 2;
    constexpr auto radius = Vec{42., 12.};
    constexpr auto smoothness = 10.;

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    Array matrices = noa::empty<Mat<f32, 2, 2>, 1>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices.span_1d()[i] = nx::rotate(static_cast<f32>(i * 4));

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::empty<f32>(shape, option);
        const Array serial = noa::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            const auto ellipse = nx::Ellipse{center, radius, smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                nx::draw_2d({}, serial.subregion(i), ellipse.get<f32>(), matrices.span_1d()[i]);
            nx::draw_2d({}, batched, ellipse.get<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            const auto sphere = nx::Sphere{center, 126., smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                nx::draw_2d({}, serial.subregion(i), sphere.get<f32>(), matrices.span_1d()[i]);
            nx::draw_2d({}, batched, sphere.get<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            const auto rectangle = nx::Rectangle{center, radius, smoothness};
            for (i64 i{}; i < shape[0]; ++i)
                nx::draw_2d({}, serial.subregion(i), rectangle.get<f32>(), matrices.span_1d()[i]);
            nx::draw_2d({}, batched, rectangle.get<f32>(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }
    }
}

TEST_CASE("xform::draw, 3d batched") {
    constexpr auto shape = Shape4{4, 55, 65, 58};
    constexpr auto center = shape.pop_front().vec.as<f64>() / 2;
    constexpr auto radius = Vec{9., 21., 8.};
    constexpr auto smoothness = 10.;

    std::vector devices{Device("cpu")};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    Array matrices = noa::empty<Mat<f64, 3, 3>, 1>(shape[0]);
    for (i64 i = 0; i < shape[0]; ++i)
        matrices.span_1d()[i] = nx::euler2matrix(Vec{static_cast<f64>(i * 4), 0., 0.}, {.axes="zyx"});

    for (auto device: devices) {
        INFO(device);
        const auto option = ArrayOption(device, Allocator::MANAGED);
        const Array batched = noa::empty<f32>(shape, option);
        const Array serial = noa::empty<f32>(shape, option);
        const Array matrices_device = matrices.to(option); // just for simplicity

        AND_THEN("ellipse") {
            const auto ellipse = nx::Ellipse{center, radius, smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                nx::draw_3d({}, serial.subregion(i), ellipse.get(), matrices.span_1d()[i]);
            nx::draw_3d({}, batched, ellipse.get(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("sphere") {
            const auto sphere = nx::Sphere{center, 91., smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                nx::draw_3d({}, serial.subregion(i), sphere.get(), matrices.span_1d()[i]);
            nx::draw_3d({}, batched, sphere.get(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("rectangle") {
            const auto rectangle = nx::Rectangle{center, radius, smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                nx::draw_3d({}, serial.subregion(i), rectangle.get(), matrices.span_1d()[i]);
            nx::draw_3d({}, batched, rectangle.get(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }

        AND_THEN("cylinder") {
            const auto cylinder = nx::Cylinder{center, 45., 71., smoothness};
            for (i64 i = 0; i < shape[0]; ++i)
                nx::draw_3d({}, serial.subregion(i), cylinder.get(), matrices.span_1d()[i]);
            nx::draw_3d({}, batched, cylinder.get(), matrices_device);
            REQUIRE(test::allclose_abs_safe(batched, serial, 1e-6));
        }
    }
}

TEST_CASE("draw, radius 0") {
    const auto array = Array<f64, 2>({64, 64});

    nx::draw_2d({}, array, nx::Ellipse{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.get_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    nx::draw_2d({}, array, nx::Ellipse{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.get());
    REQUIRE(noa::allclose(noa::sum(array), 1.));

    nx::draw_2d({}, array, nx::Sphere{.center=Vec{32., 32.}, .radius=0.}.get_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    nx::draw_2d({}, array, nx::Sphere{.center=Vec{32., 32.}, .radius=0.}.get());
    REQUIRE(noa::allclose(noa::sum(array), 1.));

    nx::draw_2d({}, array, nx::Rectangle{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.get_binary());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
    nx::draw_2d({}, array, nx::Rectangle{.center=Vec{32., 32.}, .radius=Vec{0., 0.}}.get());
    REQUIRE(noa::allclose(noa::sum(array), 1.));
}
