#include <noa/core/geometry/Transform.hpp>
#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/fft/Project.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::geometry::fft::insert_rasterize_3d", "[noa][unified][asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert_rasterize_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto grid_shape = parameters["grid_shape"].as<Shape4<i64>>();
        const auto target_shape = parameters["target_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto cutoff = parameters["cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto grid_filename = path / parameters["grid_filename"].as<Path>();

        const Float22 inv_scaling_matrix = noa::geometry::scale(1 / scale);
        Array<Float33> fwd_rotation_matrices(static_cast<i64>(rotate.size()));
        const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
        for (size_t i = 0; i < rotate.size(); ++i)
            fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, rotate[i], 0}), "zyx");

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            // Backward project.
            const Array slice_fft = noa::memory::linspace<f32>(slice_shape.rfft(), 1, 10, true, options);
            const Array grid_fft = noa::memory::zeros<f32>(grid_shape.rfft(), options);
            noa::geometry::fft::insert_rasterize_3d<fft::HC2HC>(
                    slice_fft, slice_shape, grid_fft, grid_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    cutoff, target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(grid_fft, grid_filename);
            } else {
                const Array asset_grid_fft = noa::io::load_data<f32>(grid_filename);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_grid_fft, grid_fft, 1e-5));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_rasterize_3d, remap", "[noa][unified]", f32, c32) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(math::deg2rad(Vec3<f32>{0, i * 2, 0}), "zyx");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        const Array slice_fft = noa::memory::linspace<TestType>(slice_shape.rfft(), 1, 10, true, options);
        const Array grid_fft0 = noa::memory::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();
        const Array grid_fft2 = grid_fft0.copy();

        // With centered slices.
        noa::geometry::fft::insert_rasterize_3d<fft::HC2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::geometry::fft::insert_rasterize_3d<fft::HC2H>(
                slice_fft, slice_shape, grid_fft1, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));

        // With non-centered slices.
        noa::memory::fill(grid_fft0, TestType{0});
        noa::memory::fill(grid_fft1, TestType{0});
        noa::geometry::fft::insert_rasterize_3d<fft::H2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::geometry::fft::insert_rasterize_3d<fft::H2H>(
                slice_fft, slice_shape, grid_fft1, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_rasterize_3d, value", "[noa][unified]", f32, c32, f64, c64) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 64, 64, 64};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(noa::math::deg2rad(Vec3<f32>{0, i * 2, 0}), "zyx");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        const auto value = test::Randomizer<TestType>(-10, 10).get();
        const Array slice_fft = noa::memory::fill<TestType>(slice_shape.rfft(), value, options);
        const Array grid_fft0 = noa::memory::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();

        noa::geometry::fft::insert_rasterize_3d<fft::HC2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::geometry::fft::insert_rasterize_3d<fft::HC2HC>(
                value, slice_shape, grid_fft1, grid_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_3d", "[noa][unified][asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert_interpolate_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto grid_shape = parameters["grid_shape"].as<Shape4<i64>>();
        const auto target_shape = parameters["target_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto cutoff = parameters["cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto slice_z_radius = parameters["slice_z_radius"].as<f32>();
        const auto grid_filename = path / parameters["grid_filename"].as<Path>();

        const Float22 fwd_scaling_matrix = noa::geometry::scale(scale);
        Array<Float33> inv_rotation_matrices(static_cast<i64>(rotate.size()));
        const auto inv_rotation_matrices_1d = inv_rotation_matrices.accessor_contiguous_1d();
        for (size_t i = 0; i < rotate.size(); ++i)
            inv_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, -rotate[i], 0}), "zyx");

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (inv_rotation_matrices.device() != device)
                inv_rotation_matrices = inv_rotation_matrices.to(device);

            // Backward project.
            const Array slice_fft = noa::memory::linspace<f32>(slice_shape.rfft(), 1, 10, true, options);
            const Array grid_fft = noa::memory::zeros<f32>(grid_shape.rfft(), options);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    slice_fft.eval(), slice_shape, grid_fft, grid_shape,
                    fwd_scaling_matrix, inv_rotation_matrices,
                    slice_z_radius, cutoff, target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(grid_fft, grid_filename);
                continue;
            }

            const Array asset_grid_fft = noa::io::load_data<f32>(grid_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_grid_fft, grid_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_interpolate_3d, value", "[noa][unified][asset]",
                   f32, c32, f64, c64) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 64, 64, 64};
    const auto slice_z_radius = 0.02f;

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, i * 2, 0}), "zyx");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        const auto value = test::Randomizer<TestType>(-10, 10).get();
        const Array slice_fft = noa::memory::fill<TestType>(slice_shape.rfft(), value, options);
        const Array grid_fft0 = noa::memory::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();

        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                value, slice_shape, grid_fft1, grid_shape,
                Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_interpolate_3d, using texture API and remap", "[noa][unified]",
                   f32, c32) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const i64 slice_count = GENERATE(1, 20);
    const auto slice_shape = Shape4<i64>{slice_count, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};
    const auto slice_z_radius = 0.02f;

    Array<Float33> inv_rotation_matrices(slice_shape[0]);
    const auto inv_rotation_matrices_1d = inv_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i) {
        inv_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, -static_cast<f32>(i) * 2, 0}), "zyx");
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (inv_rotation_matrices.device() != device)
            inv_rotation_matrices = inv_rotation_matrices.to(device);

        const Array slice_fft = noa::memory::linspace<TestType>(slice_shape.rfft(), 1, 10, true, options);
        const Array grid_fft0 = noa::memory::zeros<TestType>(grid_shape.rfft(), options);
        const Array grid_fft1 = grid_fft0.copy();
        const Array grid_fft2 = grid_fft0.copy();

        { // Texture
            const Texture<TestType> texture_slice_fft(
                    slice_fft, device, InterpMode::LINEAR, BorderMode::ZERO, 0.f, /*layered=*/ true);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    slice_fft, slice_shape, grid_fft0, grid_shape,
                    Float22{}, inv_rotation_matrices, slice_z_radius, 0.45f);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    texture_slice_fft, slice_shape, grid_fft1, grid_shape,
                    Float22{}, inv_rotation_matrices, slice_z_radius, 0.45f);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
        }

        { // Remap
            noa::memory::fill(grid_fft0, TestType{0});
            noa::memory::fill(grid_fft1, TestType{0});
            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    slice_fft, slice_shape, grid_fft0, grid_shape,
                    Float22{}, inv_rotation_matrices, slice_z_radius, 0.5f);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    slice_fft, slice_shape, grid_fft1, grid_shape,
                    Float22{}, inv_rotation_matrices, slice_z_radius, 0.5f);
            noa::fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::fft::extract_3d", "[noa][unified]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["extract_3d"];

    const auto grid_filename = path / tests["grid_filename"].as<Path>();
    const auto grid_shape = tests["grid_shape"].as<Shape4<i64>>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const Array<Float33> insert_fwd_rotation_matrices(36);
        const auto insert_fwd_rotation_matrices_1d = insert_fwd_rotation_matrices.accessor_contiguous_1d();
        for (i64 i = 0; i < insert_fwd_rotation_matrices.size(); ++i) {
            insert_fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{90, 0, i * 10}), "xyz", false);
        }
        const auto insert_slice_shape = Shape4<i64>{36, 1, grid_shape[2], grid_shape[3]};
        const Array grid_fft = noa::memory::empty<f32>(grid_shape.rfft());

        const Array slice_fft = noa::memory::linspace<f32>(insert_slice_shape.rfft(), 1, 20);
        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                slice_fft, insert_slice_shape, grid_fft, grid_shape,
                Float22{}, insert_fwd_rotation_matrices, 0.01f, 0.5f);
        noa::io::save(grid_fft, grid_filename);
    }

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto cutoff = parameters["cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto slice_filename = path / parameters["slice_filename"].as<Path>();

        const Float22 inv_scaling_matrix = geometry::scale(1 / scale);
        Array<Float33> fwd_rotation_matrices(static_cast<i64>(rotate.size()));
        const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
        for (size_t i = 0; i < rotate.size(); ++i)
            fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, rotate[i], 0}), "zyx");

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            const Array grid_fft = noa::io::load_data<f32>(grid_filename, false, options);
            const Array slice_fft = noa::memory::empty<f32>(slice_shape.rfft(), options);

            // Forward project.
            geometry::fft::extract_3d<fft::HC2HC>(
                    grid_fft, grid_shape, slice_fft, slice_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    cutoff, {}, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(slice_fft, slice_filename);
                continue;
            }

            Array asset_slice_fft = noa::io::load_data<f32>(slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_slice_fft, slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::extract_3d, using texture API and remap", "[noa][unified]",
                   f32, c32) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 128, 128};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, i * 2, 0}), "zyx");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        const Array grid_fft = noa::memory::linspace<TestType>(grid_shape.rfft(), -50, 50, true, options);
        const Array slice_fft0 = noa::memory::empty<TestType>(slice_shape.rfft(), options);
        const Array slice_fft1 = slice_fft0.copy();
        const Array slice_fft2 = slice_fft0.copy();

        const Texture<TestType> texture_grid_fft(grid_fft, device, InterpMode::LINEAR, BorderMode::ZERO, 0.f);
        noa::geometry::fft::extract_3d<fft::HC2HC>(
                texture_grid_fft, grid_shape, slice_fft0, slice_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::geometry::fft::extract_3d<fft::HC2H>(
                grid_fft, grid_shape, slice_fft1, slice_shape,
                Float22{}, fwd_rotation_matrices, 0.45f);
        noa::fft::remap(fft::H2HC, slice_fft1, slice_fft2, slice_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, slice_fft0, slice_fft2, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d", "[noa][unified][asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert_interpolate_and_extract_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto input_slice_shape = parameters["input_slice_shape"].as<Shape4<i64>>();
        const auto output_slice_shape = parameters["output_slice_shape"].as<Shape4<i64>>();
        const auto input_rotate = parameters["input_rotate"].as<std::vector<f32>>();
        const auto output_rotate = parameters["output_rotate"].as<std::vector<f32>>();
        const auto cutoff = parameters["cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto slice_z_radius = parameters["slice_z_radius"].as<f32>();
        const auto output_slice_filename = path / parameters["output_slice_filename"].as<Path>();

        Array<Float33> input_inv_rotation_matrices(static_cast<i64>(input_rotate.size()));
        Array<Float33> output_fwd_rotation_matrices(static_cast<i64>(output_rotate.size()));
        for (i64 i = 0; i < input_inv_rotation_matrices.size(); ++i) {
            input_inv_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, -input_rotate.data()[i], 0}), "zyx");
        }
        for (i64 i = 0; i < output_fwd_rotation_matrices.size(); ++i) {
            output_fwd_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, output_rotate.data()[i], 0}), "zyx");
        }

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (input_inv_rotation_matrices.device() != device)
                input_inv_rotation_matrices = input_inv_rotation_matrices.to(device);
            if (output_fwd_rotation_matrices.device() != device)
                output_fwd_rotation_matrices = output_fwd_rotation_matrices.to(device);

            const Array input_slice_fft = noa::memory::linspace<f32>(input_slice_shape.rfft(), -50, 50, true, options);
            const Array output_slice_fft = noa::memory::empty<f32>(output_slice_shape.rfft(), options);

            noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                    input_slice_fft, input_slice_shape,
                    output_slice_fft, output_slice_shape,
                    Float22{}, input_inv_rotation_matrices,
                    Float22{}, output_fwd_rotation_matrices,
                    slice_z_radius, false, cutoff, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output_slice_fft, output_slice_filename);
                continue;
            }

            const Array asset_output_slice_fft = io::load_data<f32>(output_slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_output_slice_fft, output_slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, using texture API and remap",
                   "[noa][unified]", f32, c32) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto input_slice_shape = Shape4<i64>{20, 1, 256, 256};
    const auto output_slice_shape = Shape4<i64>{5, 1, 256, 256};

    Array<Float33> input_inv_rotation_matrices(input_slice_shape[0]);
    for (i64 i = 0; i < input_slice_shape[0]; ++i) {
        input_inv_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, -static_cast<f32>(i) * 2, 0}), "zyx");
    }
    Array<Float33> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (i64 i = 0; i < output_slice_shape[0]; ++i) {
        output_fwd_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, static_cast<f32>(i) * 2 + 1, 0}), "zyx");
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = input_inv_rotation_matrices.to(device);
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = output_fwd_rotation_matrices.to(device);

        const Array input_slice_fft = noa::memory::linspace<TestType>(input_slice_shape.rfft(), -50, 50, true, options);
        const Texture<TestType> texture_input_slice_fft(
                input_slice_fft, device, InterpMode::LINEAR, BorderMode::ZERO, 0.f, /*layered=*/ true);
        const Array output_slice_fft0 = noa::memory::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft1 = noa::memory::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft2 = noa::memory::like(output_slice_fft1);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                texture_input_slice_fft, input_slice_shape,
                output_slice_fft0, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                0.01f, false, 0.8f);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
                input_slice_fft, input_slice_shape,
                output_slice_fft1, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                0.01f, false, 0.8f);
        noa::fft::remap(fft::H2HC, output_slice_fft1, output_slice_fft2, output_slice_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_slice_fft0, output_slice_fft2, 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, using value", "[noa][unified]",
                   f32, c32, f64, c64) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto input_slice_shape = Shape4<i64>{20, 1, 256, 256};
    const auto output_slice_shape = Shape4<i64>{5, 1, 256, 256};

    Array<Float33> input_inv_rotation_matrices(input_slice_shape[0]);
    for (i64 i = 0; i < input_slice_shape[0]; ++i) {
        input_inv_rotation_matrices(0, 0, 0, 1) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, -static_cast<f32>(i) * 2, 0}), "zyx");
    }
    Array<Float33> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (i64 i = 0; i < output_slice_shape[0]; ++i) {
        output_fwd_rotation_matrices(0, 0, 0, 1) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, static_cast<f32>(i) * 2 + 1, 0}), "zyx");
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = input_inv_rotation_matrices.to(device);
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = output_fwd_rotation_matrices.to(device);

        const auto value = test::Randomizer<TestType>(-10, 10).get();
        const Array input_slice_fft = noa::memory::fill<TestType>(input_slice_shape.rfft(), value, options);
        const Array output_slice_fft0 = noa::memory::empty<TestType>(output_slice_shape.rfft(), options);
        const Array output_slice_fft1 = noa::memory::empty<TestType>(output_slice_shape.rfft(), options);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input_slice_fft, input_slice_shape,
                output_slice_fft0, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                0.01f, false, 0.8f);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                value, input_slice_shape,
                output_slice_fft1, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                0.01f, false, 0.8f);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_slice_fft0, output_slice_fft1, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test rot", "[.]") {
    const std::vector<Device> devices{Device("cpu")};

    const auto input_slice_shape = Shape4<i64>{1, 1, 256, 256};
    const auto output_slice_shape = Shape4<i64>{11, 1, 256, 256};

    const Float33 input_slice_rotation = noa::geometry::euler2matrix(noa::math::deg2rad(Vec3<f32>{0}), "zyx", false);
    const Array<Float33> output_fwd_rotation_matrices(output_slice_shape[0]);
    f32 angle = -5;
    for (i64 i = 0; i < output_slice_shape[0]; ++i) {
        fmt::print("angle: {}\n", angle);
        output_fwd_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{angle, 0, 0}), "zyx", false);
        angle += 1;
    }

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const Array input_slice_fft = noa::memory::linspace<f32>(input_slice_shape.rfft(), -100, 100, true, options);
        const Array output_slice_fft0 = noa::memory::empty<f32>(output_slice_shape.rfft(), options);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input_slice_fft, input_slice_shape,
                output_slice_fft0, output_slice_shape,
                Float22{}, math::inverse(input_slice_rotation),
                Float22{}, output_fwd_rotation_matrices,
                0.002f, false, 0.5f);

        const auto directory = test::NOA_DATA_PATH / "geometry" / "fft" / "reproject_slices";
        noa::io::save(input_slice_fft, directory / "input_slice.mrc");
        noa::io::save(output_slice_fft0, directory / "output_slice.mrc");
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test multiplicity", "[.]") {
    const auto slice_shape = Shape4<i64>{20, 1, 512, 512};
    const auto grid_shape = Shape4<i64>{1, 512, 512, 512};
    const auto slice_z_radius = 0.0009765625f;

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.accessor_contiguous_1d();
    for (i64 i = 0; i < slice_shape[0]; ++i) {
        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{0, -20 + i * 3, 0}), "zyx");
    }

    const Array slices_fft = noa::memory::ones<f32>(slice_shape.rfft());
    const Array grid_fft = noa::memory::zeros<f32>(grid_shape.rfft());
    const Array grid2_fft = noa::memory::zeros<f32>(grid_shape.rfft());

    noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
            slices_fft, slice_shape, grid_fft, grid_shape,
            Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
    noa::io::save(grid_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_slice_radius.mrc");

//    noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
//            1.f, slice_shape, grid2_fft, grid_shape,
//            Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
//    noa::io::save(grid2_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_multiplicity.mrc");
//
//    noa::ewise_binary(grid_fft, grid2_fft, grid_fft, noa::divide_safe_t{});
//    noa::io::save(grid_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_multiplicity_after.mrc");
}
