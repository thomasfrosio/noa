#include <noa/core/geometry/Transform.hpp>
#include <noa/core/geometry/Euler.hpp>
#include <noa/unified/geometry/fft/Project.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <noa/Math.hpp>
#include <noa/Signal.hpp>
#include <noa/FFT.hpp>
#include <noa/unified/Ewise.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::geometry::fft::insert_rasterize_3d", "[noa][unified][asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["fourier_insert_rasterize_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4<i64>>();
        const auto target_shape = parameters["target_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto volume_filename = path / parameters["volume_filename"].as<Path>();

        const Float22 inv_scaling_matrix = noa::geometry::scale(1 / scale);
        Array<Float33> fwd_rotation_matrices(static_cast<i64>(rotate.size()));
        const auto fwd_rotation_matrices_span = fwd_rotation_matrices.span();
        for (size_t i = 0; i < rotate.size(); ++i)
            fwd_rotation_matrices_span[i] = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, rotate[i], 0}), "zyx");

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            // Backward project.
            const Array slice_fft = noa::memory::linspace<f32>(slice_shape.rfft(), 1, 10, true, options);
            const Array volume_fft = noa::memory::zeros<f32>(volume_shape.rfft(), options);
            noa::geometry::fft::insert_rasterize_3d<fft::HC2HC>(
                    slice_fft, slice_shape, volume_fft, volume_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    fftfreq_cutoff, target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(volume_fft, volume_filename);
            } else {
                const Array asset_volume_fft = noa::io::load_data<f32>(volume_filename);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_volume_fft, volume_fft, 1e-5));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert_rasterize_3d, remap", "[noa][unified]", f32, c32) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    {
        i64 i{0};
        for (auto& matrix: fwd_rotation_matrices.span()) // TODO C++20: move i in loop
            matrix = noa::geometry::euler2matrix(math::deg2rad(Vec3<f32>{0, i++ * 2, 0}), "zyx");
    }

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
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 64, 64, 64};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_span = fwd_rotation_matrices.span();
    for (i64 i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices_span[i] = noa::geometry::euler2matrix(noa::math::deg2rad(Vec3<f32>{0, i * 2, 0}), "zyx");

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
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["fourier_insert_interpolate_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4<i64>>();
        const auto target_shape = parameters["target_shape"].as<Shape4<i64>>();
        const auto scale = Vec2<f32>(parameters["scale"].as<f32>());
        const auto rotate = parameters["rotate"].as<std::vector<f32>>();
        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto fftfreq_sinc = parameters["fftfreq_sinc"].as<f32>();
        const auto fftfreq_blackman = parameters["fftfreq_blackman"].as<f32>();
        const auto volume_filename = path / parameters["volume_filename"].as<Path>();

        const Float22 fwd_scaling_matrix = noa::geometry::scale(scale);
        Array<Float33> inv_rotation_matrices(static_cast<i64>(rotate.size()));
        const auto inv_rotation_matrices_1d = inv_rotation_matrices.span();
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
            const Array volume_fft = noa::memory::zeros<f32>(volume_shape.rfft(), options);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    slice_fft.eval(), slice_shape, volume_fft, volume_shape,
                    fwd_scaling_matrix, inv_rotation_matrices,
                    {fftfreq_sinc, fftfreq_blackman}, fftfreq_cutoff,
                    target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(volume_fft, volume_filename);
                continue;
            }

            const Array asset_volume_fft = noa::io::load_data<f32>(volume_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_volume_fft, volume_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE(
        "unified::geometry::fft::insert_interpolate_3d, value", "[noa][unified][asset]",
        f32, c32, f64, c64
) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 64, 64, 64};
    const auto fftfreq_sinc = 0.02f;
    const auto fftfreq_blackman = 0.06f;

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.span();
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
                Float22{}, fwd_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.45f);
        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                value, slice_shape, grid_fft1, grid_shape,
                Float22{}, fwd_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.45f);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
    }
}

TEMPLATE_TEST_CASE(
        "unified::geometry::fft::insert_interpolate_3d, using texture API and remap", "[noa][unified]",
        f32, c32
) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const i64 slice_count = GENERATE(1, 20);
    const auto slice_shape = Shape4<i64>{slice_count, 1, 64, 64};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};
    const auto fftfreq_sinc = 0.02f;
    const auto fftfreq_blackman = 0.06f;

    Array<Float33> inv_rotation_matrices(slice_shape[0]);
    const auto inv_rotation_matrices_1d = inv_rotation_matrices.span();
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
                    Float22{}, inv_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.45f);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    texture_slice_fft, slice_shape, grid_fft1, grid_shape,
                    Float22{}, inv_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.45f);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
        }

        { // Remap
            noa::memory::fill(grid_fft0, TestType{0});
            noa::memory::fill(grid_fft1, TestType{0});
            noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                    slice_fft, slice_shape, grid_fft0, grid_shape,
                    Float22{}, inv_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.5f);
            noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
                    slice_fft, slice_shape, grid_fft1, grid_shape,
                    Float22{}, inv_rotation_matrices, {fftfreq_sinc, fftfreq_blackman}, 0.5f);
            noa::fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::fft::extract_3d", "[noa][unified]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["fourier_extract_3d"];

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        const auto asset_0 = tests["volumes"][0];
        const auto volume_filename = path / asset_0["filename"].as<Path>();
        const auto volume_shape = asset_0["shape"].as<Shape4<i64>>();

        const auto insert_inv_rotation_matrices = noa::memory::empty<Float33>(36);
        i64 i = 0;
        for (auto& matrix: insert_inv_rotation_matrices.span()) {
            matrix = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{90, 0, i++ * 10}), "xyz", false).transpose();
        }
        const auto insert_slice_shape = Shape4<i64>{36, 1, volume_shape[2], volume_shape[3]};
        const Array volume_fft = noa::memory::empty<f32>(volume_shape.rfft());

        const Array slice_fft = noa::memory::linspace<f32>(insert_slice_shape.rfft(), 1, 20);
        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                slice_fft, insert_slice_shape, volume_fft, volume_shape,
                Float22{}, insert_inv_rotation_matrices, {-1, 0.1f}, 0.5f);
        noa::io::save(volume_fft, volume_filename);
    }

    if constexpr (COMPUTE_ASSETS) {
        const auto asset_0 = tests["volumes"][1];
        const auto volume_filename = path / asset_0["filename"].as<Path>();
        const auto volume_shape = asset_0["shape"].as<Shape4<i64>>();
        const auto volume_fft = noa::memory::empty<f32>(volume_shape.rfft());

        const auto slice_shape = Shape4<i64>{1, 1, volume_shape[2], volume_shape[3]};
        const auto slice_fft = noa::memory::linspace<f32>(slice_shape.rfft(), 1, 20);
        noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
                slice_fft, slice_shape, volume_fft, volume_shape,
                Float22{}, Float33{}, {0.0234375, 1}, 0.5f);
        noa::io::save(volume_fft, volume_filename);
    }

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);
        const YAML::Node& parameters = tests["tests"][nb];

        const auto volume_filename = path / parameters["volume_filename"].as<Path>();
        const auto volume_shape = parameters["volume_shape"].as<Shape4<i64>>();

        const auto slice_filename = path / parameters["slice_filename"].as<Path>();
        const auto slice_shape = parameters["slice_shape"].as<Shape4<i64>>();
        const auto slice_scale = Vec2<f32>(parameters["slice_scale"].as<f32>());
        const auto slice_rotate = parameters["slice_rotate"].as<std::vector<f32>>();

        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f32>();
        const auto fftfreq_z_sinc = parameters["fftfreq_z_sinc"].as<f32>();
        const auto fftfreq_z_blackman = parameters["fftfreq_z_blackman"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());

        const Float22 inv_scaling_matrix = noa::geometry::scale(1 / slice_scale);
        auto fwd_rotation_matrices = noa::memory::empty<Float33>(static_cast<i64>(slice_rotate.size()));
        size_t i{0};
        for (auto& matrix: fwd_rotation_matrices.span())
            matrix = noa::geometry::euler2matrix(noa::math::deg2rad(Vec3<f32>{0, slice_rotate[i++], 0}), "zyx");

        for (auto& device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            const Array volume_fft = noa::io::load_data<f32>(volume_filename, false, options);
            const Array slice_fft = noa::memory::empty<f32>(slice_shape.rfft(), options);

            // Forward project.
            noa::geometry::fft::extract_3d<fft::HC2HC>(
                    volume_fft, volume_shape, slice_fft, slice_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    {fftfreq_z_sinc, fftfreq_z_blackman}, fftfreq_cutoff,
                    {}, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(slice_fft, slice_filename);
                continue;
            }

            const Array asset_slice_fft = noa::io::load_data<f32>(slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_slice_fft, slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::extract_3d, using texture API and remap", "[noa][unified]", f32, c32) {
    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = Shape4<i64>{20, 1, 128, 128};
    const auto grid_shape = Shape4<i64>{1, 128, 128, 128};

    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.span();
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
                Float22{}, fwd_rotation_matrices, {}, 0.45f);
        noa::geometry::fft::extract_3d<fft::HC2H>(
                grid_fft, grid_shape, slice_fft1, slice_shape,
                Float22{}, fwd_rotation_matrices, {}, 0.45f);
        noa::fft::remap(fft::H2HC, slice_fft1, slice_fft2, slice_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, slice_fft0, slice_fft2, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d", "[noa][unified][asset]") {
    const Path path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["fourier_insert_interpolate_and_extract_3d"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto input_slice_shape = parameters["input_slice_shape"].as<Shape4<i64>>();
        const auto output_slice_shape = parameters["output_slice_shape"].as<Shape4<i64>>();
        const auto input_rotate = parameters["input_rotate"].as<std::vector<f32>>();
        const auto output_rotate = parameters["output_rotate"].as<std::vector<f32>>();
        const auto fftfreq_cutoff = parameters["fftfreq_cutoff"].as<f32>();
        const auto fftfreq_input_sinc = parameters["fftfreq_input_sinc"].as<f32>();
        const auto fftfreq_input_blackman = parameters["fftfreq_input_blackman"].as<f32>();
        const auto fftfreq_z_sinc = parameters["fftfreq_z_sinc"].as<f32>();
        const auto fftfreq_z_blackman = parameters["fftfreq_z_blackman"].as<f32>();
        const auto ews_radius = Vec2<f32>(parameters["ews_radius"].as<f32>());
        const auto output_slice_filename = path / parameters["output_slice_filename"].as<Path>();

        Array<Float33> input_inv_rotation_matrices(static_cast<i64>(input_rotate.size()));
        Array<Float33> output_fwd_rotation_matrices(static_cast<i64>(output_rotate.size()));
        for (i64 i = 0; i < input_inv_rotation_matrices.ssize(); ++i) {
            input_inv_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                    noa::math::deg2rad(Vec3<f32>{0, -input_rotate.data()[i], 0}), "zyx");
        }
        for (i64 i = 0; i < output_fwd_rotation_matrices.ssize(); ++i) {
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
                    {fftfreq_input_sinc, fftfreq_input_blackman},
                    {fftfreq_z_sinc, fftfreq_z_blackman},
                    false, fftfreq_cutoff, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(output_slice_fft, output_slice_filename);
                continue;
            }

            const Array asset_output_slice_fft = io::load_data<f32>(output_slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_output_slice_fft, output_slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE(
        "unified::geometry::fft::insert_interpolate_and_extract_3d, using texture API and remap",
        "[noa][unified]",
        f32, c32
) {
    std::vector<Device> devices{Device("cpu")};
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
                {0.01f}, {}, false, 0.8f);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
                input_slice_fft, input_slice_shape,
                output_slice_fft1, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                {0.01f}, {}, false, 0.8f);
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
                {0.01f}, {}, false, 0.8f);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                value, input_slice_shape,
                output_slice_fft1, output_slice_shape,
                Float22{}, input_inv_rotation_matrices,
                Float22{}, output_fwd_rotation_matrices,
                {0.01f}, {}, false, 0.8f);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_slice_fft0, output_slice_fft1, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test rotation", "[noa][unified][assets]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto input_slice_shape = Shape4<i64>{1, 1, 256, 256};
    const auto output_slice_shape = Shape4<i64>{11, 1, 256, 256};

    const auto input_slice_rotation = Float33{1};
    const auto output_fwd_rotation_matrices = Array<Float33>(output_slice_shape[0]);
    f32 angle = -20;
    for (i64 i = 0; i < output_slice_shape[0]; ++i) {
        output_fwd_rotation_matrices(0, 0, 0, i) = noa::geometry::euler2matrix(
                noa::math::deg2rad(Vec3<f32>{angle, 0, 0}), "zyx", false);
        angle += 4;
    }

    const auto directory = test::NOA_DATA_PATH / "geometry" / "fft";
    const auto output_filename = directory / YAML::LoadFile(directory / "tests.yaml")
            ["fourier_insert_interpolate_and_extract_3d_others"]
            ["test_rotation_filename"]
            .as<Path>();

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        INFO(device);

        const Array input_slice_fft = noa::memory::linspace<f32>(input_slice_shape.rfft(), -100, 100, true, options);
        const Array output_slice_fft0 = noa::memory::empty<f32>(output_slice_shape.rfft(), options);

        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
                input_slice_fft, input_slice_shape,
                output_slice_fft0, output_slice_shape,
                Float22{}, input_slice_rotation,
                Float22{}, output_fwd_rotation_matrices.to(options),
                {0.002f}, {}, false, 0.498f);

        if constexpr (COMPUTE_ASSETS) {
            noa::io::save(output_slice_fft0, output_filename);
        } else {
            const auto asset = noa::io::load_data<f32>(output_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset, output_slice_fft0, 1e-4));
        }
    }
}

//TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test multiplicity", "[.]") {
//    const auto slice_shape = Shape4<i64>{20, 1, 512, 512};
//    const auto grid_shape = Shape4<i64>{1, 512, 512, 512};
//    const auto slice_z_radius = 0.0009765625f;
//
//    Array<Float33> fwd_rotation_matrices(slice_shape[0]);
//    const auto fwd_rotation_matrices_1d = fwd_rotation_matrices.span();
//    for (i64 i = 0; i < slice_shape[0]; ++i) {
//        fwd_rotation_matrices_1d[i] = noa::geometry::euler2matrix(
//                noa::math::deg2rad(Vec3<f32>{0, -20 + i * 3, 0}), "zyx");
//    }
//
//    const Array slices_fft = noa::memory::ones<f32>(slice_shape.rfft());
//    const Array grid_fft = noa::memory::zeros<f32>(grid_shape.rfft());
//    const Array grid2_fft = noa::memory::zeros<f32>(grid_shape.rfft());
//
//    noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
//            slices_fft, slice_shape, grid_fft, grid_shape,
//            Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
//    noa::io::save(grid_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_slice_radius.mrc");
//
////    noa::geometry::fft::insert_interpolate_3d<fft::HC2HC>(
////            1.f, slice_shape, grid2_fft, grid_shape,
////            Float22{}, fwd_rotation_matrices, slice_z_radius, 0.45f);
////    noa::io::save(grid2_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_multiplicity.mrc");
////
////    noa::ewise_binary(grid_fft, grid2_fft, grid_fft, noa::divide_safe_t{});
////    noa::io::save(grid_fft, test::NOA_DATA_PATH / "geometry" / "fft" / "test_multiplicity_after.mrc");
//}
//
//TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test tilt extraction", "[.]") {
//    const auto output_directory = test::NOA_DATA_PATH / "slides_away_day";
//
//    std::vector<f32> tilts;
//    for (i64 i = -60; i <= 60; i += 6)
//        if (i != 0)
//            tilts.push_back(static_cast<f32>(i));
//
//    const auto slice_shape = Shape4<i64>{1, 1, 1024, 1024};
//    const auto stack_shape = Shape4<i64>{tilts.size(), 1, 1024, 1024};
//
//    // Input slices
//    const auto input_slice_rotation = noa::memory::empty<Float33>(stack_shape[0]);
//    for (i64 i = 0; i < input_slice_rotation.ssize(); ++i) {
//        input_slice_rotation(0, 0, 0, i) = noa::geometry::euler2matrix(
//                noa::math::deg2rad(Vec3<f32>{0, tilts[static_cast<size_t>(i)], 0}), "zyx", false).inverse();
//    }
//
//    // Output slice
//    const auto output_slice_rotation = noa::geometry::euler2matrix(
//            noa::math::deg2rad(Vec3<f32>{0, 0, 0}), "zyx", false);
//
//    auto stack_rfft = noa::memory::ones<f32>(slice_shape.rfft());
//    auto output_slice_rfft = noa::memory::like(stack_rfft);
//
//    stack_rfft = noa::indexing::broadcast(stack_rfft, stack_shape.rfft()); // treat as stack
//
//    noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2HC>(
//            stack_rfft, stack_shape,
//            output_slice_rfft, slice_shape,
//            Float22{}, input_slice_rotation,
//            Float22{}, output_slice_rotation,
//            0.004f);
//
//    noa::signal::fft::lowpass<fft::HC2HC>(output_slice_rfft, output_slice_rfft, slice_shape, 0.49f, 0.01f);
//    noa::ewise_unary(output_slice_rfft, output_slice_rfft,
//                     [](f32 value) { return std::min(value, 1.f); });
//
//    const auto output_slice_fft = noa::memory::empty<f32>(slice_shape);
//    noa::fft::remap(noa::fft::Remap::HC2FC, output_slice_rfft, output_slice_fft, slice_shape);
//    noa::io::save(noa::ewise_unary(output_slice_fft, noa::abs_one_log_t{}), output_directory / "output_ps.mrc");
//}
//
//TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test recons 0deg", "[.]") {
//    const auto stack_path = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/tilt1_coarse_aligned.mrc");
//    const auto output_directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf");
//
//    auto file = noa::io::ImageFile(stack_path, noa::io::READ);
//    const auto n_slices = file.shape()[0];
//    const auto slice_shape = file.shape().set<0>(1);
//    const auto stack_shape = file.shape().set<0>(n_slices - 1);
//
//    // Load every image except the 0deg.
//    auto stack = noa::memory::empty<f32>(stack_shape);
//    i64 index{0};
//    f32 angle = -57;
//    std::vector<f32> tilts;
//    for (i64 i = 0; i < n_slices; ++i, angle += 3) {
//        if (std::abs(angle) < 1e-6f)
//            continue;
//        file.read_slice(stack.view().subregion(index), i);
//        tilts.push_back(angle);
//        ++index;
//    }
//
//    // Input slices
//    const auto input_slice_rotation = noa::memory::empty<Float33>(stack_shape[0]);
//    for (i64 i = 0; i < input_slice_rotation.ssize(); ++i) {
//        input_slice_rotation(0, 0, 0, i) = noa::geometry::euler2matrix(
//                noa::math::deg2rad(Vec3<f32>{0, tilts[static_cast<size_t>(i)], 0}), "zyx", false).inverse();
//    }
//
//    // Output slice
//    const auto output_slice_rotation = noa::geometry::euler2matrix(
//            noa::math::deg2rad(Vec3<f32>{0, 0, 0}), "zyx", false);
//
//    auto stack_rfft = noa::fft::r2c(stack);
//    auto output_slice_rfft = noa::memory::empty<c32>(slice_shape.rfft());
//    auto output_slice_rfft_weight = noa::memory::empty<f32>(slice_shape.rfft());
//
//    noa::fft::remap(fft::H2HC, stack_rfft, stack_rfft, stack_shape);
//
//    const auto rotation_center = stack_shape.vec().filter(2,3).as<f32>() / 2;
//    noa::signal::fft::phase_shift_2d<fft::HC2HC>(stack_rfft, stack_rfft, stack_shape, -rotation_center);
//
//    const auto max_output_size = static_cast<f32>(noa::math::min(slice_shape.filter(2, 3)));
//    auto slice_z_radius =  1.f / max_output_size;
//    for (auto i: noa::irange(10)) {
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                stack_rfft, stack_shape,
//                output_slice_rfft, slice_shape,
//                Float22{}, input_slice_rotation,
//                Float22{}, output_slice_rotation,
//                slice_z_radius);
//        noa::geometry::fft::insert_interpolate_and_extract_3d<fft::HC2H>(
//                1.f, stack_shape,
//                output_slice_rfft_weight, slice_shape,
//                Float22{}, input_slice_rotation,
//                Float22{}, output_slice_rotation,
//                slice_z_radius);
//
//        noa::ewise_binary(output_slice_rfft, output_slice_rfft_weight, output_slice_rfft,
//                          [](c32 lhs, f32 rhs) {
//                              if (rhs > 1)
//                                  lhs /= rhs;
//                              return lhs;
//                          });
//        noa::signal::fft::phase_shift_2d<fft::H2H>(output_slice_rfft, output_slice_rfft, slice_shape, rotation_center);
//
//        // Save PS.
//        const auto output_slice_rfft_weight_full = noa::fft::remap(
//                fft::Remap::H2FC, output_slice_rfft_weight, slice_shape);
//        const auto output_slice = noa::fft::c2r(output_slice_rfft, slice_shape);
//
//        const auto suffix = noa::string::format("{:.4f}.mrc", slice_z_radius);
//        noa::io::save(output_slice_rfft_weight_full,
//                      output_directory / noa::string::format("output_slice_ps_{}", suffix));
//        noa::io::save(output_slice, output_directory /  noa::string::format("output_slice_{}", suffix));
//
//        slice_z_radius *= 2;
//    }
//}
//
//TEST_CASE("unified::geometry::fft::insert_interpolate_and_extract_3d, test recons volume", "[.]") {
//    const auto stack_path = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf/tilt1_coarse_aligned.mrc");
//    const auto output_directory = Path("/home/thomas/Projects/quinoa/tests/ribo_ctf");
//
//    auto file = noa::io::ImageFile(stack_path, noa::io::READ);
//    const auto slice_shape = file.shape().set<0>(1);
//
//    // Load every image except the 0deg.
//    auto slice_0deg = noa::memory::empty<f32>(slice_shape);
//    file.read_slice(slice_0deg, 19);
//
//    // Input slices
//    const auto input_slice_rotation = noa::geometry::euler2matrix(
//            noa::math::deg2rad(Vec3<f32>{0, 35, 0}), "zyx", false);
//
//    // 3d rfft volume.
//    const auto volume_shape = slice_shape.set<1>(600);
//    const auto volume_rfft = noa::memory::zeros<c32>(volume_shape.rfft());
//    const auto volume_rfft_weight = noa::memory::like<f32>(volume_rfft);
//
//    // Output slice
//
//    auto slice_0deg_rfft = noa::fft::r2c(slice_0deg);
//    noa::fft::remap(fft::H2HC, slice_0deg_rfft, slice_0deg_rfft, slice_shape);
//
//    const auto rotation_center = slice_shape.vec().filter(2,3).as<f32>() / 2;
//    noa::signal::fft::phase_shift_2d<fft::HC2HC>(slice_0deg_rfft, slice_0deg_rfft, slice_shape, -rotation_center);
//
//    const auto max_output_size = static_cast<f32>(noa::math::min(slice_shape.filter(2, 3)));
//    auto slice_z_radius = 0.015f; // 1.f / max_output_size;
//    for (auto i: noa::irange(1)) {
//        noa::memory::fill(volume_rfft, c32{0});
//        noa::memory::fill(volume_rfft_weight, f32{0});
//
//        noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
//                slice_0deg_rfft, slice_shape,
//                volume_rfft, volume_shape,
//                Float22{}, input_slice_rotation,
//                slice_z_radius);
//        noa::geometry::fft::insert_interpolate_3d<fft::HC2H>(
//                1.f, slice_shape,
//                volume_rfft_weight, volume_shape,
//                Float22{}, input_slice_rotation,
//                slice_z_radius);
//
////        noa::ewise_binary(volume_rfft, volume_rfft_weight, volume_rfft,
////                          [](c32 lhs, f32 rhs) {
////                              if (rhs > 1)
////                                  lhs /= rhs;
////                              return lhs;
////                          });
//        noa::signal::fft::phase_shift_2d<fft::H2H>(volume_rfft, volume_rfft, volume_shape, rotation_center);
//
//        // Save PS.
//        const auto volume_weight_full = noa::fft::remap(
//                fft::Remap::H2FC, volume_rfft_weight, volume_shape);
//        const auto volume = noa::fft::c2r(volume_rfft, volume_shape);
//        noa::geometry::fft::gridding_correction(volume, volume, true);
//
//        const auto suffix = noa::string::format("{:.4f}.mrc", slice_z_radius);
//        noa::io::save(volume_weight_full,
//                      output_directory / noa::string::format("output_volume_weight_{}", suffix));
//        noa::io::save(volume, output_directory /  noa::string::format("output_volume_{}", suffix));
//
//        slice_z_radius *= 2;
//    }
//}
