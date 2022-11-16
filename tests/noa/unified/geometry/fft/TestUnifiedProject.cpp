#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Memory.h>
#include <noa/FFT.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
#include "Assets.h"

TEST_CASE("unified::geometry::fft::insert3D, by rasterisation", "[noa][unified]") {
    const path_t path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert3D_rasterisation"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<dim4_t>();
        const auto grid_shape = parameters["grid_shape"].as<dim4_t>();
        const auto target_shape = parameters["target_shape"].as<dim4_t>();
        const auto scale = float2_t(parameters["scale"].as<float>());
        const auto rotate = parameters["rotate"].as<std::vector<float>>();
        const auto cutoff = parameters["cutoff"].as<float>();
        const auto ews_radius = float2_t(parameters["ews_radius"].as<float>());
        const auto grid_filename = path / parameters["grid_filename"].as<path_t>();

        const float22_t inv_scaling_matrix = geometry::scale(1 / scale);
        Array<float33_t> fwd_rotation_matrices(rotate.size());
        for (size_t i = 0; i < rotate.size(); ++i)
            fwd_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, rotate[i], 0}), "ZYX");

        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            // Backward project.
            Array slice_fft = memory::linspace<float>(slice_shape.fft(), 1, 10, true, options);
            Array grid_fft = memory::zeros<float>(grid_shape.fft(), options);
            geometry::fft::insert3D<fft::HC2HC>(
                    slice_fft, slice_shape, grid_fft, grid_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    cutoff, target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                io::save(grid_fft, grid_filename);
            } else {
                Array asset_grid_fft = io::load<float>(grid_filename);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_grid_fft, grid_fft, 1e-5));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert3D, by rasterisation, remap", "[noa][unified]", float, cfloat_t) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = dim4_t{20, 1, 64, 64};
    const auto grid_shape = dim4_t{1, 128, 128, 128};

    Array<float33_t> fwd_rotation_matrices(slice_shape[0]);
    for (size_t i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, i * 2, 0}), "ZYX");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        Array slice_fft = memory::linspace<TestType>(slice_shape.fft(), 1, 10, true, options);
        Array grid_fft0 = memory::zeros<TestType>(grid_shape.fft(), options);
        Array grid_fft1 = grid_fft0.copy();
        Array grid_fft2 = grid_fft0.copy();

        // With centered slices.
        geometry::fft::insert3D<fft::HC2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        geometry::fft::insert3D<fft::HC2H>(
                slice_fft, slice_shape, grid_fft1, grid_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));

        // With non-centered slices.
        memory::fill(grid_fft0, TestType{0});
        memory::fill(grid_fft1, TestType{0});
        geometry::fft::insert3D<fft::H2HC>(
                slice_fft, slice_shape, grid_fft0, grid_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        geometry::fft::insert3D<fft::H2H>(
                slice_fft, slice_shape, grid_fft1, grid_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::insert3D, by interpolation", "[noa][unified]") {
    const path_t path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["insert3D_interpolation"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < 1; ++nb) { // tests["tests"].size()
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];
        const auto slice_shape = parameters["slice_shape"].as<dim4_t>();
        const auto grid_shape = parameters["grid_shape"].as<dim4_t>();
        const auto target_shape = parameters["target_shape"].as<dim4_t>();
        const auto scale = float2_t(parameters["scale"].as<float>());
        const auto rotate = parameters["rotate"].as<std::vector<float>>();
        const auto cutoff = parameters["cutoff"].as<float>();
        const auto ews_radius = float2_t(parameters["ews_radius"].as<float>());
        const auto slice_z_radius = parameters["slice_z_radius"].as<float>();
        const auto grid_filename = path / parameters["grid_filename"].as<path_t>();

        const float22_t fwd_scaling_matrix = geometry::scale(scale);
        Array<float33_t> inv_rotation_matrices(rotate.size());
        for (size_t i = 0; i < rotate.size(); ++i)
            inv_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, -rotate[i], 0}), "ZYX");

        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);
            INFO(device);

            if (inv_rotation_matrices.device() != device)
                inv_rotation_matrices = inv_rotation_matrices.to(device);

            // Backward project.
            Array slice_fft = memory::linspace<float>(slice_shape.fft(), 1, 10, true, options);
            Array grid_fft = memory::zeros<float>(grid_shape.fft(), options);
            geometry::fft::insert3D<fft::HC2HC>(
                    slice_fft.eval(), slice_shape, grid_fft, grid_shape,
                    fwd_scaling_matrix, inv_rotation_matrices,
                    slice_z_radius, cutoff, target_shape, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                io::save(grid_fft, grid_filename);
                continue;
            }

            Array asset_grid_fft = io::load<float>(grid_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_grid_fft, grid_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::insert3D, by interpolation using texture API and remap", "[noa][unified]",
                   float, cfloat_t) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const dim_t slice_count = GENERATE(as<dim_t>(), 1, 20);
    const auto slice_shape = dim4_t{slice_count, 1, 64, 64};
    const auto grid_shape = dim4_t{1, 128, 128, 128};
    const auto slice_z_radius = 0.02f;

    Array<float33_t> inv_rotation_matrices(slice_shape[0]);
    for (size_t i = 0; i < slice_shape[0]; ++i) {
        const auto angle = static_cast<float>(i) * 2;
        inv_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, -angle, 0}), "ZYX");
    }

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        if (inv_rotation_matrices.device() != device)
            inv_rotation_matrices = inv_rotation_matrices.to(device);

        Array slice_fft = memory::linspace<TestType>(slice_shape.fft(), 1, 10, true, options);
        Array grid_fft0 = memory::zeros<TestType>(grid_shape.fft(), options);
        Array grid_fft1 = grid_fft0.copy();
        Array grid_fft2 = grid_fft0.copy();

        { // Texture
            Texture<TestType> texture_slice_fft(slice_fft, device, INTERP_LINEAR, BORDER_ZERO, 0.f, true);
            geometry::fft::insert3D<fft::HC2H>(
                    slice_fft, slice_shape, grid_fft0, grid_shape,
                    float22_t{}, inv_rotation_matrices, slice_z_radius, 0.45f);
            geometry::fft::insert3D<fft::HC2H>(
                    texture_slice_fft, slice_shape, grid_fft1, grid_shape,
                    float22_t{}, inv_rotation_matrices, slice_z_radius, 0.45f);

            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft1, 5e-5));
        }

        { // Remap
            memory::fill(grid_fft0, TestType{0});
            memory::fill(grid_fft1, TestType{0});
            geometry::fft::insert3D<fft::HC2HC>(
                    slice_fft, slice_shape, grid_fft0, grid_shape,
                    float22_t{}, inv_rotation_matrices, slice_z_radius, 0.5f);
            geometry::fft::insert3D<fft::HC2H>(
                    slice_fft, slice_shape, grid_fft1, grid_shape,
                    float22_t{}, inv_rotation_matrices, slice_z_radius, 0.5f);
            fft::remap(fft::H2HC, grid_fft1, grid_fft2, grid_shape);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, grid_fft0, grid_fft2, 5e-5));
        }
    }
}

TEST_CASE("unified::geometry::fft::extract3D from grid", "[noa][unified]") {
    const path_t path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["extract3D_from_grid"];

    const auto grid_filename = path / tests["grid_filename"].as<path_t>();
    const auto grid_shape = tests["grid_shape"].as<dim4_t>();

    constexpr bool COMPUTE_ASSETS = false;
    if constexpr (COMPUTE_ASSETS) {
        Array<float33_t> insert_fwd_rotation_matrices(36);
        for (size_t i = 0; i < insert_fwd_rotation_matrices.size(); ++i) {
            const float3_t euler_angles = math::deg2rad(float3_t{90, 0, i * 10});
            insert_fwd_rotation_matrices[i] = geometry::euler2matrix(euler_angles, "XYZ", false);
        }
        dim4_t insert_slice_shape{36, 1, grid_shape[2], grid_shape[3]};
        Array grid_fft = memory::empty<float>(grid_shape.fft());

        Array slice_fft = memory::linspace<float>(insert_slice_shape.fft(), 1, 20);
        geometry::fft::insert3D<fft::HC2HC>(
                slice_fft, insert_slice_shape, grid_fft, grid_shape,
                float22_t{}, insert_fwd_rotation_matrices, 0.01f, 0.5f);
        io::save(grid_fft, grid_filename);
    }

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto slice_shape = parameters["slice_shape"].as<dim4_t>();
        const auto scale = float2_t(parameters["scale"].as<float>());
        const auto rotate = parameters["rotate"].as<std::vector<float>>();
        const auto cutoff = parameters["cutoff"].as<float>();
        const auto ews_radius = float2_t(parameters["ews_radius"].as<float>());
        const auto slice_filename = path / parameters["slice_filename"].as<path_t>();

        const float22_t inv_scaling_matrix = geometry::scale(1 / scale);
        Array<float33_t> fwd_rotation_matrices(rotate.size());
        for (size_t i = 0; i < rotate.size(); ++i)
            fwd_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, rotate[i], 0}), "ZYX");

        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);
            INFO(device);

            if (fwd_rotation_matrices.device() != device)
                fwd_rotation_matrices = fwd_rotation_matrices.to(device);

            Array grid_fft = io::load<float>(grid_filename, false, options);
            Array slice_fft = memory::empty<float>(slice_shape.fft(), options);

            // Forward project.
            geometry::fft::extract3D<fft::HC2HC>(
                    grid_fft, grid_shape, slice_fft, slice_shape,
                    inv_scaling_matrix, fwd_rotation_matrices,
                    cutoff, {}, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                io::save(slice_fft, slice_filename);
                continue;
            }

            Array asset_slice_fft = io::load<float>(slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_slice_fft, slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::extract3D from grid, using texture API and remap", "[noa][unified]",
                   float, cfloat_t) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const auto slice_shape = dim4_t{20, 1, 128, 128};
    const auto grid_shape = dim4_t{1, 128, 128, 128};

    Array<float33_t> fwd_rotation_matrices(slice_shape[0]);
    for (size_t i = 0; i < slice_shape[0]; ++i)
        fwd_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, i * 2, 0}), "ZYX");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        if (fwd_rotation_matrices.device() != device)
            fwd_rotation_matrices = fwd_rotation_matrices.to(device);

        Array grid_fft = memory::linspace<TestType>(grid_shape.fft(), -50, 50, true, options);
        Array slice_fft0 = memory::empty<TestType>(slice_shape.fft(), options);
        Array slice_fft1 = slice_fft0.copy();
        Array slice_fft2 = slice_fft0.copy();

        Texture<TestType> texture_grid_fft(grid_fft, device, INTERP_LINEAR, BORDER_ZERO, 0.f);
        geometry::fft::extract3D<fft::HC2HC>(
                texture_grid_fft, grid_shape, slice_fft0, slice_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        geometry::fft::extract3D<fft::HC2H>(
                grid_fft, grid_shape, slice_fft1, slice_shape,
                float22_t{}, fwd_rotation_matrices, 0.45f);
        fft::remap(fft::H2HC, slice_fft1, slice_fft2, slice_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, slice_fft0, slice_fft2, 5e-5));
    }
}

TEST_CASE("unified::geometry::fft::extract3D from slices", "[noa][unified]") {
    const path_t path = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["extract3D_from_slices"];
    constexpr bool COMPUTE_ASSETS = false;

    std::vector<Device> devices = {Device("cpu")};
    if (!COMPUTE_ASSETS && Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (size_t nb = 0; nb < tests["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& parameters = tests["tests"][nb];

        const auto input_slice_shape = parameters["input_slice_shape"].as<dim4_t>();
        const auto output_slice_shape = parameters["output_slice_shape"].as<dim4_t>();
        const auto input_rotate = parameters["input_rotate"].as<std::vector<float>>();
        const auto output_rotate = parameters["output_rotate"].as<std::vector<float>>();
        const auto cutoff = parameters["cutoff"].as<float>();
        const auto ews_radius = float2_t(parameters["ews_radius"].as<float>());
        const auto slice_z_radius = parameters["slice_z_radius"].as<float>();
        const auto output_slice_filename = path / parameters["output_slice_filename"].as<path_t>();

        Array<float33_t> input_inv_rotation_matrices(input_rotate.size());
        Array<float33_t> output_fwd_rotation_matrices(output_rotate.size());
        for (size_t i = 0; i < input_inv_rotation_matrices.size(); ++i) {
            input_inv_rotation_matrices[i] =
                    geometry::euler2matrix(math::deg2rad(float3_t{0, -input_rotate[i], 0}),"ZYX");
        }
        for (size_t i = 0; i < output_fwd_rotation_matrices.size(); ++i) {
            output_fwd_rotation_matrices[i] =
                    geometry::euler2matrix(math::deg2rad(float3_t{0, output_rotate[i], 0}),"ZYX");
        }

        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);
            INFO(device);

            if (input_inv_rotation_matrices.device() != device)
                input_inv_rotation_matrices = input_inv_rotation_matrices.to(device);
            if (output_fwd_rotation_matrices.device() != device)
                output_fwd_rotation_matrices = output_fwd_rotation_matrices.to(device);

            Array input_slice_fft = memory::linspace<float>(input_slice_shape.fft(), -50, 50, true, options);
            Array output_slice_fft = memory::empty<float>(output_slice_shape.fft(), options);

            geometry::fft::extract3D<fft::HC2HC>(
                    input_slice_fft, input_slice_shape,
                    output_slice_fft, output_slice_shape,
                    float22_t{}, input_inv_rotation_matrices,
                    float22_t{}, output_fwd_rotation_matrices,
                    slice_z_radius, cutoff, ews_radius);

            if constexpr (COMPUTE_ASSETS) {
                io::save(output_slice_fft, output_slice_filename);
                continue;
            }

            Array asset_output_slice_fft = io::load<float>(output_slice_filename);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, asset_output_slice_fft, output_slice_fft, 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::geometry::fft::extract3D from slices, using texture API and remap", "[noa][unified]",
                   float, cfloat_t) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const auto input_slice_shape = dim4_t{20, 1, 256, 256};
    const auto output_slice_shape = dim4_t{5, 1, 256, 256};

    Array<float33_t> input_inv_rotation_matrices(input_slice_shape[0]);
    for (size_t i = 0; i < input_slice_shape[0]; ++i) {
        const auto angle = static_cast<float>(i) * 2;
        input_inv_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, -angle, 0}), "ZYX");
    }
    Array<float33_t> output_fwd_rotation_matrices(output_slice_shape[0]);
    for (size_t i = 0; i < output_slice_shape[0]; ++i) {
        const auto angle = static_cast<float>(i) * 2 + 1;
        output_fwd_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, angle, 0}), "ZYX");
    }

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        if (input_inv_rotation_matrices.device() != device)
            input_inv_rotation_matrices = input_inv_rotation_matrices.to(device);
        if (output_fwd_rotation_matrices.device() != device)
            output_fwd_rotation_matrices = output_fwd_rotation_matrices.to(device);

        Array input_slice_fft = memory::linspace<TestType>(input_slice_shape.fft(), -50, 50, true, options);
        Texture<TestType> texture_input_slice_fft(input_slice_fft, device, INTERP_LINEAR, BORDER_ZERO, 0.f, true);
        Array output_slice_fft0 = memory::empty<TestType>(output_slice_shape.fft(), options);
        Array output_slice_fft1 = memory::like(output_slice_fft0);
        Array output_slice_fft2 = memory::like(output_slice_fft0);

        geometry::fft::extract3D<fft::HC2HC>(
                texture_input_slice_fft, input_slice_shape,
                output_slice_fft0, output_slice_shape,
                float22_t{}, input_inv_rotation_matrices,
                float22_t{}, output_fwd_rotation_matrices,
                0.01f, 0.8f);

        geometry::fft::extract3D<fft::HC2H>(
                input_slice_fft, input_slice_shape,
                output_slice_fft1, output_slice_shape,
                float22_t{}, input_inv_rotation_matrices,
                float22_t{}, output_fwd_rotation_matrices,
                0.01f, 0.8f);
        fft::remap(fft::H2HC, output_slice_fft1, output_slice_fft2, output_slice_shape);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output_slice_fft0, output_slice_fft2, 5e-5));
    }
}
