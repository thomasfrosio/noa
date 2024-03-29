#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Subregion.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>
#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::memory::extract_subregions()", "[asset][noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["subregions"];

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (size_t nb = 0; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto shape = test["shape"].as<Shape4<i64>>();
            const auto subregion_shape = test["sub_shape"].as<Shape4<i64>>();
            const auto subregion_origins = test["sub_origins"].as<std::vector<Vec4<i64>>>();
            const auto border_mode = test["border"].as<BorderMode>();
            const auto border_value = test["border_value"].as<f32>();

            const auto input = noa::memory::arange<f32>(shape, 0, 1, options);
            const auto subregions = noa::memory::fill(subregion_shape, 4.f, options);
            const auto origins = View(subregion_origins.data(), subregion_origins.size()).to(options);

            // Extract:
            noa::memory::extract_subregions(input, subregions, origins, border_mode, border_value);

            const auto expected_subregion_filenames = test["expected_extract"].as<std::vector<Path>>();
            const auto subregion_count = static_cast<size_t>(subregion_shape[0]);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::ImageFile file;
                for (size_t i = 0; i < subregion_count; ++i) {
                    file.open(path_base / expected_subregion_filenames[i], io::WRITE);
                    file.write(subregions.subregion(i));
                }
            } else {
                const auto expected_subregions = noa::memory::like(subregions);
                noa::io::ImageFile file;
                for (size_t i = 0; i < subregion_count; ++i) {
                    file.open(path_base / expected_subregion_filenames[i], io::READ);
                    file.read(expected_subregions.subregion(i));
                }
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected_subregions, subregions, 1e-7));
            }

            // Insert:
            noa::memory::fill(input, 4.f);
            noa::memory::insert_subregions(subregions, input, origins);

            const auto expected_insert_filename = path_base / test["expected_insert"][0].as<Path>();
            if constexpr (COMPUTE_ASSETS) {
                noa::io::save(input, expected_insert_filename);
            } else {
                const auto expected_insert_back = noa::io::load_data<f32>(expected_insert_filename).reshape(input.shape());
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected_insert_back, input, 1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::memory::{extract|insert}_subregions()", "[noa][unified]",
                   i32, f32, f64, c32) {
    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const auto data = noa::math::random<TestType>(math::uniform_t{}, {2, 100, 200, 300}, -5, 5, options);
        const auto subregions = noa::memory::fill({3, 64, 64, 64}, TestType{0}, options);
        const auto origins = noa::memory::empty<Vec4<i64>>({1, 1, 1, 3}, options);
        origins(0, 0, 0, 0) = {0, 0, 0, 0};
        origins(0, 0, 0, 1) = {0, 34, 130, -20};
        origins(0, 0, 0, 2) = {1, 60, 128, 255};

        noa::memory::extract_subregions(data, subregions, origins);
        const Array result = data.to(options);
        noa::memory::insert_subregions(subregions, result, origins);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, data, result, 1e-7));
    }
}

TEMPLATE_TEST_CASE("unified::memory::atlas_layout(), insert_subregions()", "[noa][unified]", i32, f32) {
    const i64 ndim = GENERATE(2, 3);
    test::Randomizer<uint> dim_randomizer(40, 60);
    const auto subregion_shape = Shape4<i64>{
        test::Randomizer<i64>(10, 40).get(), // subregion count
        ndim == 3 ? dim_randomizer.get() : 1,
        dim_randomizer.get(),
        dim_randomizer.get()};

    // Prepare subregions.
    auto subregions = noa::memory::empty<TestType>(subregion_shape);
    for (i64 idx = 0; idx < subregion_shape[0]; ++idx)
        noa::memory::fill(subregions.subregion(idx), static_cast<TestType>(idx));
    // The loop could be replaced by: noa::memory::arange<TestType>(subregion_shape[0]).to(subregions);

    // Insert into atlas.
    auto [atlas_shape, atlas_origins] = noa::memory::atlas_layout(subregion_shape);
    const auto atlas_origins_1d = atlas_origins.accessor_contiguous_1d();
    for (i64 idx = 0; idx < subregion_shape[0]; ++idx)
        REQUIRE((atlas_origins_1d[idx][0] == 0 && atlas_origins_1d[idx][1] == 0));

    std::vector<Device> devices{Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        if (device != subregions.device())
            subregions = subregions.to(options);
        if (device != atlas_origins.device())
            atlas_origins = atlas_origins.to(options);

        const auto atlas = noa::memory::zeros<TestType>(atlas_shape, options);
        noa::memory::insert_subregions(subregions, atlas, atlas_origins);

        // Extract from atlas
        const auto o_subregions = noa::memory::like(subregions);
        noa::memory::extract_subregions(atlas, o_subregions, atlas_origins);

        REQUIRE(test::Matcher(test::MATCH_ABS, subregions, o_subregions, 1e-8));
    }
}
