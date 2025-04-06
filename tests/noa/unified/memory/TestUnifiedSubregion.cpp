#include <noa/unified/Random.hpp>
#include <noa/unified/Subregion.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEST_CASE("unified::extract_subregions()", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    const Path path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["subregions"];

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (size_t nb{}; nb < tests.size(); ++nb) {
            INFO("test number = " << nb);

            const YAML::Node& test = tests[nb];
            const auto shape = test["shape"].as<Shape4<i64>>();
            const auto subregion_shape = test["sub_shape"].as<Shape4<i64>>();
            const auto subregion_origins = test["sub_origins"].as<std::vector<Vec4<i64>>>();
            const auto border_mode = test["border"].as<noa::Border>();
            const auto border_value = test["border_value"].as<f32>();

            const auto input = noa::arange(shape, noa::Arange<f32>{}, options);
            const auto subregions = noa::fill(subregion_shape, 4.f, options);
            const auto origins = View(subregion_origins.data(), static_cast<i64>(subregion_origins.size())).to(options);

            // Extract:
            noa::extract_subregions(input, subregions, origins, border_mode, border_value);
            subregions.eval();

            const auto expected_subregion_filenames = test["expected_extract"].as<std::vector<Path>>();
            const auto subregion_count = static_cast<size_t>(subregion_shape[0]);
            if constexpr (COMPUTE_ASSETS) {
                noa::io::ImageFile file;
                for (size_t i{}; i < subregion_count; ++i) {
                    file.open(path_base / expected_subregion_filenames[i], {.write=true});
                    file.write_all(subregions.span().as_const().subregion(i));
                }
            } else {
                const auto expected_subregions = noa::like(subregions);
                noa::io::ImageFile file;
                for (size_t i{}; i < subregion_count; ++i) {
                    file.open(path_base / expected_subregion_filenames[i], {.read=true});
                    file.read_all(expected_subregions.span().subregion(i));
                }
                REQUIRE(test::allclose_abs_safe(expected_subregions, subregions, 1e-7));
            }

            // Insert:
            noa::fill(input, 4.f);
            noa::insert_subregions(subregions, input, origins);

            const auto expected_insert_filename = path_base / test["expected_insert"][0].as<Path>();
            if constexpr (COMPUTE_ASSETS) {
                noa::write(input, expected_insert_filename);
            } else {
                const auto expected_insert_back = noa::io::read_data<f32>(expected_insert_filename).reshape(input.shape());
                REQUIRE(test::allclose_abs_safe(expected_insert_back, input, 1e-7));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::extract|insert_subregions()", "", i32, f32, f64, c32) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const auto data = noa::random(noa::Uniform<TestType>{-5, 5}, {2, 100, 200, 300}, options);
        const auto subregions = noa::fill({3, 64, 64, 64}, TestType{0}, options);
        const auto origins = noa::empty<Vec4<i64>>({1, 1, 1, 3}, options);
        origins(0, 0, 0, 0) = {0, 0, 0, 0};
        origins(0, 0, 0, 1) = {0, 34, 130, -20};
        origins(0, 0, 0, 2) = {1, 60, 128, 255};

        noa::extract_subregions(data, subregions, origins);
        const Array result = data.to(options);
        noa::insert_subregions(subregions, result, origins);

        REQUIRE(test::allclose_abs_safe(data, result, 1e-7));
    }
}

TEMPLATE_TEST_CASE("unified::atlas_layout(), insert_subregions()", "", i32, f32) {
    const i64 ndim = GENERATE(2, 3);
    test::Randomizer<u32> dim_randomizer(40, 60);
    const auto subregion_shape = Shape4<i64>{
        test::Randomizer<i64>(10, 40).get(), // subregion count
        ndim == 3 ? dim_randomizer.get() : 1,
        dim_randomizer.get(),
        dim_randomizer.get()};

    // Prepare subregions.
    auto subregions = noa::empty<TestType>(subregion_shape);
    for (i64 idx{}; idx < subregion_shape[0]; ++idx)
        noa::fill(subregions.subregion(idx), static_cast<TestType>(idx));
    // The loop could be replaced by: noa::arange<TestType>(subregion_shape[0]).flat(0).to(subregions);

    auto [atlas_shape, atlas_origins] = noa::atlas_layout<i64, 2>(subregion_shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        if (device != subregions.device())
            subregions = subregions.to(options);
        if (device != atlas_origins.device())
            atlas_origins = atlas_origins.to(options);

        const auto atlas = noa::zeros<TestType>(atlas_shape, options);
        noa::insert_subregions(subregions, atlas, atlas_origins);

        // Extract from atlas
        const auto o_subregions = noa::like(subregions);
        noa::extract_subregions(atlas, o_subregions, atlas_origins);

        REQUIRE(test::allclose_abs(subregions, o_subregions, 1e-8));
    }
}
