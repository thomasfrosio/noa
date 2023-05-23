#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Helpers.h"
using namespace ::noa;

TEST_CASE("unified::fft::fc2f(), f2fc() -- vs numpy", "[asset][noa][unified]") {
    const fs::path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["remap"];

    const std::array<std::string, 2> keys = {"2D", "3D"};

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (const auto& key: keys) {
            const auto array = noa::io::load_data<f32>(path / tests[key]["input"].as<Path>(), false, options);

            // fftshift
            auto reordered_expected = noa::io::load_data<f32>(path / tests[key]["fftshift"].as<Path>(), false, options);
            auto reordered_results = noa::fft::remap(fft::F2FC, array, reordered_expected.shape());
            REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected, reordered_results, 1e-10));

            // ifftshift
            reordered_expected = noa::io::load_data<f32>(path / tests[key]["ifftshift"].as<Path>(), false, options);
            reordered_results = noa::fft::remap(fft::FC2F, array, reordered_expected.shape());
            REQUIRE(test::Matcher(test::MATCH_ABS, reordered_expected, reordered_results, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap()", "[noa][unified]", f16, f32, f64, c16, c32, c64) {
    const i64 ndim = GENERATE(1, 2, 3);
    INFO("ndim: " << ndim);

    const auto shape = test::get_random_shape4_batched(ndim);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        AND_THEN("h2hc, in-place") {
            const auto shape_even = test::get_random_shape4_batched(ndim, /*even=*/ true);
            const auto half_in = noa::math::random<TestType>(noa::math::uniform_t{}, shape_even.rfft(), -5, 5, options);

            const auto half_out = noa::fft::remap(fft::H2HC, half_in, shape_even);
            noa::fft::remap(fft::H2HC, half_in, half_in, shape_even);
            REQUIRE(test::Matcher(test::MATCH_ABS, half_in, half_out, 1e-10));
        }

        const Array input_full = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, options);
        const Array input_half = noa::math::random<TestType>(noa::math::uniform_t{}, shape.rfft(), -5, 5, options);

        AND_THEN("fc->f->fc") {
            const auto full = noa::fft::remap(fft::FC2F, input_full, shape);
            const auto full_centered_out = noa::fft::remap(fft::F2FC, full, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, input_full, full_centered_out, 1e-10));
        }

        AND_THEN("f->fc->f") {
            const auto full_centered = noa::fft::remap(fft::F2FC, input_full, shape);
            const auto full_out = noa::fft::remap(fft::FC2F, full_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, input_full, full_out, 1e-10));
        }

        AND_THEN("(f->h->hc) vs (f->hc)") {
            const auto expected_half_centered = noa::fft::remap(fft::F2HC, input_full, shape);
            const auto half_ = noa::fft::remap(fft::F2H, input_full, shape);
            const auto result_half_centered = noa::fft::remap(fft::H2HC, half_, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_half_centered, result_half_centered, 1e-10));
        }

        AND_THEN("hc->h->hc") {
            const auto half_ = noa::fft::remap(fft::HC2H, input_half, shape);
            const auto half_centered_out = noa::fft::remap(fft::H2HC, half_, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, input_half, half_centered_out, 1e-10));
        }

        AND_THEN("h->hc->h") {
            const auto half_centered = noa::fft::remap(fft::H2HC, input_half, shape);
            const auto half_out = noa::fft::remap(fft::HC2H, half_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, input_half, half_out, 1e-10));
        }

        AND_THEN("h->f->h") {
            const auto full = noa::fft::remap(fft::H2F, input_half, shape);
            const auto half_ = noa::fft::remap(fft::F2H, full, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, input_half, half_, 1e-10));
        }

        AND_THEN("(h->hc->f) vs (h->f)") {
            const auto expected_full = noa::fft::remap(fft::H2F, input_half, shape);
            const auto half_centered = noa::fft::remap(fft::H2HC, input_half, shape);
            const auto result_full = noa::fft::remap(fft::HC2F, half_centered, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_full, result_full, 1e-10));
        }

        AND_THEN("(hc->fc) vs (hc->f->fc)") {
            const auto full_centered_result = noa::fft::remap(fft::HC2FC, input_half, shape);
            const auto full = noa::fft::remap(fft::HC2F, input_half, shape);
            const auto full_centered_expected = noa::fft::remap(fft::F2FC, full, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, full_centered_result, full_centered_expected, 1e-10));
        }

        AND_THEN("(h->fc) vs (h->f->fc)") {
            const auto full_centered_result = noa::fft::remap(fft::H2FC, input_half, shape);
            const auto full = noa::fft::remap(fft::H2F, input_half, shape);
            const auto full_centered_expected = noa::fft::remap(fft::F2FC, full, shape);
            REQUIRE(test::Matcher(test::MATCH_ABS, full_centered_result, full_centered_expected, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap(), cpu vs gpu", "[noa][unified]", f16, f32, f64, c16, c32, c64) {
    if (!Device::is_any(DeviceType::GPU))
        return;

    const i64 ndim = GENERATE(1, 2, 3);
    const auto remap = GENERATE(as<noa::fft::Remap>(),
            noa::fft::H2H, noa::fft::HC2HC, noa::fft::F2F, noa::fft::FC2FC, noa::fft::H2HC,
            noa::fft::HC2H, noa::fft::H2F, noa::fft::F2H, noa::fft::F2FC, noa::fft::FC2F, noa::fft::HC2F,
            noa::fft::F2HC, noa::fft::FC2H, noa::fft::FC2HC, noa::fft::HC2FC, noa::fft::H2FC);

    INFO("ndim: " << ndim);
    INFO("remap: " << remap);

    const auto shape = test::get_random_shape4_batched(ndim);
    const auto input_shape = noa::traits::to_underlying(remap) & noa::fft::SRC_HALF ? shape.rfft() : shape;

    const auto gpu_options = ArrayOption(Device("gpu"), Allocator::PITCHED);
    const auto input_cpu = noa::math::random<TestType>(noa::math::uniform_t{}, input_shape, -5, 5);
    const auto input_gpu = input_cpu.to(gpu_options);

    const auto output_cpu = noa::fft::remap(remap, input_cpu, shape);
    const auto output_gpu = noa::fft::remap(remap, input_gpu, shape);
    REQUIRE(test::Matcher(test::MATCH_ABS, output_cpu, output_gpu.to_cpu(), 1e-10));
}
