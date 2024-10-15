#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Assets.h"
#include "Utils.hpp"

using namespace ::noa::types;
using Remap = noa::Remap;
using Path = std::filesystem::path;

TEST_CASE("unified::fft::(i)fftshift -- vs numpy", "[asset][noa][unified]") {
    const Path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["remap"];

    const std::array<std::string, 2> keys{"2D", "3D"};

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        for (const auto& key: keys) {
            const auto array = noa::io::read_data<f32>(path / tests[key]["input"].as<Path>(), {}, options);

            // fftshift
            auto reordered_expected = noa::io::read_data<f32>(path / tests[key]["fftshift"].as<Path>(), {}, options);
            auto reordered_results = noa::fft::remap("f2fc", array, reordered_expected.shape());
            REQUIRE(test::allclose_abs_safe(reordered_expected, reordered_results, 1e-10));

            // ifftshift
            reordered_expected = noa::io::read_data<f32>(path / tests[key]["ifftshift"].as<Path>(), {}, options);
            reordered_results = noa::fft::remap("fc2f", array, reordered_expected.shape());
            REQUIRE(test::allclose_abs_safe(reordered_expected, reordered_results, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap()", "[noa][unified]", f32, f64, c32, c64) {
    const i64 ndim = GENERATE(1, 2, 3);
    INFO("ndim: " << ndim);

    const auto shape = test::random_shape_batched(ndim);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        AND_THEN("h2hc, in-place") {
            const auto shape_even = test::random_shape_batched(ndim, {.only_even_sizes = true});
            const auto half_in = noa::random(noa::Uniform<TestType>{-5, 5}, shape_even.rfft(), options);
            const auto half_out = noa::fft::remap("h2hc", half_in, shape_even);
            noa::fft::remap(Remap::H2HC, half_in, half_in, shape_even);
            REQUIRE(test::allclose_abs(half_in, half_out, 1e-10));
        }

        const Array input_full = noa::random(noa::Uniform<TestType>{-5, 5}, shape, options);
        const Array input_half = noa::random(noa::Uniform<TestType>{-5, 5}, shape.rfft(), options);

        AND_THEN("fc->f->fc") {
            const auto full = noa::fft::remap(Remap::FC2F, input_full, shape);
            const auto full_centered_out = noa::fft::remap(Remap::F2FC, full, shape);
            REQUIRE(test::allclose_abs(input_full, full_centered_out, 1e-10));
        }

        AND_THEN("f->fc->f") {
            const auto full_centered = noa::fft::remap(Remap::F2FC, input_full, shape);
            const auto full_out = noa::fft::remap(Remap::FC2F, full_centered, shape);
            REQUIRE(test::allclose_abs(input_full, full_out, 1e-10));
        }

        AND_THEN("(f->h->hc) vs (f->hc)") {
            const auto expected_half_centered = noa::fft::remap(Remap::F2HC, input_full, shape);
            const auto half_ = noa::fft::remap(Remap::F2H, input_full, shape);
            const auto result_half_centered = noa::fft::remap(Remap::H2HC, half_, shape);
            REQUIRE(test::allclose_abs(expected_half_centered, result_half_centered, 1e-10));
        }

        AND_THEN("hc->h->hc") {
            const auto half_ = noa::fft::remap(Remap::HC2H, input_half, shape);
            const auto half_centered_out = noa::fft::remap(Remap::H2HC, half_, shape);
            REQUIRE(test::allclose_abs(input_half, half_centered_out, 1e-10));
        }

        AND_THEN("h->hc->h") {
            const auto half_centered = noa::fft::remap(Remap::H2HC, input_half, shape);
            const auto half_out = noa::fft::remap(Remap::HC2H, half_centered, shape);
            REQUIRE(test::allclose_abs(input_half, half_out, 1e-10));
        }

        AND_THEN("h->f->h") {
            const auto full = noa::fft::remap(Remap::H2F, input_half, shape);
            const auto half_ = noa::fft::remap(Remap::F2H, full, shape);
            REQUIRE(test::allclose_abs(input_half, half_, 1e-10));
        }

        AND_THEN("(h->hc->f) vs (h->f)") {
            const auto expected_full = noa::fft::remap(Remap::H2F, input_half, shape);
            const auto half_centered = noa::fft::remap(Remap::H2HC, input_half, shape);
            const auto result_full = noa::fft::remap(Remap::HC2F, half_centered, shape);
            REQUIRE(test::allclose_abs(expected_full, result_full, 1e-10));
        }

        AND_THEN("(hc->fc) vs (hc->f->fc)") {
            const auto full_centered_result = noa::fft::remap(Remap::HC2FC, input_half, shape);
            const auto full = noa::fft::remap(Remap::HC2F, input_half, shape);
            const auto full_centered_expected = noa::fft::remap(Remap::F2FC, full, shape);
            REQUIRE(test::allclose_abs(full_centered_result, full_centered_expected, 1e-10));
        }

        AND_THEN("(h->fc) vs (h->f->fc)") {
            const auto full_centered_result = noa::fft::remap(Remap::H2FC, input_half, shape);
            const auto full = noa::fft::remap(Remap::H2F, input_half, shape);
            const auto full_centered_expected = noa::fft::remap(Remap::F2FC, full, shape);
            REQUIRE(test::allclose_abs(full_centered_result, full_centered_expected, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::remap(), cpu vs gpu", "[noa][unified]", f32, f64, c32, c64) {
    if (not Device::is_any_gpu())
        return;

    const i64 ndim = GENERATE(1, 2, 3);
    const Remap remap = GENERATE(
        Remap::H2H, Remap::HC2HC, Remap::F2F, Remap::FC2FC, Remap::H2HC,
        Remap::HC2H, Remap::H2F, Remap::F2H, Remap::F2FC, Remap::FC2F, Remap::HC2F,
        Remap::F2HC, Remap::FC2H, Remap::FC2HC, Remap::HC2FC, Remap::H2FC);

    INFO("ndim: " << ndim);
    INFO("remap: " << remap);

    const auto shape = test::random_shape_batched(ndim);
    const auto input_shape = remap.is_hx2xx() ? shape.rfft() : shape;

    const auto input_cpu = noa::random(noa::Uniform<TestType>{-5, 5}, input_shape);
    const auto input_gpu = input_cpu.to({.device="gpu", .allocator=Allocator::PITCHED});

    const auto output_cpu = noa::fft::remap(remap, input_cpu, shape);
    const auto output_gpu = noa::fft::remap(remap, input_gpu, shape);
    REQUIRE(test::allclose_abs(output_cpu, output_gpu.to_cpu(), 1e-10));
}
